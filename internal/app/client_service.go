package app

import (
	"context"
	"errors"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/bridge"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

type ClientService struct {
	client         pb.AuthorityServiceClient
	oprfClient     *crypto.OPRFClient
	mldsaPublicKey *mldsa65.PublicKey
	recursionSteps int
	epochMaxSkew   uint64
	requestTimeout time.Duration
}

func NewClientService(client pb.AuthorityServiceClient, recursionSteps int, mldsaPublicKey *mldsa65.PublicKey) *ClientService {
	return &ClientService{
		client:         client,
		oprfClient:     crypto.NewOPRFClient(),
		mldsaPublicKey: mldsaPublicKey,
		recursionSteps: recursionSteps,
		epochMaxSkew:   1,
		requestTimeout: 3 * time.Second,
	}
}

// ScanImage runs the end-to-end client flow with placeholder cryptography.
func (s *ClientService) ScanImage(ctx context.Context, imagePath, modelVersion string) error {
	imgBytes, err := bridge.LoadImage(imagePath)
	if err != nil {
		return err
	}
	hashVec := bridge.RecursiveInference(imgBytes, s.recursionSteps)
	latticePoint := bridge.MapToLattice(hashVec)

	state, blindedRequest, err := s.oprfClient.Blind(latticePoint.Marshal())
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, s.requestTimeout)
	defer cancel()

	resp, err := s.client.CheckImage(ctx, &pb.BlindCheckRequest{
		BlindedElement: blindedRequest,
		ModelVersion:   modelVersion,
	})
	if err != nil {
		return err
	}

	if !crypto.IsEpochFresh(resp.EpochId, time.Now(), s.epochMaxSkew) {
		return errors.New("stale epoch proof")
	}
	if !crypto.VerifySplitAccumulation(resp.ProofInstance, resp.MembershipProof) {
		return errors.New("split accumulator verification failed")
	}

	sigPayload := crypto.ProofSignaturePayload(resp.ProofInstance, resp.BlindedSignature)
	if !crypto.VerifyMLDSA(s.mldsaPublicKey, sigPayload, resp.MembershipProof) {
		return errors.New("invalid proof signature")
	}

	oprfOutput, err := s.oprfClient.Finalize(state, resp.BlindedSignature)
	if err != nil {
		return err
	}
	if len(oprfOutput) == 0 {
		return nil
	}

	shareData, shareIndex, metadata, err := crypto.NewDummyShare()
	if err != nil {
		return err
	}
	if len(resp.UploadToken) == 0 {
		return errors.New("missing upload token")
	}

	_, err = s.client.UploadShare(ctx, &pb.VoucherShare{
		ShareData:         shareData,
		ShareIndex:        shareIndex,
		EncryptedMetadata: metadata,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: []byte("TODO-attestation"),
	})
	return err
}
