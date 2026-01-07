package app

import (
	"context"
	"encoding/base64"
	"sync"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const proofVersion = 1

type ServerService struct {
	oprfServer     *crypto.OPRFServer
	mldsaPrivateKey *mldsa65.PrivateKey
	store          storage.ProofStore
	tokenTTL       time.Duration
	tokens         map[string]time.Time
	tokensMu       sync.Mutex
}

func NewServerService(oprfServer *crypto.OPRFServer, mldsaPrivateKey *mldsa65.PrivateKey, store storage.ProofStore, tokenTTL time.Duration) *ServerService {
	return &ServerService{
		oprfServer:     oprfServer,
		mldsaPrivateKey: mldsaPrivateKey,
		store:          store,
		tokenTTL:       tokenTTL,
		tokens:         make(map[string]time.Time),
	}
}

func (s *ServerService) HandleCheckImage(ctx context.Context, req *pb.BlindCheckRequest) (*pb.BlindCheckResponse, error) {
	if req.GetModelVersion() == "" {
		return nil, status.Error(codes.FailedPrecondition, "missing model_version")
	}
	if len(req.GetBlindedElement()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "empty blinded_element")
	}

	epoch := crypto.CurrentEpochID(time.Now())
	proofInstance, _, err := s.store.GetProofBundle(epoch)
	if err != nil {
		return nil, status.Error(codes.Internal, "proof generation failed")
	}

	eval, err := s.oprfServer.Evaluate(req.BlindedElement)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "invalid oprf request")
	}
	sigPayload := crypto.ProofSignaturePayload(proofInstance, eval)
	proofSig, err := crypto.SignMLDSA(s.mldsaPrivateKey, sigPayload)
	if err != nil {
		return nil, status.Error(codes.Internal, "proof signing failed")
	}
	token, err := s.issueToken()
	if err != nil {
		return nil, status.Error(codes.Internal, "token generation failed")
	}

	return &pb.BlindCheckResponse{
		BlindedSignature: eval,
		ProofInstance:    proofInstance,
		MembershipProof:  proofSig,
		EpochId:          epoch,
		ProofVersion:     proofVersion,
		UploadToken:      token,
	}, nil
}

func (s *ServerService) HandleUploadShare(ctx context.Context, req *pb.VoucherShare) (*pb.ShareResponse, error) {
	if len(req.GetUploadToken()) == 0 || len(req.GetDeviceAttestation()) == 0 {
		return &pb.ShareResponse{Status: pb.ShareResponse_REJECTED}, nil
	}
	if !s.consumeToken(req.GetUploadToken()) {
		return &pb.ShareResponse{Status: pb.ShareResponse_REJECTED}, nil
	}
	return &pb.ShareResponse{Status: pb.ShareResponse_ACCEPTED}, nil
}

func (s *ServerService) issueToken() ([]byte, error) {
	token, err := crypto.NewUploadToken()
	if err != nil {
		return nil, err
	}
	s.tokensMu.Lock()
	defer s.tokensMu.Unlock()
	s.tokens[base64.StdEncoding.EncodeToString(token)] = time.Now().Add(s.tokenTTL)
	return token, nil
}

func (s *ServerService) consumeToken(token []byte) bool {
	s.tokensMu.Lock()
	defer s.tokensMu.Unlock()

	key := base64.StdEncoding.EncodeToString(token)
	expiry, ok := s.tokens[key]
	if !ok {
		return false
	}
	delete(s.tokens, key)
	return time.Now().Before(expiry)
}
