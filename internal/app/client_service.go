package app

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/bridge"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

var (
	ErrStaleEpoch           = errors.New("stale epoch proof")
	ErrAccumulatorInvalid   = errors.New("split accumulator verification failed")
	ErrSignatureInvalid     = errors.New("invalid proof signature")
	ErrMissingUploadToken   = errors.New("missing upload token")
	ErrOPRFFailed           = errors.New("OPRF finalization failed")
	ErrProofVersionMismatch = errors.New("unsupported proof version")
)

const (
	supportedProofVersion = 1
)

// ClientConfig holds configuration for the client service.
type ClientConfig struct {
	RecursionSteps int
	EpochMaxSkew   uint64
	RequestTimeout time.Duration
	// DeviceID is a unique identifier for this device (from secure enclave)
	DeviceID []byte
	// AttestationSecret is the key for signing attestations
	AttestationSecret []byte
}

// DefaultClientConfig returns sensible defaults.
func DefaultClientConfig() ClientConfig {
	return ClientConfig{
		RecursionSteps: 16,
		EpochMaxSkew:   1,
		RequestTimeout: 30 * time.Second,
	}
}

type ClientService struct {
	client         pb.AuthorityServiceClient
	oprfClient     *crypto.OPRFClient
	mldsaPublicKey *mldsa65.PublicKey
	recursionSteps int
	epochMaxSkew   uint64
	requestTimeout time.Duration
	attestationCfg crypto.AttestationConfig
	deviceID       []byte
}

func NewClientService(client pb.AuthorityServiceClient, recursionSteps int, mldsaPublicKey *mldsa65.PublicKey) *ClientService {
	cfg := DefaultClientConfig()
	return NewClientServiceWithConfig(client, mldsaPublicKey, cfg)
}

// NewClientServiceWithConfig creates a client service with explicit configuration.
func NewClientServiceWithConfig(client pb.AuthorityServiceClient, mldsaPublicKey *mldsa65.PublicKey, cfg ClientConfig) *ClientService {
	// If no device ID provided, generate one (in production, use secure enclave)
	deviceID := cfg.DeviceID
	if len(deviceID) == 0 {
		deviceID, _ = crypto.GenerateDeviceID()
	}

	// If no attestation secret provided, derive one from device ID
	attestationSecret := cfg.AttestationSecret
	if len(attestationSecret) == 0 {
		// In production, this master key should come from secure storage
		masterKey := []byte("dazzled-master-key-replace-in-production")
		attestationSecret = crypto.DeriveAttestationSecret(masterKey, deviceID)
	}

	return &ClientService{
		client:         client,
		oprfClient:     crypto.NewOPRFClient(),
		mldsaPublicKey: mldsaPublicKey,
		recursionSteps: cfg.RecursionSteps,
		epochMaxSkew:   cfg.EpochMaxSkew,
		requestTimeout: cfg.RequestTimeout,
		attestationCfg: crypto.NewAttestationConfig(attestationSecret),
		deviceID:       deviceID,
	}
}

// ScanImage runs the end-to-end client flow with cryptographic verification.
func (s *ClientService) ScanImage(ctx context.Context, imagePath, modelVersion string) error {
	// Step 1: Load and process image locally
	imgBytes, err := bridge.LoadImage(imagePath)
	if err != nil {
		return fmt.Errorf("load image: %w", err)
	}
	log.Printf("Loaded image: %d bytes", len(imgBytes))

	// Step 2: Generate recursive perceptual hash
	hashVec := bridge.RecursiveInference(imgBytes, s.recursionSteps)
	log.Printf("Generated hash vector: %d dimensions", len(hashVec))

	// Step 3: Map to lattice point for crypto operations
	latticePoint := bridge.MapToLattice(hashVec)

	// Step 4: Blind the lattice point for OPRF
	state, blindedRequest, err := s.oprfClient.Blind(latticePoint.Marshal())
	if err != nil {
		return fmt.Errorf("OPRF blind: %w", err)
	}

	// Step 5: Send blinded request to server
	ctx, cancel := context.WithTimeout(ctx, s.requestTimeout)
	defer cancel()

	resp, err := s.client.CheckImage(ctx, &pb.BlindCheckRequest{
		BlindedElement: blindedRequest,
		ModelVersion:   modelVersion,
	})
	if err != nil {
		return fmt.Errorf("CheckImage RPC: %w", err)
	}

	// Step 6: Verify server response
	if err := s.verifyServerResponse(resp); err != nil {
		return err
	}

	// Step 7: Finalize OPRF to get unblinded result
	oprfOutput, err := s.oprfClient.Finalize(state, resp.BlindedSignature)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrOPRFFailed, err)
	}

	// Step 8: Check for match (non-empty output indicates potential match)
	if len(oprfOutput) == 0 {
		log.Println("No match detected")
		return nil
	}

	log.Println("Potential match detected, uploading voucher share")

	// Step 9: Generate and upload voucher share
	return s.uploadVoucherShare(ctx, resp.UploadToken)
}

// verifyServerResponse performs all cryptographic verification steps.
func (s *ClientService) verifyServerResponse(resp *pb.BlindCheckResponse) error {
	// Check proof version compatibility
	if resp.ProofVersion != supportedProofVersion {
		return fmt.Errorf("%w: got %d, want %d", ErrProofVersionMismatch, resp.ProofVersion, supportedProofVersion)
	}

	// Verify epoch freshness (rollback protection)
	if !crypto.IsEpochFresh(resp.EpochId, time.Now(), s.epochMaxSkew) {
		return ErrStaleEpoch
	}

	// Verify split accumulator structure
	if !crypto.VerifySplitAccumulation(resp.ProofInstance, resp.MembershipProof) {
		return ErrAccumulatorInvalid
	}

	// Parse and validate proof instance
	version, epochFromProof, _, err := crypto.ParseProofInstance(resp.ProofInstance)
	if err != nil {
		return fmt.Errorf("parse proof instance: %w", err)
	}

	// Ensure epoch in proof matches response epoch
	if epochFromProof != resp.EpochId {
		return fmt.Errorf("epoch mismatch: proof=%d, response=%d", epochFromProof, resp.EpochId)
	}
	log.Printf("Proof version=%d, epoch=%d", version, epochFromProof)

	// Verify ML-DSA signature over the proof+signature payload
	sigPayload := crypto.ProofSignaturePayload(resp.ProofInstance, resp.BlindedSignature)
	if !crypto.VerifyMLDSA(s.mldsaPublicKey, sigPayload, resp.MembershipProof) {
		return ErrSignatureInvalid
	}

	log.Println("All cryptographic verifications passed")
	return nil
}

// uploadVoucherShare generates and uploads a threshold secret share.
func (s *ClientService) uploadVoucherShare(ctx context.Context, uploadToken []byte) error {
	if len(uploadToken) == 0 {
		return ErrMissingUploadToken
	}

	shareData, shareIndex, metadata, err := crypto.NewDummyShare()
	if err != nil {
		return fmt.Errorf("generate share: %w", err)
	}

	// Generate device attestation
	attestation, err := crypto.GenerateAttestation(s.attestationCfg, s.deviceID)
	if err != nil {
		return fmt.Errorf("generate attestation: %w", err)
	}

	resp, err := s.client.UploadShare(ctx, &pb.VoucherShare{
		ShareData:         shareData,
		ShareIndex:        shareIndex,
		EncryptedMetadata: metadata,
		UploadToken:       uploadToken,
		DeviceAttestation: attestation,
	})
	if err != nil {
		return fmt.Errorf("UploadShare RPC: %w", err)
	}

	if resp.Status != pb.ShareResponse_ACCEPTED {
		return fmt.Errorf("share rejected: status=%v", resp.Status)
	}

	log.Println("Voucher share accepted")
	return nil
}
