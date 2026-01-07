package app

import (
	"context"
	"errors"
	"fmt"
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
	// supportedProofVersion kept for legacy V1 compat, but we now prefer V2
	supportedProofVersion   = 1
	supportedProofVersionV2 = 2
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
	// OPRFPublicKey is the server's OPRF public key for VOPRF verification
	// This MUST be provided to enable verifiable OPRF (prevents opaque-list attacks)
	OPRFPublicKey []byte
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

// NewClientService creates a client service with OPRF public key for verifiable mode.
// The oprfPublicKey is REQUIRED - without it, the client cannot verify server's OPRF key usage.
func NewClientService(client pb.AuthorityServiceClient, recursionSteps int, mldsaPublicKey *mldsa65.PublicKey, oprfPublicKey []byte) (*ClientService, error) {
	cfg := DefaultClientConfig()
	cfg.RecursionSteps = recursionSteps
	cfg.OPRFPublicKey = oprfPublicKey
	return NewClientServiceWithConfig(client, mldsaPublicKey, cfg)
}

// NewClientServiceWithConfig creates a client service with explicit configuration.
func NewClientServiceWithConfig(client pb.AuthorityServiceClient, mldsaPublicKey *mldsa65.PublicKey, cfg ClientConfig) (*ClientService, error) {
	// OPRF public key is REQUIRED for verifiable OPRF
	if len(cfg.OPRFPublicKey) == 0 {
		return nil, errors.New("OPRF public key is required for verifiable mode")
	}
	oprfPubKey, err := crypto.ParseOPRFPublicKey(cfg.OPRFPublicKey)
	if err != nil {
		return nil, fmt.Errorf("invalid OPRF public key: %w", err)
	}

	// Device secret for attestation - MUST be provided from secure storage
	// In production, this should come from secure enclave, HSM, or encrypted keychain
	deviceSecret := cfg.DeviceID
	if len(deviceSecret) == 0 {
		// Generate ephemeral secret for this session (NOT recommended for production)
		// Production deployments MUST provide a persistent device secret from secure storage
		deviceSecret, _ = crypto.GenerateDeviceSecret()
	}

	// Attestation secret should be provided from secure storage, not derived from hardcoded keys
	attestationSecret := cfg.AttestationSecret
	if len(attestationSecret) == 0 {
		// In dev mode, use the well-known dev secret
		if IsDevMode() && len(DevAttestationSecret) > 0 {
			attestationSecret = DevAttestationSecret
		} else {
			// Use device secret directly as attestation secret
			// In production, both should come from secure enclave
			attestationSecret = deviceSecret
		}
	}

	return &ClientService{
		client:         client,
		oprfClient:     crypto.NewOPRFClientWithPublicKey(oprfPubKey),
		mldsaPublicKey: mldsaPublicKey,
		recursionSteps: cfg.RecursionSteps,
		epochMaxSkew:   cfg.EpochMaxSkew,
		requestTimeout: cfg.RequestTimeout,
		attestationCfg: crypto.NewAttestationConfig(attestationSecret),
		deviceID:       deviceSecret, // Now stores device secret, not cleartext ID
	}, nil
}

// ScanImage runs the end-to-end client flow with cryptographic verification.
func (s *ClientService) ScanImage(ctx context.Context, imagePath, modelVersion string) error {
	// Step 1: Load and process image locally
	imgBytes, err := bridge.LoadImage(imagePath)
	if err != nil {
		return fmt.Errorf("load image: %w", err)
	}
	// NOTE: Removed logging of image details to prevent sensitive audit trail
	// that could reveal scan activity if logs are centralized

	// Step 2: Generate recursive perceptual hash
	hashVec := bridge.RecursiveInference(imgBytes, s.recursionSteps)

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

	// Step 8: Check for match by verifying OPRF output against the set digest (Bloom filter)
	// The membership proof (ML-DSA signature) was already verified in verifyServerResponse
	isMatch := crypto.VerifyMembershipWithOPRF(oprfOutput, resp.ProofInstance)
	if !isMatch {
		// No match - return silently without logging scan details
		return nil
	}

	// Step 9: Generate and upload voucher share (only on verified match)
	return s.uploadVoucherShare(ctx, resp.UploadToken)
}

// verifyServerResponse performs all cryptographic verification steps.
func (s *ClientService) verifyServerResponse(resp *pb.BlindCheckResponse) error {
	// Verify epoch freshness first (rollback protection)
	if !crypto.IsEpochFresh(resp.EpochId, time.Now(), s.epochMaxSkew) {
		return ErrStaleEpoch
	}

	// Parse and validate proof instance to determine version
	version, epochFromProof, _, err := crypto.ParseProofInstance(resp.ProofInstance)
	if err != nil {
		return fmt.Errorf("parse proof instance: %w", err)
	}

	// Ensure epoch in proof matches response epoch
	if epochFromProof != resp.EpochId {
		return fmt.Errorf("epoch mismatch: proof=%d, response=%d", epochFromProof, resp.EpochId)
	}

	// Check proof version compatibility
	if version != supportedProofVersion && version != supportedProofVersionV2 {
		return fmt.Errorf("%w: got %d, want %d or %d", ErrProofVersionMismatch, version, supportedProofVersion, supportedProofVersionV2)
	}

	// For V2 proofs, verify the commitment signature (addresses "opaque list" criticism)
	// This proves the authority approved this specific Bloom filter
	if version == supportedProofVersionV2 {
		if !crypto.VerifyProofInstanceSignature(s.mldsaPublicKey, resp.ProofInstance) {
			return fmt.Errorf("commitment signature verification failed")
		}
	}

	// Verify split accumulator structure (works for both V1 and V2)
	if !crypto.VerifySplitAccumulation(resp.ProofInstance, resp.MembershipProof) {
		return ErrAccumulatorInvalid
	}

	// Verify ML-DSA signature over the proof+signature payload
	// This proves the server used the correct OPRF key for this request
	sigPayload := crypto.ProofSignaturePayload(resp.ProofInstance, resp.BlindedSignature)
	if !crypto.VerifyMLDSA(s.mldsaPublicKey, sigPayload, resp.MembershipProof) {
		return ErrSignatureInvalid
	}

	// NOTE: Removed logging of verification status to prevent sensitive audit trails
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

	// NOTE: Removed logging of voucher share status to prevent sensitive audit trails
	return nil
}
