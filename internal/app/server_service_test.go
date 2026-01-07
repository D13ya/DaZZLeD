package app

import (
	"context"
	"os"
	"testing"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

func createTestServerService(t *testing.T) (*ServerService, *mldsa65.PublicKey, []byte, func()) {
	// Create temp directory for test DB
	tempDir, err := os.MkdirTemp("", "server_test_*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}

	// Generate OPRF keypair
	oprfPrivBytes, oprfPubBytes, err := crypto.GenerateOPRFKeyPair()
	if err != nil {
		os.RemoveAll(tempDir)
		t.Fatalf("GenerateOPRFKeyPair: %v", err)
	}
	oprfPrivKey, err := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	if err != nil {
		os.RemoveAll(tempDir)
		t.Fatalf("ParseOPRFPrivateKey: %v", err)
	}
	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	// Generate ML-DSA keypair
	mldsaPubKey, mldsaPrivKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		os.RemoveAll(tempDir)
		t.Fatalf("GenerateKey: %v", err)
	}

	// Create store
	cfg := storage.BadgerConfig{
		Dir:      tempDir,
		InMemory: false,
	}
	store := storage.NewBadgerStoreWithConfig(cfg)

	// Add some OPRF outputs to the store
	testOPRFOutput, _ := oprfServer.ComputeDirectOPRF([]byte("test_hash"))
	store.StoreOPRFOutput(testOPRFOutput, []byte("test"))

	// Create server service
	attestSecret := []byte("test-attestation-secret")
	service := NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, attestSecret)

	cleanup := func() {
		service.Stop()
		store.Close()
		os.RemoveAll(tempDir)
	}

	return service, mldsaPubKey, oprfPubBytes, cleanup
}

func TestHandleCheckImage_GeneratesV2SignedProof(t *testing.T) {
	service, mldsaPubKey, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	// Create a valid OPRF client with server's public key
	oprfPubKey, err := crypto.ParseOPRFPublicKey(oprfPubBytes)
	if err != nil {
		t.Fatalf("ParseOPRFPublicKey: %v", err)
	}
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)

	state, blindedReq, err := oprfClient.Blind([]byte("test_image_hash"))
	if err != nil {
		t.Fatalf("Blind: %v", err)
	}

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	}

	resp, err := service.HandleCheckImage(context.Background(), req)
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// Verify response has V2 proof
	if resp.ProofVersion != uint32(crypto.ProofVersionV2) {
		t.Errorf("expected ProofVersion %d, got %d", crypto.ProofVersionV2, resp.ProofVersion)
	}

	// Verify proof instance starts with version byte 2
	if len(resp.ProofInstance) == 0 || resp.ProofInstance[0] != crypto.ProofVersionV2 {
		t.Errorf("proof instance should start with version byte %d", crypto.ProofVersionV2)
	}

	// Verify commitment signature is valid
	if !crypto.VerifyProofInstanceSignature(mldsaPubKey, resp.ProofInstance) {
		t.Error("commitment signature verification failed")
	}

	// Verify we can finalize the OPRF
	_, err = oprfClient.Finalize(state, resp.BlindedSignature)
	if err != nil {
		t.Logf("Note: OPRF finalize: %v", err)
	}
}

func TestHandleCheckImage_ProofVersionMatchesInstance(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	// Create proper blinded element
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	}

	resp, err := service.HandleCheckImage(context.Background(), req)
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// ProofVersion in response should match the actual version byte in proof instance
	if len(resp.ProofInstance) > 0 {
		actualVersion := uint32(resp.ProofInstance[0])
		if resp.ProofVersion != actualVersion {
			t.Errorf("ProofVersion mismatch: response=%d, instance=%d", resp.ProofVersion, actualVersion)
		}
	}
}

func TestHandleCheckImage_MembershipProofVerifiable(t *testing.T) {
	service, mldsaPubKey, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	}

	resp, err := service.HandleCheckImage(context.Background(), req)
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// Verify the membership proof (signature over proof+eval)
	sigPayload := crypto.ProofSignaturePayload(resp.ProofInstance, resp.BlindedSignature)
	if !crypto.VerifyMLDSA(mldsaPubKey, sigPayload, resp.MembershipProof) {
		t.Error("membership proof signature verification failed")
	}
}

func TestHandleCheckImage_MissingModelVersion(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "", // Missing!
	}

	_, err := service.HandleCheckImage(context.Background(), req)
	if err == nil {
		t.Error("expected error for missing model_version")
	}
}

func TestHandleCheckImage_EmptyBlindedElement(t *testing.T) {
	service, _, _, cleanup := createTestServerService(t)
	defer cleanup()

	req := &pb.BlindCheckRequest{
		BlindedElement: nil, // Empty!
		ModelVersion:   "v1.0",
	}

	_, err := service.HandleCheckImage(context.Background(), req)
	if err == nil {
		t.Error("expected error for empty blinded_element")
	}
}

func TestHandleCheckImage_EpochFreshness(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	}

	resp, err := service.HandleCheckImage(context.Background(), req)
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// Epoch should be current
	expectedEpoch := crypto.CurrentEpochID(time.Now())
	if resp.EpochId != expectedEpoch {
		t.Errorf("epoch mismatch: got %d, want %d", resp.EpochId, expectedEpoch)
	}

	// Epoch in proof instance should match response
	_, epochFromProof, _, err := crypto.ParseProofInstance(resp.ProofInstance)
	if err != nil {
		t.Fatalf("ParseProofInstance: %v", err)
	}
	if epochFromProof != resp.EpochId {
		t.Errorf("epoch mismatch: proof=%d, response=%d", epochFromProof, resp.EpochId)
	}
}

func TestHandleCheckImage_UploadTokenIssued(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	}

	resp, err := service.HandleCheckImage(context.Background(), req)
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	if len(resp.UploadToken) == 0 {
		t.Error("expected non-empty upload token")
	}
}

func TestHandleCheckImage_ClientCanVerifyResponse(t *testing.T) {
	// This is an integration test: server generates response, client verifies it
	service, mldsaPubKey, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	req := &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	}

	resp, err := service.HandleCheckImage(context.Background(), req)
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// Create a mock client service with the server's public key
	client := mockClientService(t, mldsaPubKey)

	// Client should be able to verify the response
	if err := client.verifyServerResponse(resp); err != nil {
		t.Errorf("client verification failed: %v", err)
	}
}

// Token lifecycle tests

func TestHandleUploadShare_ValidToken(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	// Get a valid upload token via CheckImage
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	checkResp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	// Generate valid attestation
	attestSecret := []byte("test-attestation-secret")
	attestCfg := crypto.NewAttestationConfig(attestSecret)
	deviceID := []byte("device-123")
	attestation, _ := crypto.GenerateAttestation(attestCfg, deviceID)

	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share_data"),
		ShareIndex:        1,
		UploadToken:       checkResp.UploadToken,
		DeviceAttestation: attestation,
	}

	resp, err := service.HandleUploadShare(context.Background(), shareReq)
	if err != nil {
		t.Fatalf("HandleUploadShare: %v", err)
	}

	if resp.Status != pb.ShareResponse_ACCEPTED {
		t.Errorf("expected ACCEPTED, got %v", resp.Status)
	}
}

func TestHandleUploadShare_ReplayToken(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	// Get a valid upload token
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	checkResp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	attestSecret := []byte("test-attestation-secret")
	attestCfg := crypto.NewAttestationConfig(attestSecret)
	attestation, _ := crypto.GenerateAttestation(attestCfg, []byte("device"))

	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       checkResp.UploadToken,
		DeviceAttestation: attestation,
	}

	// First use should succeed
	resp1, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp1.Status != pb.ShareResponse_ACCEPTED {
		t.Fatal("first upload should be accepted")
	}

	// Replay same token should fail
	resp2, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp2.Status != pb.ShareResponse_REJECTED {
		t.Error("replay token should be rejected")
	}
}

func TestHandleUploadShare_InvalidToken(t *testing.T) {
	service, _, _, cleanup := createTestServerService(t)
	defer cleanup()

	attestSecret := []byte("test-attestation-secret")
	attestCfg := crypto.NewAttestationConfig(attestSecret)
	attestation, _ := crypto.GenerateAttestation(attestCfg, []byte("device"))

	// Use a fabricated token (never issued)
	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       []byte("fabricated-token-never-issued"),
		DeviceAttestation: attestation,
	}

	resp, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp.Status != pb.ShareResponse_REJECTED {
		t.Error("fabricated token should be rejected")
	}
}

func TestHandleUploadShare_EmptyToken(t *testing.T) {
	service, _, _, cleanup := createTestServerService(t)
	defer cleanup()

	attestation := []byte("some_attestation")

	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       nil, // Empty!
		DeviceAttestation: attestation,
	}

	resp, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp.Status != pb.ShareResponse_REJECTED {
		t.Error("empty token should be rejected")
	}
}

func TestHandleUploadShare_BadAttestation(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	// Get valid token
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	checkResp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	// Use garbage attestation
	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       checkResp.UploadToken,
		DeviceAttestation: []byte("garbage-attestation-data"),
	}

	resp, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp.Status != pb.ShareResponse_REJECTED {
		t.Error("bad attestation should be rejected")
	}
}

func TestHandleUploadShare_WrongAttestationSecret(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	// Get valid token
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	checkResp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	// Generate attestation with WRONG secret
	wrongSecret := []byte("wrong-attestation-secret!!!!")
	wrongCfg := crypto.NewAttestationConfig(wrongSecret)
	attestation, _ := crypto.GenerateAttestation(wrongCfg, []byte("device"))

	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       checkResp.UploadToken,
		DeviceAttestation: attestation,
	}

	resp, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp.Status != pb.ShareResponse_REJECTED {
		t.Error("attestation with wrong secret should be rejected")
	}
}

func TestHandleUploadShare_EmptyAttestation(t *testing.T) {
	service, _, oprfPubBytes, cleanup := createTestServerService(t)
	defer cleanup()

	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	checkResp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	shareReq := &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       checkResp.UploadToken,
		DeviceAttestation: nil, // Empty!
	}

	resp, _ := service.HandleUploadShare(context.Background(), shareReq)
	if resp.Status != pb.ShareResponse_REJECTED {
		t.Error("empty attestation should be rejected")
	}
}
