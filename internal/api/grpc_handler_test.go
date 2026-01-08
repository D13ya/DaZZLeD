package api

import (
	"context"
	"net"
	"os"
	"testing"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/app"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// testServer creates a gRPC server with the authority service for testing.
// Returns the server, client connection, OPRF public key bytes, and cleanup function.
func testServer(t *testing.T) (*grpc.Server, *grpc.ClientConn, []byte, func()) {
	t.Helper()

	tempDir, err := os.MkdirTemp("", "grpc_test_*")
	if err != nil {
		t.Fatalf("tempdir: %v", err)
	}

	// Generate keys
	oprfPrivBytes, oprfPubBytes, err := crypto.GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("generate OPRF key: %v", err)
	}
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: true}
	store := storage.NewBadgerStoreWithConfig(cfg)

	attestSecret := []byte("test-secret")
	service := app.NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, attestSecret)

	handler := NewGRPCHandler(service)

	// Start gRPC server on random port
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	srv := grpc.NewServer(
		grpc.MaxRecvMsgSize(1024 * 1024), // 1MB max message size
	)
	pb.RegisterAuthorityServiceServer(srv, handler)

	go srv.Serve(lis)

	// Create client connection
	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}

	cleanup := func() {
		conn.Close()
		srv.GracefulStop()
		service.Stop()
		store.Close()
		os.RemoveAll(tempDir)
	}

	return srv, conn, oprfPubBytes, cleanup
}

// TestGRPC_CheckImage_EmptyBlindedElement verifies server rejects empty blinded element.
func TestGRPC_CheckImage_EmptyBlindedElement(t *testing.T) {
	_, conn, _, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	_, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: nil,
		ModelVersion:   "v1.0",
	})

	if err == nil {
		t.Error("expected error for empty blinded_element")
	}
	if st, ok := status.FromError(err); ok {
		if st.Code() != codes.InvalidArgument {
			t.Errorf("expected InvalidArgument, got %v", st.Code())
		}
	}
}

// TestGRPC_CheckImage_MissingModelVersion verifies server rejects missing model_version.
func TestGRPC_CheckImage_MissingModelVersion(t *testing.T) {
	_, conn, _, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	_, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: []byte("some blinded element"),
		ModelVersion:   "",
	})

	if err == nil {
		t.Error("expected error for missing model_version")
	}
	if st, ok := status.FromError(err); ok {
		if st.Code() != codes.FailedPrecondition {
			t.Errorf("expected FailedPrecondition, got %v", st.Code())
		}
	}
}

// TestGRPC_CheckImage_MalformedBlindedElement verifies server handles malformed input.
func TestGRPC_CheckImage_MalformedBlindedElement(t *testing.T) {
	_, conn, _, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	// Send garbage that isn't a valid OPRF blinded element
	_, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: []byte("this is not a valid blinded element at all"),
		ModelVersion:   "v1.0",
	})

	if err == nil {
		t.Error("expected error for malformed blinded_element")
	}
	if st, ok := status.FromError(err); ok {
		if st.Code() != codes.InvalidArgument {
			t.Errorf("expected InvalidArgument, got %v", st.Code())
		}
	}
}

// TestGRPC_CheckImage_OversizedPayload verifies server rejects oversized payloads.
func TestGRPC_CheckImage_OversizedPayload(t *testing.T) {
	_, conn, _, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	// Send a very large blinded element (shouldn't be valid anyway)
	oversized := make([]byte, 10*1024*1024) // 10MB
	_, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: oversized,
		ModelVersion:   "v1.0",
	})

	if err == nil {
		t.Error("expected error for oversized payload")
	}
	// Could be ResourceExhausted or InvalidArgument depending on where it fails
	if st, ok := status.FromError(err); ok {
		t.Logf("oversized payload error code: %v", st.Code())
	}
}

// TestGRPC_UploadShare_EmptyToken verifies server rejects empty upload token.
func TestGRPC_UploadShare_EmptyToken(t *testing.T) {
	_, conn, _, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	resp, err := client.UploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       nil,
		DeviceAttestation: []byte("attestation"),
	})

	// Server may return REJECTED status or an error
	if err == nil && resp.Status != pb.ShareResponse_REJECTED {
		t.Error("expected error or REJECTED for empty upload_token")
	}
}

// TestGRPC_UploadShare_InvalidToken verifies server rejects invalid tokens.
func TestGRPC_UploadShare_InvalidToken(t *testing.T) {
	_, conn, _, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	resp, err := client.UploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       []byte("invalid-token-not-from-server"),
		DeviceAttestation: []byte("attestation"),
	})

	// Server may return REJECTED status or an error
	if err == nil && resp.Status != pb.ShareResponse_REJECTED {
		t.Error("expected error or REJECTED for invalid upload_token")
	}
}

// TestGRPC_UploadShare_EmptyAttestation verifies server rejects empty attestation.
func TestGRPC_UploadShare_EmptyAttestation(t *testing.T) {
	_, conn, oprfPubBytes, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	// First get a valid token using the server's OPRF public key
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)

	// Use server's OPRF - need to get a valid blinded element
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("CheckImage for token: %v", err)
	}

	// Now try to upload with empty attestation
	uploadResp, err := client.UploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: nil,
	})

	// Server may return REJECTED status or an error
	if err == nil && uploadResp.Status != pb.ShareResponse_REJECTED {
		t.Error("expected error or REJECTED for empty attestation")
	}
}

// TestGRPC_UploadShare_MalformedAttestation verifies server rejects malformed attestation.
func TestGRPC_UploadShare_MalformedAttestation(t *testing.T) {
	_, conn, oprfPubBytes, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	// Get a valid token first
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)

	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("CheckImage for token: %v", err)
	}

	// Upload with garbage attestation
	uploadResp, err := client.UploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: []byte("not a valid attestation format"),
	})

	// Server may return REJECTED status or an error
	if err == nil && uploadResp.Status != pb.ShareResponse_REJECTED {
		t.Error("expected error or REJECTED for malformed attestation")
	}
}

// TestGRPC_UploadShare_ReplayToken verifies server rejects reused tokens.
func TestGRPC_UploadShare_ReplayToken(t *testing.T) {
	_, conn, oprfPubBytes, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	// Get a valid token
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("CheckImage: %v", err)
	}

	// Generate valid attestation
	attestCfg := crypto.NewAttestationConfig([]byte("test-secret"))
	attestation, _ := crypto.GenerateAttestation(attestCfg, []byte("device"))

	// First upload should succeed
	firstResp, err := client.UploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: attestation,
	})
	if err != nil {
		t.Fatalf("first upload: %v", err)
	}
	if firstResp.Status != pb.ShareResponse_ACCEPTED {
		t.Fatalf("first upload should be ACCEPTED, got %v", firstResp.Status)
	}

	// Second upload with same token should fail (replay protection)
	secondResp, err := client.UploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share2"),
		ShareIndex:        2,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: attestation,
	})

	// Should either error or return REJECTED
	if err == nil && secondResp.Status != pb.ShareResponse_REJECTED {
		t.Error("expected error or REJECTED for replayed token")
	}
}

// TestGRPC_ValidCheckImageFlow verifies the happy path works.
func TestGRPC_ValidCheckImageFlow(t *testing.T) {
	_, conn, oprfPubBytes, cleanup := testServer(t)
	defer cleanup()

	client := pb.NewAuthorityServiceClient(conn)

	// Use the server's OPRF public key
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)

	state, blindedReq, err := oprfClient.Blind([]byte("test image hash"))
	if err != nil {
		t.Fatalf("Blind: %v", err)
	}

	resp, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("CheckImage: %v", err)
	}

	// Verify response has expected fields
	if len(resp.BlindedSignature) == 0 {
		t.Error("missing blinded_signature")
	}
	if len(resp.ProofInstance) == 0 {
		t.Error("missing proof_instance")
	}
	if len(resp.MembershipProof) == 0 {
		t.Error("missing membership_proof")
	}
	if resp.EpochId == 0 {
		t.Error("missing epoch_id")
	}
	if len(resp.UploadToken) == 0 {
		t.Error("missing upload_token")
	}

	// Finalize should succeed with matching OPRF key
	output, err := oprfClient.Finalize(state, resp.BlindedSignature)
	if err != nil {
		t.Errorf("Finalize failed: %v", err)
	}
	if len(output) == 0 {
		t.Error("expected non-empty OPRF output")
	}
}

// TestGRPC_ConnectionTimeout verifies client timeout handling.
func TestGRPC_ConnectionTimeout(t *testing.T) {
	// Create a listener but don't accept connections
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	conn, err := grpc.DialContext(ctx, lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)

	if err == nil {
		conn.Close()
		t.Error("expected timeout error")
	}
}
