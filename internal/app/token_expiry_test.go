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

// Test constants
const (
	testSecret = "test-secret"
)

// TestTokenExpiryExpiredTokenRejected tests that expired tokens are rejected.
func TestTokenExpiryExpiredTokenRejected(t *testing.T) {
	tempDir, _ := os.MkdirTemp("", "token_expiry_test_*")
	defer os.RemoveAll(tempDir)

	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: true}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Very short TTL for testing
	attestSecret := []byte("test-attestation-secret")
	tokenTTL := 100 * time.Millisecond
	service := NewServerService(oprfServer, mldsaPrivKey, store, tokenTTL, attestSecret)
	defer service.Stop()

	// Get a token
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, err := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	token := resp.UploadToken

	// Wait for token to expire
	time.Sleep(150 * time.Millisecond)

	// Try to use expired token
	attestCfg := crypto.NewAttestationConfig(attestSecret)
	attestation, _ := crypto.GenerateAttestation(attestCfg, []byte("device"))

	uploadResp, _ := service.HandleUploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       token,
		DeviceAttestation: attestation,
	})

	if uploadResp.Status != pb.ShareResponse_REJECTED {
		t.Errorf("expected REJECTED for expired token, got %v", uploadResp.Status)
	}
}

// TestTokenExpiryCleanupGoroutineRemovesExpired tests the cleanup goroutine.
func TestTokenExpiryCleanupGoroutineRemovesExpired(t *testing.T) {
	tempDir, _ := os.MkdirTemp("", "token_cleanup_test_*")
	defer os.RemoveAll(tempDir)

	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: true}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Very short TTL
	attestSecret := []byte(testSecret)
	tokenTTL := 50 * time.Millisecond
	service := NewServerService(oprfServer, mldsaPrivKey, store, tokenTTL, attestSecret)
	defer service.Stop()

	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)

	// Generate multiple tokens
	for i := 0; i < 5; i++ {
		_, blindedReq, _ := oprfClient.Blind([]byte("test"))
		_, err := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
			BlindedElement: blindedReq,
			ModelVersion:   "v1.0",
		})
		if err != nil {
			t.Fatalf("HandleCheckImage: %v", err)
		}
	}

	// Verify tokens exist
	service.tokensMu.Lock()
	initialCount := len(service.tokens)
	service.tokensMu.Unlock()

	if initialCount != 5 {
		t.Errorf("expected 5 tokens, got %d", initialCount)
	}

	// Wait for tokens to expire and cleanup to run
	// tokenCleanupInterval is 30s by default, so we trigger cleanup manually
	time.Sleep(100 * time.Millisecond) // Wait for expiry
	service.cleanupExpiredTokens()     // Manually trigger cleanup

	service.tokensMu.Lock()
	finalCount := len(service.tokens)
	service.tokensMu.Unlock()

	if finalCount != 0 {
		t.Errorf("expected 0 tokens after cleanup, got %d", finalCount)
	}
}

// TestTokenExpiryValidTokenWithinTTL tests that valid tokens work within TTL.
func TestTokenExpiryValidTokenWithinTTL(t *testing.T) {
	tempDir, _ := os.MkdirTemp("", "token_valid_test_*")
	defer os.RemoveAll(tempDir)

	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: true}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Use DevAttestationSecret when in dev mode, otherwise use test secret
	attestSecret := []byte(testSecret)
	if IsDevMode() && len(DevAttestationSecret) > 0 {
		attestSecret = DevAttestationSecret
	}
	tokenTTL := 5 * time.Second // Long enough for test
	service := NewServerService(oprfServer, mldsaPrivKey, store, tokenTTL, attestSecret)
	defer service.Stop()

	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	// Use token immediately (within TTL)
	attestCfg := crypto.NewAttestationConfig(attestSecret)
	attestation, _ := crypto.GenerateAttestation(attestCfg, []byte("device"))

	uploadResp, _ := service.HandleUploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("share"),
		ShareIndex:        1,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: attestation,
	})

	if uploadResp.Status != pb.ShareResponse_ACCEPTED {
		t.Errorf("expected ACCEPTED for valid token, got %v", uploadResp.Status)
	}
}

// TestTokenExpiryConcurrentTokenAccess tests thread-safety of token operations.
func TestTokenExpiryConcurrentTokenAccess(t *testing.T) {
	tempDir, _ := os.MkdirTemp("", "token_concurrent_test_*")
	defer os.RemoveAll(tempDir)

	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: true}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	attestSecret := []byte(testSecret)
	service := NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, attestSecret)
	defer service.Stop()

	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)

	done := make(chan bool)
	errors := make(chan error, 100)

	// Concurrent token generation
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 10; j++ {
				_, blindedReq, _ := oprfClient.Blind([]byte("test"))
				_, err := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
					BlindedElement: blindedReq,
					ModelVersion:   "v1.0",
				})
				if err != nil {
					errors <- err
				}
			}
			done <- true
		}()
	}

	// Concurrent cleanup
	go func() {
		for i := 0; i < 20; i++ {
			service.cleanupExpiredTokens()
			time.Sleep(10 * time.Millisecond)
		}
		done <- true
	}()

	// Wait for all goroutines
	for i := 0; i < 11; i++ {
		<-done
	}

	close(errors)
	for err := range errors {
		t.Errorf("concurrent error: %v", err)
	}
}
