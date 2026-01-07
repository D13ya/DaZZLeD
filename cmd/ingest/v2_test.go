package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// TestIngestV2ProducesSignedBundle tests that ingest produces V2 signed bundles.
func TestIngestV2ProducesSignedBundle(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "db")

	// Generate keys
	oprfPrivBytes, _, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	mldsaPubKey, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	// Create store and ingest some hashes
	cfg := storage.BadgerConfig{Dir: dbPath, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Simulate ingest: compute OPRF outputs and store them
	hashes := [][]byte{
		[]byte("bad_hash_1"),
		[]byte("bad_hash_2"),
		[]byte("bad_hash_3"),
	}
	for _, hash := range hashes {
		oprfOutput, _ := oprfServer.ComputeDirectOPRF(hash)
		store.StoreOPRFOutput(oprfOutput, nil)
	}

	// Generate signed proof bundle (V2)
	epoch := crypto.CurrentEpochID(time.Now())
	proofInstance, err := store.GetSignedProofBundle(epoch, mldsaPrivKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle: %v", err)
	}

	// Verify it's V2 format
	if len(proofInstance) == 0 {
		t.Fatal("empty proof instance")
	}
	if proofInstance[0] != byte(crypto.ProofVersionV2) {
		t.Errorf("expected V2 (2), got version %d", proofInstance[0])
	}

	// Verify client can verify the signature
	if !crypto.VerifyProofInstanceSignature(mldsaPubKey, proofInstance) {
		t.Error("client should be able to verify V2 commitment signature")
	}
}

// TestIngestV2ClientAcceptsBundle tests that a client can accept V2 bundles.
func TestIngestV2ClientAcceptsBundle(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "db")

	// Generate keys
	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	mldsaPubKey, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	// Create store
	cfg := storage.BadgerConfig{Dir: dbPath, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Ingest a known-bad hash
	knownBadImage := []byte("known_bad_image_hash")
	oprfOutput, _ := oprfServer.ComputeDirectOPRF(knownBadImage)
	store.StoreOPRFOutput(oprfOutput, nil)

	// Generate V2 proof
	epoch := crypto.CurrentEpochID(time.Now())
	proofInstance, _ := store.GetSignedProofBundle(epoch, mldsaPrivKey)

	// Client verifies the commitment signature
	if !crypto.VerifyProofInstanceSignature(mldsaPubKey, proofInstance) {
		t.Error("commitment signature verification failed")
	}

	// Client checks membership
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	state, blindedReq, _ := oprfClient.Blind(knownBadImage)

	// Server evaluates
	eval, _ := oprfServer.Evaluate(blindedReq)

	// Client finalizes
	clientOPRFOutput, _ := oprfClient.Finalize(state, eval)

	// Check membership in the Bloom filter
	isMatch := crypto.VerifyMembershipWithOPRF(clientOPRFOutput, proofInstance)
	if !isMatch {
		t.Error("known-bad hash should match in V2 proof")
	}
}

// TestIngestV2NonMemberNotInBundle tests that non-members don't match.
func TestIngestV2NonMemberNotInBundle(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "db")

	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: dbPath, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Ingest different hashes
	store.StoreOPRFOutput([]byte("some_other_hash_oprf"), nil)

	epoch := crypto.CurrentEpochID(time.Now())
	proofInstance, _ := store.GetSignedProofBundle(epoch, mldsaPrivKey)

	// Client checks a non-matching image
	nonMatchingImage := []byte("innocent_image")
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	state, blindedReq, _ := oprfClient.Blind(nonMatchingImage)

	eval, _ := oprfServer.Evaluate(blindedReq)
	clientOPRFOutput, _ := oprfClient.Finalize(state, eval)

	isMatch := crypto.VerifyMembershipWithOPRF(clientOPRFOutput, proofInstance)
	if isMatch {
		t.Error("non-member should NOT match in V2 proof")
	}
}

// TestIngestV2BundlePersistsAcrossRestart tests V2 bundles persist.
func TestIngestV2BundlePersistsAcrossRestart(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "db")

	oprfPrivBytes, _, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	mldsaPubKey, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	// First session: ingest and generate bundle
	cfg := storage.BadgerConfig{Dir: dbPath, InMemory: false, SyncWrites: true}
	store1 := storage.NewBadgerStoreWithConfig(cfg)

	oprfOutput, _ := oprfServer.ComputeDirectOPRF([]byte("hash"))
	store1.StoreOPRFOutput(oprfOutput, nil)

	epoch := crypto.CurrentEpochID(time.Now())
	proof1, _ := store1.GetSignedProofBundle(epoch, mldsaPrivKey)
	store1.Close()

	// Second session: reopen and verify bundle
	store2 := storage.NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	proof2, _ := store2.GetSignedProofBundle(epoch, mldsaPrivKey)

	// Both should be identical
	if string(proof1) != string(proof2) {
		t.Error("V2 bundle should persist across restart")
	}

	// Client should still verify
	if !crypto.VerifyProofInstanceSignature(mldsaPubKey, proof2) {
		t.Error("persisted V2 bundle signature should verify")
	}
}

// TestIngestV2IngestHashesFunction tests the ingestHashes function directly.
func TestIngestV2IngestHashesFunction(t *testing.T) {
	tempDir := t.TempDir()
	hashFile := filepath.Join(tempDir, "hashes.txt")
	dbPath := filepath.Join(tempDir, "db")

	// Create hash file
	content := `# Test hashes
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592
`
	os.WriteFile(hashFile, []byte(content), 0o600)

	oprfPrivBytes, _, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: dbPath, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Use ingestHashes function
	count, err := ingestHashes(hashFile, oprfServer, store)
	if err != nil {
		t.Fatalf("ingestHashes: %v", err)
	}
	if count != 2 {
		t.Errorf("expected 2 hashes, got %d", count)
	}

	// Generate V2 bundle
	epoch := crypto.CurrentEpochID(time.Now())
	proofInstance, err := store.GetSignedProofBundle(epoch, mldsaPrivKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle: %v", err)
	}

	if proofInstance[0] != byte(crypto.ProofVersionV2) {
		t.Error("ingest should produce V2 bundle")
	}
}
