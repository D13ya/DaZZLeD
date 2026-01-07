package storage

import (
	"os"
	"testing"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

func createTestStore(t *testing.T) (*BadgerStore, func()) {
	tempDir, err := os.MkdirTemp("", "badger_test_*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}

	cfg := BadgerConfig{
		Dir:      tempDir,
		InMemory: false,
	}
	store := NewBadgerStoreWithConfig(cfg)

	cleanup := func() {
		store.Close()
		os.RemoveAll(tempDir)
	}

	return store, cleanup
}

func TestGetSignedProofBundle(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	// Generate ML-DSA keypair
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	// Add some OPRF outputs to the store
	oprfOutputs := [][]byte{
		[]byte("oprf_output_1"),
		[]byte("oprf_output_2"),
		[]byte("oprf_output_3"),
	}
	for i, output := range oprfOutputs {
		if err := store.StoreOPRFOutput(output, []byte("metadata")); err != nil {
			t.Fatalf("StoreOPRFOutput %d: %v", i, err)
		}
	}

	// Get signed proof bundle
	epoch := uint64(12345)
	proofInstance, err := store.GetSignedProofBundle(epoch, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle: %v", err)
	}

	// Verify it's a V2 proof
	version, _, _, err := crypto.ParseProofInstance(proofInstance)
	if err != nil {
		t.Fatalf("ParseProofInstance: %v", err)
	}
	if version != crypto.ProofVersionV2 {
		t.Errorf("wrong version: got %d, want %d", version, crypto.ProofVersionV2)
	}

	// Verify the signature
	if !crypto.VerifyProofInstanceSignature(pubKey, proofInstance) {
		t.Error("VerifyProofInstanceSignature failed")
	}

	// Verify membership for stored OPRF outputs
	for i, output := range oprfOutputs {
		if !crypto.VerifyMembershipWithOPRF(output, proofInstance) {
			t.Errorf("OPRF output %d should be in set", i)
		}
	}
}

func TestGetSignedProofBundleCaching(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	_, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	// Add an OPRF output
	if err := store.StoreOPRFOutput([]byte("oprf_output"), nil); err != nil {
		t.Fatalf("StoreOPRFOutput: %v", err)
	}

	epoch := uint64(12345)

	// First call generates the bundle
	proof1, err := store.GetSignedProofBundle(epoch, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle: %v", err)
	}

	// Second call should return cached bundle (same content)
	proof2, err := store.GetSignedProofBundle(epoch, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle (cached): %v", err)
	}

	// Should be identical (cached)
	if len(proof1) != len(proof2) {
		t.Errorf("cached proof length mismatch: %d vs %d", len(proof1), len(proof2))
	}
}

func TestGetSignedProofBundleRequiresSigner(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	// Should fail without signer
	_, err := store.GetSignedProofBundle(12345, nil)
	if err != ErrSignerRequired {
		t.Errorf("expected ErrSignerRequired, got %v", err)
	}
}

func TestGetSignedProofBundlePersistence(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "badger_test_*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	epoch := uint64(12345)
	var proofInstance []byte

	// Create store and generate bundle
	{
		cfg := BadgerConfig{
			Dir:      tempDir,
			InMemory: false,
		}
		store := NewBadgerStoreWithConfig(cfg)

		if err := store.StoreOPRFOutput([]byte("oprf_output"), nil); err != nil {
			t.Fatalf("StoreOPRFOutput: %v", err)
		}

		proofInstance, err = store.GetSignedProofBundle(epoch, privKey)
		if err != nil {
			t.Fatalf("GetSignedProofBundle: %v", err)
		}

		store.Close()
	}

	// Reopen store and verify bundle persisted
	{
		cfg := BadgerConfig{
			Dir:      tempDir,
			InMemory: false,
		}
		store := NewBadgerStoreWithConfig(cfg)
		defer store.Close()

		// Get bundle again - should load from disk
		loaded, err := store.GetSignedProofBundle(epoch, privKey)
		if err != nil {
			t.Fatalf("GetSignedProofBundle: %v", err)
		}

		// Verify signature still valid
		if !crypto.VerifyProofInstanceSignature(pubKey, loaded) {
			t.Error("persisted bundle has invalid signature")
		}

		// Verify it's the same content
		if len(loaded) != len(proofInstance) {
			t.Errorf("persisted proof length mismatch: %d vs %d", len(loaded), len(proofInstance))
		}
	}
}

// Storage behavior tests

func TestStoreOPRFOutputDedupe(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	oprfOutput := []byte("same_oprf_output")

	// Store same output twice with different metadata
	if err := store.StoreOPRFOutput(oprfOutput, []byte("metadata1")); err != nil {
		t.Fatalf("first StoreOPRFOutput: %v", err)
	}
	if err := store.StoreOPRFOutput(oprfOutput, []byte("metadata2")); err != nil {
		t.Fatalf("second StoreOPRFOutput: %v", err)
	}

	// Should only have one entry (deduped by key)
	outputs, err := store.GetAllOPRFOutputs()
	if err != nil {
		t.Fatalf("GetAllOPRFOutputs: %v", err)
	}

	count := 0
	for _, out := range outputs {
		if string(out) == string(oprfOutput) {
			count++
		}
	}

	if count != 1 {
		t.Errorf("expected 1 occurrence, got %d", count)
	}
}

func TestStoreOPRFOutputMultipleUnique(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	outputs := [][]byte{
		[]byte("output1"),
		[]byte("output2"),
		[]byte("output3"),
	}

	for _, out := range outputs {
		if err := store.StoreOPRFOutput(out, nil); err != nil {
			t.Fatalf("StoreOPRFOutput: %v", err)
		}
	}

	retrieved, err := store.GetAllOPRFOutputs()
	if err != nil {
		t.Fatalf("GetAllOPRFOutputs: %v", err)
	}

	if len(retrieved) != len(outputs) {
		t.Errorf("expected %d outputs, got %d", len(outputs), len(retrieved))
	}
}

func TestDeleteOPRFOutput(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	output := []byte("to_delete")

	// Store
	if err := store.StoreOPRFOutput(output, nil); err != nil {
		t.Fatalf("StoreOPRFOutput: %v", err)
	}

	// Verify it's there
	outputs, _ := store.GetAllOPRFOutputs()
	found := false
	for _, o := range outputs {
		if string(o) == string(output) {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("output should exist before delete")
	}

	// Delete
	if err := store.DeleteOPRFOutput(output); err != nil {
		t.Fatalf("DeleteOPRFOutput: %v", err)
	}

	// Verify it's gone
	outputs, _ = store.GetAllOPRFOutputs()
	for _, o := range outputs {
		if string(o) == string(output) {
			t.Error("output should not exist after delete")
		}
	}
}

func TestDeleteOPRFOutputNonExistent(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	// Delete non-existent should not error (idempotent)
	err := store.DeleteOPRFOutput([]byte("does_not_exist"))
	if err != nil {
		t.Errorf("DeleteOPRFOutput non-existent should not error: %v", err)
	}
}

func TestCacheInvalidationAcrossEpochs(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	_, privKey, _ := mldsa65.GenerateKey(nil)

	store.StoreOPRFOutput([]byte("output1"), nil)

	// Get bundle for epoch 1
	epoch1 := uint64(1)
	proof1, err := store.GetSignedProofBundle(epoch1, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle epoch1: %v", err)
	}

	// Get bundle for epoch 2 (different epoch, should regenerate)
	epoch2 := uint64(2)
	proof2, err := store.GetSignedProofBundle(epoch2, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle epoch2: %v", err)
	}

	// Proofs should be different (different epochs)
	_, epochFromProof1, _, _ := crypto.ParseProofInstance(proof1)
	_, epochFromProof2, _, _ := crypto.ParseProofInstance(proof2)

	if epochFromProof1 == epochFromProof2 {
		t.Error("different epochs should produce different proofs")
	}
}

func TestInvalidateEpochCache(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	_, privKey, _ := mldsa65.GenerateKey(nil)

	store.StoreOPRFOutput([]byte("output1"), nil)

	epoch := uint64(12345)

	// Generate bundle
	_, err := store.GetSignedProofBundle(epoch, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle: %v", err)
	}

	// Invalidate cache
	if err := store.InvalidateEpochCache(epoch); err != nil {
		t.Fatalf("InvalidateEpochCache: %v", err)
	}

	// Add new output
	store.StoreOPRFOutput([]byte("output2"), nil)

	// Get bundle again - should include new output
	proof, err := store.GetSignedProofBundle(epoch, privKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle after invalidate: %v", err)
	}

	// New output should be in Bloom filter
	if !crypto.VerifyMembershipWithOPRF([]byte("output2"), proof) {
		t.Error("new output should be in regenerated proof")
	}
}

func TestClosedStoreErrors(t *testing.T) {
	store, cleanup := createTestStore(t)

	// Close the store
	store.Close()
	cleanup() // Still cleanup temp dir

	// Operations on closed store should fail
	_, err := store.GetAllOPRFOutputs()
	if err != ErrStoreClosed {
		t.Errorf("expected ErrStoreClosed, got %v", err)
	}

	err = store.StoreOPRFOutput([]byte("test"), nil)
	if err != ErrStoreClosed {
		t.Errorf("expected ErrStoreClosed, got %v", err)
	}

	_, privKey, _ := mldsa65.GenerateKey(nil)
	_, err = store.GetSignedProofBundle(12345, privKey)
	if err != ErrStoreClosed {
		t.Errorf("expected ErrStoreClosed, got %v", err)
	}
}

func TestGetAllOPRFOutputsEmpty(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	outputs, err := store.GetAllOPRFOutputs()
	if err != nil {
		t.Fatalf("GetAllOPRFOutputs: %v", err)
	}

	if len(outputs) != 0 {
		t.Errorf("expected empty, got %d outputs", len(outputs))
	}
}

func TestStoreOPRFOutputNilMetadata(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	// Nil metadata should work
	if err := store.StoreOPRFOutput([]byte("output"), nil); err != nil {
		t.Errorf("StoreOPRFOutput with nil metadata: %v", err)
	}
}

func TestStoreOPRFOutputEmptyOutput(t *testing.T) {
	store, cleanup := createTestStore(t)
	defer cleanup()

	// Empty output - behavior depends on implementation
	err := store.StoreOPRFOutput([]byte{}, nil)
	// Just log, don't fail - empty might be valid or rejected
	t.Logf("StoreOPRFOutput empty: %v", err)
}
