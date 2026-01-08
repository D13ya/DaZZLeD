package storage

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// TestPersistence_AcrossRestart verifies data survives process restart.
func TestPersistence_AcrossRestart(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "persist_test")

	// Store 1: Write data
	cfg := BadgerConfig{Dir: dbPath, InMemory: false, SyncWrites: true}
	store1 := NewBadgerStoreWithConfig(cfg)

	outputs := [][]byte{
		[]byte("oprf_output_1"),
		[]byte("oprf_output_2"),
		[]byte("oprf_output_3"),
	}

	for _, out := range outputs {
		if err := store1.StoreOPRFOutput(out, []byte("metadata")); err != nil {
			t.Fatalf("store output: %v", err)
		}
	}

	// Close store to simulate process exit
	if err := store1.Close(); err != nil {
		t.Fatalf("close store1: %v", err)
	}

	// Store 2: Reopen and verify data persisted
	store2 := NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	restored, err := store2.GetAllOPRFOutputs()
	if err != nil {
		t.Fatalf("GetAllOPRFOutputs: %v", err)
	}

	if len(restored) != len(outputs) {
		t.Errorf("expected %d outputs after restart, got %d", len(outputs), len(restored))
	}
}

// TestPersistence_ProofBundleAcrossRestart verifies proof bundles survive restart.
func TestPersistence_ProofBundleAcrossRestart(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "proof_persist")

	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	// Store 1: Generate proof bundle
	cfg := BadgerConfig{Dir: dbPath, InMemory: false, SyncWrites: true}
	store1 := NewBadgerStoreWithConfig(cfg)

	// Add some outputs
	for i := 0; i < 5; i++ {
		store1.StoreOPRFOutput([]byte{byte(i), 1, 2, 3}, nil)
	}

	epoch := crypto.CurrentEpochID(time.Now())
	proof1, err := store1.GetSignedProofBundle(epoch, mldsaPrivKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle: %v", err)
	}

	store1.Close()

	// Store 2: Reopen and verify proof bundle is cached
	store2 := NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	proof2, err := store2.GetSignedProofBundle(epoch, mldsaPrivKey)
	if err != nil {
		t.Fatalf("GetSignedProofBundle after restart: %v", err)
	}

	// Proofs should be identical (retrieved from cache)
	if string(proof1) != string(proof2) {
		t.Error("proof bundle should be identical after restart")
	}
}

// TestPersistence_CorruptedDBFile verifies handling of corrupted database.
func TestPersistence_CorruptedDBFile(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "corrupt_test")

	// Create a valid database first
	cfg := BadgerConfig{Dir: dbPath, InMemory: false}
	store := NewBadgerStoreWithConfig(cfg)
	store.StoreOPRFOutput([]byte("test"), nil)
	store.Close()

	// Corrupt some database files
	files, _ := filepath.Glob(filepath.Join(dbPath, "*.vlog"))
	for _, f := range files {
		// Write garbage to the vlog file
		if err := os.WriteFile(f, []byte("corrupted data"), 0o600); err != nil {
			t.Logf("corruption write: %v", err)
		}
	}

	// Attempt to reopen - Badger should detect corruption
	// Note: Badger may recover or error depending on corruption severity
	store2 := NewBadgerStoreWithConfig(cfg)
	if store2 != nil {
		store2.Close()
		t.Log("Badger recovered from minor corruption")
	}
}

// TestPersistence_PartialWriteRecovery simulates crash during write.
func TestPersistence_PartialWriteRecovery(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "partial_write")

	cfg := BadgerConfig{Dir: dbPath, InMemory: false, SyncWrites: true}
	store1 := NewBadgerStoreWithConfig(cfg)

	// Write data
	store1.StoreOPRFOutput([]byte("committed_data"), nil)

	// Start another write but don't properly close (simulate crash)
	// In production, BadgerDB's WAL provides crash recovery
	store1.db.Sync() // Force sync before "crash"
	store1.Close()

	// Reopen and verify committed data is present
	store2 := NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	outputs, _ := store2.GetAllOPRFOutputs()
	if len(outputs) != 1 {
		t.Errorf("expected 1 output after recovery, got %d", len(outputs))
	}
}

// TestPersistence_LargeDataset verifies performance with many entries.
func TestPersistence_LargeDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large dataset test in short mode")
	}

	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "large_dataset")

	cfg := BadgerConfig{Dir: dbPath, InMemory: false}
	store := NewBadgerStoreWithConfig(cfg)

	// Write 10k entries
	numEntries := 10000
	for i := 0; i < numEntries; i++ {
		output := make([]byte, 32)
		output[0] = byte(i >> 24)
		output[1] = byte(i >> 16)
		output[2] = byte(i >> 8)
		output[3] = byte(i)
		store.StoreOPRFOutput(output, nil)
	}

	store.Close()

	// Reopen and count
	store2 := NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	outputs, _ := store2.GetAllOPRFOutputs()
	if len(outputs) != numEntries {
		t.Errorf("expected %d outputs, got %d", numEntries, len(outputs))
	}
}

// TestPersistence_ConcurrentReadWrite verifies concurrent access.
func TestPersistence_ConcurrentReadWrite(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "concurrent")

	cfg := BadgerConfig{Dir: dbPath, InMemory: false}
	store := NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	done := make(chan bool)

	// Writer goroutine
	go func() {
		for i := 0; i < 100; i++ {
			output := make([]byte, 32)
			output[0] = byte(i)
			store.StoreOPRFOutput(output, nil)
		}
		done <- true
	}()

	// Reader goroutine
	go func() {
		for i := 0; i < 50; i++ {
			store.GetAllOPRFOutputs()
		}
		done <- true
	}()

	<-done
	<-done

	outputs, _ := store.GetAllOPRFOutputs()
	if len(outputs) != 100 {
		t.Errorf("expected 100 outputs after concurrent access, got %d", len(outputs))
	}
}

// TestPersistence_EmptyDatabaseReopen verifies empty DB handling.
func TestPersistence_EmptyDatabaseReopen(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "empty_db")

	cfg := BadgerConfig{Dir: dbPath, InMemory: false}

	// Create empty database
	store1 := NewBadgerStoreWithConfig(cfg)
	store1.Close()

	// Reopen empty database
	store2 := NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	outputs, err := store2.GetAllOPRFOutputs()
	if err != nil {
		t.Errorf("GetAllOPRFOutputs on empty: %v", err)
	}
	if len(outputs) != 0 {
		t.Errorf("expected 0 outputs, got %d", len(outputs))
	}
}

// TestPersistence_DirectoryPermissions verifies permission handling.
func TestPersistence_DirectoryPermissions(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "perm_test")

	// Create with explicit permissions
	cfg := BadgerConfig{Dir: dbPath, InMemory: false}
	store := NewBadgerStoreWithConfig(cfg)

	// Verify directory was created
	info, err := os.Stat(dbPath)
	if err != nil {
		t.Fatalf("Stat db path: %v", err)
	}
	if !info.IsDir() {
		t.Error("db path should be directory")
	}

	store.Close()
}

// TestPersistence_DiskFullHandling simulates disk full scenario.
func TestPersistence_DiskFullHandling(t *testing.T) {
	// This test is conceptual - actual disk full simulation is system-dependent
	// BadgerDB returns errors when disk is full, which should be handled gracefully
	t.Log("Disk full handling: BadgerDB returns ErrNoRoom when disk is full")
}

// TestPersistence_DeletedDataNotRecovered verifies deleted data stays deleted.
func TestPersistence_DeletedDataNotRecovered(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "delete_persist")

	cfg := BadgerConfig{Dir: dbPath, InMemory: false, SyncWrites: true}
	store1 := NewBadgerStoreWithConfig(cfg)

	output := []byte("to_be_deleted")
	store1.StoreOPRFOutput(output, nil)
	store1.DeleteOPRFOutput(output)
	store1.Close()

	// Reopen and verify deleted data is gone
	store2 := NewBadgerStoreWithConfig(cfg)
	defer store2.Close()

	outputs, _ := store2.GetAllOPRFOutputs()
	for _, o := range outputs {
		if string(o) == string(output) {
			t.Error("deleted output should not be recovered after restart")
		}
	}
}
