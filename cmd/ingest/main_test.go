package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// Test error message constants
const (
	errWriteTempFile = "write temp file: %v"
	errWriteKeyFile  = "write key file: %v"
	errIngestHashes  = "ingestHashes: %v"
)

// Test loadOPRFPrivateKey with missing file
func TestLoadOPRFPrivateKeyMissingFile(t *testing.T) {
	_, err := loadOPRFPrivateKey("/nonexistent/path/oprf.bin")
	if err == nil {
		t.Error("expected error for missing OPRF key file")
	}
}

// Test loadOPRFPrivateKey with invalid key data
func TestLoadOPRFPrivateKeyInvalidData(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "bad_oprf.bin")

	// Write garbage data
	if err := os.WriteFile(keyPath, []byte("not a valid key"), 0o600); err != nil {
		t.Fatalf(errWriteTempFile, err)
	}

	_, err := loadOPRFPrivateKey(keyPath)
	if err == nil {
		t.Error("expected error for invalid OPRF key data")
	}
}

// Test loadOPRFPrivateKey with valid key
func TestLoadOPRFPrivateKeyValid(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "oprf_priv.bin")

	// Generate and save a valid key
	privBytes, _, err := crypto.GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("generate OPRF key: %v", err)
	}
	if err := os.WriteFile(keyPath, privBytes, 0o600); err != nil {
		t.Fatalf(errWriteKeyFile, err)
	}

	key, err := loadOPRFPrivateKey(keyPath)
	if err != nil {
		t.Errorf("loadOPRFPrivateKey: %v", err)
	}
	if key == nil {
		t.Error("expected non-nil key")
	}
}

// Test loadMLDSAPrivateKey with missing file
func TestLoadMLDSAPrivateKeyMissingFile(t *testing.T) {
	_, err := loadMLDSAPrivateKey("/nonexistent/path/mldsa.bin")
	if err == nil {
		t.Error("expected error for missing ML-DSA key file")
	}
}

// Test loadMLDSAPrivateKey with invalid key data
func TestLoadMLDSAPrivateKeyInvalidData(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "bad_mldsa.bin")

	// Write garbage data
	if err := os.WriteFile(keyPath, []byte("not a valid key"), 0o600); err != nil {
		t.Fatalf(errWriteTempFile, err)
	}

	_, err := loadMLDSAPrivateKey(keyPath)
	if err == nil {
		t.Error("expected error for invalid ML-DSA key data")
	}
}

// Test loadMLDSAPrivateKey with valid key
func TestLoadMLDSAPrivateKeyValid(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "mldsa_priv.bin")

	// Generate and save a valid key
	_, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("generate ML-DSA key: %v", err)
	}
	privBytes, _ := privKey.MarshalBinary()
	if err := os.WriteFile(keyPath, privBytes, 0o600); err != nil {
		t.Fatalf(errWriteKeyFile, err)
	}

	key, err := loadMLDSAPrivateKey(keyPath)
	if err != nil {
		t.Errorf("loadMLDSAPrivateKey: %v", err)
	}
	if key == nil {
		t.Error("expected non-nil key")
	}
}

// Test ingestHashes with missing file
func TestIngestHashesMissingFile(t *testing.T) {
	// Create temp store
	privBytes, _, _ := crypto.GenerateOPRFKeyPair()
	privKey, _ := crypto.ParseOPRFPrivateKey(privBytes)
	server := crypto.NewOPRFServer(privKey)

	_, err := ingestHashes("/nonexistent/hashes.txt", server, nil)
	if err == nil {
		t.Error("expected error for missing hash file")
	}
}

// Test ingestHashes with empty file
func TestIngestHashesEmptyFile(t *testing.T) {
	tempDir := t.TempDir()
	hashPath := filepath.Join(tempDir, "empty.txt")

	// Create empty file
	if err := os.WriteFile(hashPath, []byte(""), 0o600); err != nil {
		t.Fatalf(errWriteTempFile, err)
	}

	privBytes, _, _ := crypto.GenerateOPRFKeyPair()
	privKey, _ := crypto.ParseOPRFPrivateKey(privBytes)
	server := crypto.NewOPRFServer(privKey)

	// Need a mock store that implements Store interface
	count, err := ingestHashes(hashPath, server, &mockStore{})
	if err != nil {
		t.Errorf("ingestHashes empty: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0 hashes, got %d", count)
	}
}

// Test ingestHashes with comments and blank lines
func TestIngestHashesCommentsAndBlanks(t *testing.T) {
	tempDir := t.TempDir()
	hashPath := filepath.Join(tempDir, "with_comments.txt")

	content := `# This is a comment
   
# Another comment
	
`
	if err := os.WriteFile(hashPath, []byte(content), 0o600); err != nil {
		t.Fatalf(errWriteTempFile, err)
	}

	privBytes, _, _ := crypto.GenerateOPRFKeyPair()
	privKey, _ := crypto.ParseOPRFPrivateKey(privBytes)
	server := crypto.NewOPRFServer(privKey)

	count, err := ingestHashes(hashPath, server, &mockStore{})
	if err != nil {
		t.Errorf(errIngestHashes, err)
	}
	if count != 0 {
		t.Errorf("expected 0 hashes (all comments), got %d", count)
	}
}

// Test ingestHashes with invalid hex
func TestIngestHashesInvalidHex(t *testing.T) {
	tempDir := t.TempDir()
	hashPath := filepath.Join(tempDir, "bad_hex.txt")

	content := `not-valid-hex
abcdef1234
zzz-invalid
0123456789abcdef`
	if err := os.WriteFile(hashPath, []byte(content), 0o600); err != nil {
		t.Fatalf(errWriteTempFile, err)
	}

	privBytes, _, _ := crypto.GenerateOPRFKeyPair()
	privKey, _ := crypto.ParseOPRFPrivateKey(privBytes)
	server := crypto.NewOPRFServer(privKey)

	count, err := ingestHashes(hashPath, server, &mockStore{})
	if err != nil {
		t.Errorf(errIngestHashes, err)
	}
	// Should only process valid hex lines (2 out of 4)
	if count != 2 {
		t.Errorf("expected 2 valid hashes, got %d", count)
	}
}

// Test ingestHashes with valid hashes
func TestIngestHashesValidHashes(t *testing.T) {
	tempDir := t.TempDir()
	hashPath := filepath.Join(tempDir, "valid.txt")

	// Valid SHA256 hex hashes
	content := `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592
5e884898da28047d9174b40a8da10e2e7dc22ec2c72ffb3c4b48a4b6e9f5a7e4`
	if err := os.WriteFile(hashPath, []byte(content), 0o600); err != nil {
		t.Fatalf(errWriteTempFile, err)
	}

	privBytes, _, _ := crypto.GenerateOPRFKeyPair()
	privKey, _ := crypto.ParseOPRFPrivateKey(privBytes)
	server := crypto.NewOPRFServer(privKey)

	store := &mockStore{}
	count, err := ingestHashes(hashPath, server, store)
	if err != nil {
		t.Errorf(errIngestHashes, err)
	}
	if count != 3 {
		t.Errorf("expected 3 hashes, got %d", count)
	}
	if len(store.stored) != 3 {
		t.Errorf("expected 3 stored outputs, got %d", len(store.stored))
	}
}

// mockStore implements storage.Store for testing
type mockStore struct {
	stored [][]byte
}

func (m *mockStore) StoreOPRFOutput(output, metadata []byte) error {
	m.stored = append(m.stored, output)
	return nil
}

func (m *mockStore) GetAllOPRFOutputs() ([][]byte, error) {
	return m.stored, nil
}

func (m *mockStore) DeleteOPRFOutput(output []byte) error {
	for i, o := range m.stored {
		if string(o) == string(output) {
			m.stored = append(m.stored[:i], m.stored[i+1:]...)
			return nil
		}
	}
	return nil
}

func (m *mockStore) GetProofBundle(epoch uint64) ([]byte, []byte, error) {
	return nil, nil, nil
}

func (m *mockStore) GetSignedProofBundle(epoch uint64, signer *mldsa65.PrivateKey) ([]byte, error) {
	return nil, nil
}

func (m *mockStore) Close() error { return nil }
