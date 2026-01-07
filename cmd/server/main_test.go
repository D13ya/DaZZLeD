package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// Test error message constants
const (
	errWrite       = "write: %v"
	errGenerateKey = "generate key: %v"
)

// Test loadMLDSAPrivateKey with missing file
func TestLoadMLDSAPrivateKeyMissingFile(t *testing.T) {
	_, err := loadMLDSAPrivateKey("/nonexistent/path/mldsa_priv.bin")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

// Test loadMLDSAPrivateKey with invalid data
func TestLoadMLDSAPrivateKeyInvalidData(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "bad_key.bin")

	if err := os.WriteFile(keyPath, []byte("garbage"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := loadMLDSAPrivateKey(keyPath)
	if err == nil {
		t.Error("expected error for invalid key data")
	}
}

// Test loadMLDSAPrivateKey with valid key
func TestLoadMLDSAPrivateKeyValid(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "mldsa_priv.bin")

	_, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf(errGenerateKey, err)
	}

	privBytes, _ := privKey.MarshalBinary()
	if err := os.WriteFile(keyPath, privBytes, 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	key, err := loadMLDSAPrivateKey(keyPath)
	if err != nil {
		t.Errorf("loadMLDSAPrivateKey: %v", err)
	}
	if key == nil {
		t.Error("expected non-nil key")
	}
}

// Test loadOPRFPrivateKey with missing file
func TestLoadOPRFPrivateKeyMissingFile(t *testing.T) {
	_, err := loadOPRFPrivateKey("/nonexistent/oprf.bin")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

// Test loadOPRFPrivateKey with invalid data
func TestLoadOPRFPrivateKeyInvalidData(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "bad_oprf.bin")

	if err := os.WriteFile(keyPath, []byte("garbage"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := loadOPRFPrivateKey(keyPath)
	if err == nil {
		t.Error("expected error for invalid key data")
	}
}

// Test loadOPRFPrivateKey with valid key
func TestLoadOPRFPrivateKeyValid(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "oprf_priv.bin")

	privBytes, _, err := crypto.GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf(errGenerateKey, err)
	}

	if err := os.WriteFile(keyPath, privBytes, 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	key, err := loadOPRFPrivateKey(keyPath)
	if err != nil {
		t.Errorf("loadOPRFPrivateKey: %v", err)
	}
	if key == nil {
		t.Error("expected non-nil key")
	}
}

// Test buildServerCreds in insecure mode
func TestBuildServerCredsInsecureMode(t *testing.T) {
	creds, err := buildServerCreds("", "", "", true)
	if err != nil {
		t.Errorf("buildServerCreds insecure: %v", err)
	}
	if creds != nil {
		t.Error("expected nil creds for insecure mode")
	}
}

// Test buildServerCreds requires cert and key
func TestBuildServerCredsRequiresCertKey(t *testing.T) {
	_, err := buildServerCreds("", "", "", false)
	if err == nil {
		t.Error("expected error when cert/key missing")
	}

	_, err = buildServerCreds("cert.pem", "", "", false)
	if err == nil {
		t.Error("expected error when key missing")
	}

	_, err = buildServerCreds("", "key.pem", "", false)
	if err == nil {
		t.Error("expected error when cert missing")
	}
}

// Test buildServerCreds with invalid cert/key path
func TestBuildServerCredsInvalidCertPath(t *testing.T) {
	_, err := buildServerCreds("/nonexistent/cert.pem", "/nonexistent/key.pem", "", false)
	if err == nil {
		t.Error("expected error for missing cert/key files")
	}
}

// Test buildServerCreds with invalid cert/key content
func TestBuildServerCredsInvalidCertContent(t *testing.T) {
	tempDir := t.TempDir()
	certPath := filepath.Join(tempDir, "bad_cert.pem")
	keyPath := filepath.Join(tempDir, "bad_key.pem")

	if err := os.WriteFile(certPath, []byte("bad"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}
	if err := os.WriteFile(keyPath, []byte("bad"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := buildServerCreds(certPath, keyPath, "", false)
	if err == nil {
		t.Error("expected error for invalid cert/key content")
	}
}

// Test buildServerCreds with valid cert/key
func TestBuildServerCredsValidCertKey(t *testing.T) {
	tempDir := t.TempDir()
	certPath, keyPath := generateTestServerCert(t, tempDir)

	creds, err := buildServerCreds(certPath, keyPath, "", false)
	if err != nil {
		t.Errorf("buildServerCreds valid: %v", err)
	}
	if creds == nil {
		t.Error("expected non-nil creds")
	}
}

// Test buildServerCreds with client CA
func TestBuildServerCredsWithClientCA(t *testing.T) {
	tempDir := t.TempDir()
	certPath, keyPath := generateTestServerCert(t, tempDir)
	caPath := generateTestCAFile(t, tempDir)

	creds, err := buildServerCreds(certPath, keyPath, caPath, false)
	if err != nil {
		t.Errorf("buildServerCreds with CA: %v", err)
	}
	if creds == nil {
		t.Error("expected non-nil creds")
	}
}

// Test buildServerCreds with invalid client CA
func TestBuildServerCredsInvalidClientCA(t *testing.T) {
	tempDir := t.TempDir()
	certPath, keyPath := generateTestServerCert(t, tempDir)
	caPath := filepath.Join(tempDir, "bad_ca.pem")

	if err := os.WriteFile(caPath, []byte("not a PEM"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := buildServerCreds(certPath, keyPath, caPath, false)
	if err == nil {
		t.Error("expected error for invalid CA PEM")
	}
}

// Test buildServerCreds with missing client CA file
func TestBuildServerCredsMissingClientCA(t *testing.T) {
	tempDir := t.TempDir()
	certPath, keyPath := generateTestServerCert(t, tempDir)

	_, err := buildServerCreds(certPath, keyPath, "/nonexistent/ca.pem", false)
	if err == nil {
		t.Error("expected error for missing CA file")
	}
}

// Test attestation secret loading
func TestAttestationSecretLoading(t *testing.T) {
	tempDir := t.TempDir()
	secretPath := filepath.Join(tempDir, "attest_secret.bin")

	secret, err := crypto.GenerateDeviceSecret()
	if err != nil {
		t.Fatalf("generate secret: %v", err)
	}

	if err := os.WriteFile(secretPath, secret, 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	readSecret, err := os.ReadFile(secretPath)
	if err != nil {
		t.Errorf("read attestation secret: %v", err)
	}
	if len(readSecret) == 0 {
		t.Error("empty attestation secret")
	}
}

// Test attestation secret missing
func TestAttestationSecretMissing(t *testing.T) {
	_, err := os.ReadFile("/nonexistent/attest_secret.bin")
	if err == nil {
		t.Error("expected error for missing attestation secret")
	}
}

// Helper to generate a test server cert
func generateTestServerCert(t *testing.T, dir string) (string, string) {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf(errGenerateKey, err)
	}

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "localhost",
		},
		NotBefore:   time.Now(),
		NotAfter:    time.Now().Add(time.Hour),
		KeyUsage:    x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:    []string{"localhost"},
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create cert: %v", err)
	}

	certPath := filepath.Join(dir, "server.pem")
	keyPath := filepath.Join(dir, "server_key.pem")

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	if err := os.WriteFile(certPath, certPEM, 0o600); err != nil {
		t.Fatalf("write cert: %v", err)
	}

	keyDER, _ := x509.MarshalECPrivateKey(key)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	if err := os.WriteFile(keyPath, keyPEM, 0o600); err != nil {
		t.Fatalf("write key: %v", err)
	}

	return certPath, keyPath
}

// Helper to generate a test CA file
func generateTestCAFile(t *testing.T, dir string) string {
	t.Helper()

	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "Test CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}

	certDER, _ := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)

	caPath := filepath.Join(dir, "ca.pem")
	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	if err := os.WriteFile(caPath, caPEM, 0o600); err != nil {
		t.Fatalf("write CA: %v", err)
	}

	return caPath
}
