package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/crypto/dilithium"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// Test error message constants
const (
	errWrite = "write: %v"
)

// Test loadMLDSAPublicKey with missing file
func TestLoadMLDSAPublicKeyMissingFile(t *testing.T) {
	_, err := loadMLDSAPublicKey("/nonexistent/path/mldsa_pub.bin")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

// Test loadMLDSAPublicKey with invalid data
func TestLoadMLDSAPublicKeyInvalidData(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "bad_key.bin")

	if err := os.WriteFile(keyPath, []byte("garbage"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := loadMLDSAPublicKey(keyPath)
	if err == nil {
		t.Error("expected error for invalid key data")
	}
}

// Test loadMLDSAPublicKey with valid key
func TestLoadMLDSAPublicKeyValid(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "mldsa_pub.bin")

	pubKey, _, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	pubBytes, _ := pubKey.MarshalBinary()
	if err := os.WriteFile(keyPath, pubBytes, 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	key, err := loadMLDSAPublicKey(keyPath)
	if err != nil {
		t.Errorf("loadMLDSAPublicKey: %v", err)
	}
	if key == nil {
		t.Error("expected non-nil key")
	}
}

// Test buildCreds with insecure mode (in dev build)
func TestBuildCredsInsecureMode(t *testing.T) {
	if !IsInsecureAllowed() {
		t.Skip("insecure mode not allowed in this build")
	}

	creds, err := buildCreds("", "", "", true)
	if err != nil {
		t.Errorf("buildCreds insecure: %v", err)
	}
	if creds == nil {
		t.Error("expected non-nil creds for insecure mode")
	}
}

// Test buildCreds requires CA when not insecure
func TestBuildCredsRequiresCA(t *testing.T) {
	_, err := buildCreds("", "", "", false)
	if err == nil {
		t.Error("expected error when TLS CA is missing")
	}
}

// Test buildCreds with invalid CA path
func TestBuildCredsInvalidCAPath(t *testing.T) {
	_, err := buildCreds("/nonexistent/ca.pem", "", "", false)
	if err == nil {
		t.Error("expected error for missing CA file")
	}
}

// Test buildCreds with invalid CA PEM
func TestBuildCredsInvalidCAPEM(t *testing.T) {
	tempDir := t.TempDir()
	caPath := filepath.Join(tempDir, "bad_ca.pem")

	if err := os.WriteFile(caPath, []byte("not a valid PEM"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := buildCreds(caPath, "", "", false)
	if err == nil {
		t.Error("expected error for invalid CA PEM")
	}
}

// Test buildCreds with valid CA but missing client certs
func TestBuildCredsPartialClientCerts(t *testing.T) {
	tempDir := t.TempDir()
	caPath, _ := generateTestCA(t, tempDir)

	// Only cert, no key
	_, err := buildCreds(caPath, "some_cert.pem", "", false)
	if err != errMissingClientCreds {
		t.Errorf("expected errMissingClientCreds, got: %v", err)
	}

	// Only key, no cert
	_, err = buildCreds(caPath, "", "some_key.pem", false)
	if err != errMissingClientCreds {
		t.Errorf("expected errMissingClientCreds, got: %v", err)
	}
}

// Test buildCreds with invalid client cert/key
func TestBuildCredsInvalidClientCert(t *testing.T) {
	tempDir := t.TempDir()
	caPath, _ := generateTestCA(t, tempDir)
	certPath := filepath.Join(tempDir, "bad_cert.pem")
	keyPath := filepath.Join(tempDir, "bad_key.pem")

	if err := os.WriteFile(certPath, []byte("bad"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}
	if err := os.WriteFile(keyPath, []byte("bad"), 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	_, err := buildCreds(caPath, certPath, keyPath, false)
	if err == nil {
		t.Error("expected error for invalid client cert/key")
	}
}

// Test buildCreds with valid full TLS config
func TestBuildCredsFullTLSConfig(t *testing.T) {
	tempDir := t.TempDir()
	caPath, caCert := generateTestCA(t, tempDir)
	certPath, keyPath := generateTestClientCert(t, tempDir, caCert)

	creds, err := buildCreds(caPath, certPath, keyPath, false)
	if err != nil {
		t.Errorf("buildCreds full TLS: %v", err)
	}
	if creds == nil {
		t.Error("expected non-nil creds")
	}
}

// Test OPRF public key loading
func TestOPRFPublicKeyLoading(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "oprf_pub.bin")

	_, pubBytes, err := crypto.GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("generate OPRF key: %v", err)
	}

	if err := os.WriteFile(keyPath, pubBytes, 0o600); err != nil {
		t.Fatalf(errWrite, err)
	}

	// Verify the file can be read
	readBytes, err := os.ReadFile(keyPath)
	if err != nil {
		t.Errorf("read OPRF pub key: %v", err)
	}
	if len(readBytes) == 0 {
		t.Error("empty OPRF public key file")
	}
}

// Test OPRF public key missing
func TestOPRFPublicKeyMissing(t *testing.T) {
	_, err := os.ReadFile("/nonexistent/oprf_pub.bin")
	if err == nil {
		t.Error("expected error for missing OPRF public key")
	}
}

// Helper to generate a test CA
func generateTestCA(t *testing.T, dir string) (string, *x509.Certificate) {
	t.Helper()

	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate CA key: %v", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "Test CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
	}

	caDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create CA cert: %v", err)
	}

	caCert, err := x509.ParseCertificate(caDER)
	if err != nil {
		t.Fatalf("parse CA cert: %v", err)
	}

	caPath := filepath.Join(dir, "ca.pem")
	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER})
	if err := os.WriteFile(caPath, caPEM, 0o600); err != nil {
		t.Fatalf("write CA: %v", err)
	}

	return caPath, caCert
}

// Helper to generate a test client cert signed by CA
func generateTestClientCert(t *testing.T, dir string, caCert *x509.Certificate) (string, string) {
	t.Helper()

	// We need the CA key to sign - regenerate for simplicity
	_, _ = ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	clientKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

	clientTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName: "Test Client",
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(time.Hour),
		KeyUsage:  x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{
			x509.ExtKeyUsageClientAuth,
		},
	}

	// Self-sign for test (would normally use CA)
	clientDER, err := x509.CreateCertificate(rand.Reader, clientTemplate, clientTemplate, &clientKey.PublicKey, clientKey)
	if err != nil {
		t.Fatalf("create client cert: %v", err)
	}

	certPath := filepath.Join(dir, "client.pem")
	keyPath := filepath.Join(dir, "client_key.pem")

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: clientDER})
	if err := os.WriteFile(certPath, certPEM, 0o600); err != nil {
		t.Fatalf("write cert: %v", err)
	}

	keyDER, _ := x509.MarshalECPrivateKey(clientKey)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	if err := os.WriteFile(keyPath, keyPEM, 0o600); err != nil {
		t.Fatalf("write key: %v", err)
	}

	return certPath, keyPath
}

// Placeholder - these are imported from the main package
var _ = dilithium.ParsePublicKey
var _ = tls.Config{}
