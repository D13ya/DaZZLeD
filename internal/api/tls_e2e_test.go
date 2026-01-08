package api

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/app"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// TestTLS_EndToEnd_ServerClientHandshake tests actual TLS handshake between server and client.
func TestTLS_EndToEnd_ServerClientHandshake(t *testing.T) {
	tempDir := t.TempDir()

	// Generate CA
	caKey, caCert, caPEM := generateCA(t, tempDir)

	// Generate server cert signed by CA
	serverCertPEM, serverKeyPEM := generateServerCert(t, tempDir, caKey, caCert)

	// Start TLS server
	srv, lis, cleanup := startTLSServer(t, tempDir, serverCertPEM, serverKeyPEM, caPEM)
	defer cleanup()

	// Create TLS client with CA trust
	conn, err := dialTLSClient(t, lis.Addr().String(), caPEM)
	if err != nil {
		t.Fatalf("TLS dial failed: %v", err)
	}
	defer conn.Close()

	// Make an actual RPC call
	client := pb.NewAuthorityServiceClient(conn)

	// Generate proper OPRF blinded element
	_, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("CheckImage over TLS failed: %v", err)
	}

	if len(resp.BlindedSignature) == 0 {
		t.Error("expected non-empty response over TLS")
	}

	_ = srv // keep reference
}

// TestTLS_EndToEnd_mTLS tests mutual TLS with client certificate.
func TestTLS_EndToEnd_mTLS(t *testing.T) {
	tempDir := t.TempDir()

	// Generate CA
	caKey, caCert, caPEM := generateCA(t, tempDir)

	// Generate server cert signed by CA
	serverCertPEM, serverKeyPEM := generateServerCert(t, tempDir, caKey, caCert)

	// Generate client cert signed by CA
	clientCertPEM, clientKeyPEM := generateClientCert(t, tempDir, caKey, caCert)

	// Start mTLS server (requires client cert)
	srv, lis, cleanup := startMTLSServer(t, tempDir, serverCertPEM, serverKeyPEM, caPEM)
	defer cleanup()

	// Create mTLS client
	conn, err := dialMTLSClient(t, lis.Addr().String(), caPEM, clientCertPEM, clientKeyPEM)
	if err != nil {
		t.Fatalf("mTLS dial failed: %v", err)
	}
	defer conn.Close()

	// Make an actual RPC call
	client := pb.NewAuthorityServiceClient(conn)

	_, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, err := client.CheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("CheckImage over mTLS failed: %v", err)
	}

	if len(resp.BlindedSignature) == 0 {
		t.Error("expected non-empty response over mTLS")
	}

	_ = srv
}

// TestTLS_EndToEnd_ClientRejectsUntrustedServer tests client rejects untrusted server cert.
func TestTLS_EndToEnd_ClientRejectsUntrustedServer(t *testing.T) {
	tempDir := t.TempDir()

	// Generate two different CAs
	caKey1, caCert1, _ := generateCA(t, tempDir)
	_, _, caPEM2 := generateCAWithName(t, tempDir, "Other CA")

	// Generate server cert signed by CA1
	serverCertPEM, serverKeyPEM := generateServerCert(t, tempDir, caKey1, caCert1)

	// Start TLS server with CA1's cert
	_, lis, cleanup := startTLSServer(t, tempDir, serverCertPEM, serverKeyPEM, nil)
	defer cleanup()

	// Client trusts CA2, not CA1 - should fail
	_, err := dialTLSClient(t, lis.Addr().String(), caPEM2)
	if err == nil {
		t.Error("expected TLS handshake to fail with untrusted CA")
	}
}

// TestTLS_EndToEnd_mTLS_ClientWithoutCertRejected tests server rejects client without cert.
func TestTLS_EndToEnd_mTLS_ClientWithoutCertRejected(t *testing.T) {
	tempDir := t.TempDir()

	// Generate CA
	caKey, caCert, caPEM := generateCA(t, tempDir)

	// Generate server cert
	serverCertPEM, serverKeyPEM := generateServerCert(t, tempDir, caKey, caCert)

	// Start mTLS server (requires client cert)
	_, lis, cleanup := startMTLSServer(t, tempDir, serverCertPEM, serverKeyPEM, caPEM)
	defer cleanup()

	// Client without certificate - should fail
	_, err := dialTLSClient(t, lis.Addr().String(), caPEM)
	if err == nil {
		t.Error("expected mTLS handshake to fail without client cert")
	}
}

// Helper functions

func generateCA(t *testing.T, dir string) (*ecdsa.PrivateKey, *x509.Certificate, []byte) {
	t.Helper()
	return generateCAWithName(t, dir, "Test CA")
}

func generateCAWithName(t *testing.T, dir, name string) (*ecdsa.PrivateKey, *x509.Certificate, []byte) {
	t.Helper()

	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate CA key: %v", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: name,
		},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
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

	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER})

	return caKey, caCert, caPEM
}

func generateServerCert(t *testing.T, dir string, caKey *ecdsa.PrivateKey, caCert *x509.Certificate) ([]byte, []byte) {
	t.Helper()

	serverKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate server key: %v", err)
	}

	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName: "localhost",
		},
		NotBefore:   time.Now().Add(-time.Hour),
		NotAfter:    time.Now().Add(24 * time.Hour),
		KeyUsage:    x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:    []string{"localhost"},
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}

	serverDER, err := x509.CreateCertificate(rand.Reader, serverTemplate, caCert, &serverKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create server cert: %v", err)
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverDER})
	keyDER, _ := x509.MarshalECPrivateKey(serverKey)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})

	return certPEM, keyPEM
}

func generateClientCert(t *testing.T, dir string, caKey *ecdsa.PrivateKey, caCert *x509.Certificate) ([]byte, []byte) {
	t.Helper()

	clientKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate client key: %v", err)
	}

	clientTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(3),
		Subject: pkix.Name{
			CommonName: "Test Client",
		},
		NotBefore:   time.Now().Add(-time.Hour),
		NotAfter:    time.Now().Add(24 * time.Hour),
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	clientDER, err := x509.CreateCertificate(rand.Reader, clientTemplate, caCert, &clientKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create client cert: %v", err)
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: clientDER})
	keyDER, _ := x509.MarshalECPrivateKey(clientKey)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})

	return certPEM, keyPEM
}

func startTLSServer(t *testing.T, tempDir string, certPEM, keyPEM, clientCAPEM []byte) (*grpc.Server, net.Listener, func()) {
	t.Helper()

	// Write certs to files
	certPath := filepath.Join(tempDir, "server.crt")
	keyPath := filepath.Join(tempDir, "server.key")
	os.WriteFile(certPath, certPEM, 0o600)
	os.WriteFile(keyPath, keyPEM, 0o600)

	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		t.Fatalf("load server cert: %v", err)
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS13,
	}

	creds := credentials.NewTLS(tlsConfig)

	return startServerWithCreds(t, tempDir, creds)
}

func startMTLSServer(t *testing.T, tempDir string, certPEM, keyPEM, clientCAPEM []byte) (*grpc.Server, net.Listener, func()) {
	t.Helper()

	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("load server cert: %v", err)
	}

	clientCAPool := x509.NewCertPool()
	if !clientCAPool.AppendCertsFromPEM(clientCAPEM) {
		t.Fatal("failed to add client CA")
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    clientCAPool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
		MinVersion:   tls.VersionTLS13,
	}

	creds := credentials.NewTLS(tlsConfig)

	return startServerWithCreds(t, tempDir, creds)
}

func startServerWithCreds(t *testing.T, tempDir string, creds credentials.TransportCredentials) (*grpc.Server, net.Listener, func()) {
	t.Helper()

	// Generate keys
	oprfPrivBytes, _, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	dbPath := filepath.Join(tempDir, "db")
	cfg := storage.BadgerConfig{Dir: dbPath, InMemory: true}
	store := storage.NewBadgerStoreWithConfig(cfg)

	service := app.NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, []byte("secret"))
	handler := NewGRPCHandler(service)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	srv := grpc.NewServer(grpc.Creds(creds))
	pb.RegisterAuthorityServiceServer(srv, handler)

	go srv.Serve(lis)

	cleanup := func() {
		srv.GracefulStop()
		service.Stop()
		store.Close()
	}

	return srv, lis, cleanup
}

func dialTLSClient(t *testing.T, addr string, caPEM []byte) (*grpc.ClientConn, error) {
	t.Helper()

	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caPEM) {
		t.Fatal("failed to add CA")
	}

	tlsConfig := &tls.Config{
		RootCAs:    caPool,
		MinVersion: tls.VersionTLS13,
	}

	creds := credentials.NewTLS(tlsConfig)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return grpc.DialContext(ctx, addr, grpc.WithTransportCredentials(creds), grpc.WithBlock())
}

func dialMTLSClient(t *testing.T, addr string, caPEM, clientCertPEM, clientKeyPEM []byte) (*grpc.ClientConn, error) {
	t.Helper()

	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caPEM) {
		t.Fatal("failed to add CA")
	}

	clientCert, err := tls.X509KeyPair(clientCertPEM, clientKeyPEM)
	if err != nil {
		t.Fatalf("load client cert: %v", err)
	}

	tlsConfig := &tls.Config{
		RootCAs:      caPool,
		Certificates: []tls.Certificate{clientCert},
		MinVersion:   tls.VersionTLS13,
	}

	creds := credentials.NewTLS(tlsConfig)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return grpc.DialContext(ctx, addr, grpc.WithTransportCredentials(creds), grpc.WithBlock())
}
