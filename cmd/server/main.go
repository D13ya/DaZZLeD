package main

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"flag"
	"log"
	"net"
	"os"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/api"
	"github.com/D13ya/DaZZLeD/internal/app"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/crypto/dilithium"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/oprf"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

func main() {
	port := flag.String("port", "50051", "listen port")
	mldsaPrivPath := flag.String("mldsa-priv", "keys/mldsa_priv.bin", "path to ML-DSA private key")
	oprfPrivPath := flag.String("oprf-priv", "keys/oprf_priv.bin", "path to OPRF private key")
	attestSecretPath := flag.String("attest-secret", "keys/attest_secret.bin", "path to attestation verification secret")
	tlsCert := flag.String("tls-cert", "", "path to server certificate (PEM)")
	tlsKey := flag.String("tls-key", "", "path to server private key (PEM)")
	tlsCA := flag.String("tls-ca", "", "path to CA bundle for client auth (PEM)")
	insecureMode := flag.Bool("insecure", false, "disable TLS (development only)")
	flag.Parse()

	mldsaKey, err := loadMLDSAPrivateKey(*mldsaPrivPath)
	if err != nil {
		log.Fatalf("ML-DSA key load failed: %v", err)
	}
	oprfKey, err := loadOPRFPrivateKey(*oprfPrivPath)
	if err != nil {
		log.Fatalf("OPRF key load failed: %v", err)
	}

	// Load attestation secret for verifying device attestations
	attestSecret, err := os.ReadFile(*attestSecretPath)
	if err != nil {
		log.Fatalf("attestation secret load failed: %v", err)
	}

	lis, err := net.Listen("tcp", ":"+*port)
	if err != nil {
		log.Fatalf("listen failed: %v", err)
	}

	opts := []grpc.ServerOption{}
	if creds, err := buildServerCreds(*tlsCert, *tlsKey, *tlsCA, *insecureMode); err != nil {
		log.Fatalf("TLS config failed: %v", err)
	} else if creds != nil {
		opts = append(opts, grpc.Creds(creds))
	}

	store := storage.NewBadgerStore()
	service := app.NewServerService(crypto.NewOPRFServer(oprfKey), mldsaKey, store, 10*time.Minute, attestSecret)
	handler := api.NewGRPCHandler(service)

	grpcServer := grpc.NewServer(opts...)
	pb.RegisterAuthorityServiceServer(grpcServer, handler)

	log.Printf("authority server listening on :%s", *port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve failed: %v", err)
	}
}

func loadMLDSAPrivateKey(path string) (*mldsa65.PrivateKey, error) {
	keyBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return dilithium.ParsePrivateKey(keyBytes)
}

func loadOPRFPrivateKey(path string) (*oprf.PrivateKey, error) {
	keyBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return crypto.ParseOPRFPrivateKey(keyBytes)
}

func buildServerCreds(certPath, keyPath, caPath string, insecureMode bool) (credentials.TransportCredentials, error) {
	if insecureMode {
		return nil, nil
	}
	if certPath == "" || keyPath == "" {
		return nil, errors.New("server TLS requires --tls-cert and --tls-key")
	}
	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, err
	}

	tlsConfig := &tls.Config{Certificates: []tls.Certificate{cert}}
	if caPath != "" {
		caPEM, err := os.ReadFile(caPath)
		if err != nil {
			return nil, err
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			return nil, errors.New("failed to load client CA")
		}
		tlsConfig.ClientCAs = pool
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	return credentials.NewTLS(tlsConfig), nil
}
