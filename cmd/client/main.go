package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"flag"
	"log"
	"os"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/app"
	"github.com/D13ya/DaZZLeD/internal/crypto/dilithium"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	imagePath := flag.String("image", "", "path to image to scan")
	serverAddr := flag.String("server", "localhost:50051", "authority server address")
	modelVersion := flag.String("model-version", "TRM-v2-DINO", "model version identifier")
	recursionSteps := flag.Int("recursion-steps", 16, "number of recursive hashing steps")
	mldsaPubPath := flag.String("mldsa-pub", "keys/mldsa_pub.bin", "path to ML-DSA public key")
	oprfPubPath := flag.String("oprf-pub", "keys/oprf_pub.bin", "path to OPRF public key (required for VOPRF)")
	tlsCA := flag.String("tls-ca", "", "path to server CA certificate (PEM)")
	tlsCert := flag.String("tls-cert", "", "path to client certificate (PEM)")
	tlsKey := flag.String("tls-key", "", "path to client private key (PEM)")
	insecureMode := flag.Bool("insecure", false, "disable TLS (development only)")
	flag.Parse()

	if *imagePath == "" {
		log.Fatal("missing --image")
	}
	creds, err := buildCreds(*tlsCA, *tlsCert, *tlsKey, *insecureMode)
	if err != nil {
		log.Fatalf("failed to configure TLS: %v", err)
	}

	conn, err := grpc.Dial(*serverAddr, grpc.WithTransportCredentials(creds))
	if err != nil {
		log.Fatalf("grpc dial failed: %v", err)
	}
	defer conn.Close()

	client := pb.NewAuthorityServiceClient(conn)
	mldsaKey, err := loadMLDSAPublicKey(*mldsaPubPath)
	if err != nil {
		log.Fatalf("ML-DSA public key load failed: %v", err)
	}

	// OPRF public key is required for verifiable OPRF mode
	oprfPubKey, err := os.ReadFile(*oprfPubPath)
	if err != nil {
		log.Fatalf("OPRF public key load failed: %v", err)
	}

	service, err := app.NewClientService(client, *recursionSteps, mldsaKey, oprfPubKey)
	if err != nil {
		log.Fatalf("client service creation failed: %v", err)
	}
	if err := service.ScanImage(context.Background(), *imagePath, *modelVersion); err != nil {
		log.Fatalf("scan failed: %v", err)
	}
}

func buildCreds(caPath, certPath, keyPath string, insecureMode bool) (credentials.TransportCredentials, error) {
	if insecureMode {
		// Check if insecure mode is allowed (only in dev builds)
		if !IsInsecureAllowed() {
			return nil, errors.New("insecure mode is disabled in production builds - TLS is required")
		}
		// Insecure mode is only allowed in development builds
		log.Println("WARNING: Running in insecure mode - TLS disabled. DO NOT USE IN PRODUCTION.")
		return insecure.NewCredentials(), nil
	}

	// TLS is mandatory in non-insecure mode
	if caPath == "" {
		return nil, errors.New("TLS is required: provide --tls-ca or use --insecure for development only")
	}

	tlsConfig := &tls.Config{
		MinVersion: tls.VersionTLS13, // Require TLS 1.3 for modern security
	}

	caPEM, err := os.ReadFile(caPath)
	if err != nil {
		return nil, err
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		return nil, errors.New("failed to parse --tls-ca")
	}
	tlsConfig.RootCAs = pool

	if certPath != "" || keyPath != "" {
		if certPath == "" || keyPath == "" {
			return nil, errMissingClientCreds
		}
		cert, err := tls.LoadX509KeyPair(certPath, keyPath)
		if err != nil {
			return nil, err
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	return credentials.NewTLS(tlsConfig), nil
}

var errMissingClientCreds = errors.New("both --tls-cert and --tls-key are required when one is set")

func loadMLDSAPublicKey(path string) (*mldsa65.PublicKey, error) {
	keyBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return dilithium.ParsePublicKey(keyBytes)
}
