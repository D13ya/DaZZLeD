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
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

func main() {
	port := flag.String("port", "50051", "listen port")
	privKey := flag.Uint("privkey", 0, "server private scalar (optional)")
	tlsCert := flag.String("tls-cert", "", "path to server certificate (PEM)")
	tlsKey := flag.String("tls-key", "", "path to server private key (PEM)")
	tlsCA := flag.String("tls-ca", "", "path to CA bundle for client auth (PEM)")
	insecureMode := flag.Bool("insecure", false, "disable TLS (development only)")
	flag.Parse()

	var sk uint32
	if *privKey == 0 {
		val, err := dilithium.GeneratePrivateScalar()
		if err != nil {
			log.Fatalf("key generation failed: %v", err)
		}
		sk = val
	} else {
		sk = uint32(*privKey)
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
	service := app.NewServerService(crypto.Signer{PrivateKey: sk}, store, 10*time.Minute)
	handler := api.NewGRPCHandler(service)

	grpcServer := grpc.NewServer(opts...)
	pb.RegisterAuthorityServiceServer(grpcServer, handler)

	log.Printf("authority server listening on :%s", *port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve failed: %v", err)
	}
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
