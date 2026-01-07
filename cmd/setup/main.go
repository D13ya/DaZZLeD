package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/crypto/dilithium"
)

func main() {
	outDir := flag.String("out-dir", "keys", "output directory for authority keys")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o700); err != nil && *outDir != "." {
		log.Fatalf("mkdir failed: %v", err)
	}

	pub, priv, err := dilithium.GenerateKeyPair()
	if err != nil {
		log.Fatalf("ML-DSA key generation failed: %v", err)
	}

	oprfPriv, oprfPub, err := crypto.GenerateOPRFKeyPair()
	if err != nil {
		log.Fatalf("OPRF key generation failed: %v", err)
	}

	// Generate attestation secret for device attestation verification
	attestSecret, err := crypto.GenerateDeviceSecret()
	if err != nil {
		log.Fatalf("attestation secret generation failed: %v", err)
	}

	mldsaPrivPath := filepath.Join(*outDir, "mldsa_priv.bin")
	mldsaPubPath := filepath.Join(*outDir, "mldsa_pub.bin")
	oprfPrivPath := filepath.Join(*outDir, "oprf_priv.bin")
	oprfPubPath := filepath.Join(*outDir, "oprf_pub.bin")
	attestSecretPath := filepath.Join(*outDir, "attest_secret.bin")

	if err := os.WriteFile(mldsaPrivPath, priv.Bytes(), 0o600); err != nil {
		log.Fatalf("write ML-DSA private key failed: %v", err)
	}
	if err := os.WriteFile(mldsaPubPath, pub.Bytes(), 0o600); err != nil {
		log.Fatalf("write ML-DSA public key failed: %v", err)
	}
	if err := os.WriteFile(oprfPrivPath, oprfPriv, 0o600); err != nil {
		log.Fatalf("write OPRF private key failed: %v", err)
	}
	if err := os.WriteFile(oprfPubPath, oprfPub, 0o600); err != nil {
		log.Fatalf("write OPRF public key failed: %v", err)
	}
	if err := os.WriteFile(attestSecretPath, attestSecret, 0o600); err != nil {
		log.Fatalf("write attestation secret failed: %v", err)
	}

	log.Printf("authority keys written under %s", *outDir)
}
