package main

import (
	"encoding/hex"
	"flag"
	"log"
	"os"
	"path/filepath"

	"github.com/D13ya/DaZZLeD/internal/crypto/dilithium"
)

func main() {
	outPath := flag.String("out", "authority.key", "output key path")
	flag.Parse()

	key, err := dilithium.GeneratePrivateScalar()
	if err != nil {
		log.Fatalf("key generation failed: %v", err)
	}

	encoded := []byte(hex.EncodeToString([]byte{
		byte(key >> 24),
		byte(key >> 16),
		byte(key >> 8),
		byte(key),
	}))

	if err := os.MkdirAll(filepath.Dir(*outPath), 0o700); err != nil && filepath.Dir(*outPath) != "." {
		log.Fatalf("mkdir failed: %v", err)
	}
	if err := os.WriteFile(*outPath, encoded, 0o600); err != nil {
		log.Fatalf("write failed: %v", err)
	}
	log.Printf("authority key written to %s", *outPath)
}
