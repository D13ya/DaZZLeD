// Package main provides the ingest tool for precomputing OPRF outputs.
// This is used to populate the known-bad hash database for PSI matching.
//
// The ingest pipeline:
// 1. Reads known-bad image hashes from a source file
// 2. Computes OPRF output F(key, hash) for each hash
// 3. Stores outputs in the database via StoreOPRFOutput
// 4. Triggers a new epoch proof bundle (Bloom filter rebuild)
package main

import (
	"bufio"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

func main() {
	oprfPrivPath := flag.String("oprf-priv", "keys/oprf_priv.bin", "path to OPRF private key")
	mldsaPrivPath := flag.String("mldsa-priv", "keys/mldsa_priv.bin", "path to ML-DSA private key for signing")
	dbDir := flag.String("db-dir", "data", "base directory for database files")
	hashFile := flag.String("hashes", "", "path to file containing known-bad hashes (one hex hash per line)")
	rebuildEpoch := flag.Bool("rebuild", false, "force rebuild of current epoch proof bundle")
	flag.Parse()

	if *hashFile == "" && !*rebuildEpoch {
		log.Fatal("either --hashes or --rebuild is required")
	}

	// Load OPRF private key
	oprfKey, err := loadOPRFPrivateKey(*oprfPrivPath)
	if err != nil {
		log.Fatalf("failed to load OPRF key: %v", err)
	}
	oprfServer := crypto.NewOPRFServer(oprfKey)

	// Load ML-DSA private key for signing proof bundles
	mldsaPrivKey, err := loadMLDSAPrivateKey(*mldsaPrivPath)
	if err != nil {
		log.Fatalf("failed to load ML-DSA key: %v", err)
	}

	// Initialize storage
	cfg := storage.DefaultBadgerConfig(*dbDir)
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Process hash file if provided
	if *hashFile != "" {
		count, err := ingestHashes(*hashFile, oprfServer, store)
		if err != nil {
			log.Fatalf("ingest failed: %v", err)
		}
		log.Printf("ingested %d hashes", count)
	}

	// Trigger epoch proof bundle rebuild (V2 signed format)
	epoch := crypto.CurrentEpochID(time.Now())
	log.Printf("triggering signed proof bundle rebuild for epoch %d", epoch)

	// Invalidate cache to force regeneration
	if err := store.InvalidateEpochCache(epoch); err != nil {
		log.Printf("warning: cache invalidation failed: %v", err)
	}

	// Force signed bundle generation (V2 format with ML-DSA signature)
	proofInstance, err := store.GetSignedProofBundle(epoch, mldsaPrivKey)
	if err != nil {
		log.Fatalf("signed proof bundle generation failed: %v", err)
	}

	log.Printf("signed proof bundle (V2) generated: %d bytes", len(proofInstance))
	log.Printf("ingest complete")
}

func loadOPRFPrivateKey(path string) (*crypto.OPRFPrivateKey, error) {
	keyBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return crypto.ParseOPRFPrivateKey(keyBytes)
}

func loadMLDSAPrivateKey(path string) (*mldsa65.PrivateKey, error) {
	keyBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var privKey mldsa65.PrivateKey
	if err := privKey.UnmarshalBinary(keyBytes); err != nil {
		return nil, fmt.Errorf("parse ML-DSA private key: %w", err)
	}
	return &privKey, nil
}

// ingestHashes reads hashes from a file and computes OPRF outputs.
func ingestHashes(path string, oprfServer *crypto.OPRFServer, store storage.Store) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("open hash file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	count := 0
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse hex-encoded hash
		hashBytes, err := hex.DecodeString(line)
		if err != nil {
			log.Printf("warning: line %d: invalid hex: %v", lineNum, err)
			continue
		}

		// Compute OPRF output F(key, hash)
		// Note: For direct hash input (not blinded), we compute the PRF directly
		oprfOutput, err := oprfServer.ComputeDirectOPRF(hashBytes)
		if err != nil {
			log.Printf("warning: line %d: OPRF failed: %v", lineNum, err)
			continue
		}

		// Store the OPRF output
		metadata := []byte(fmt.Sprintf("ingested:%d", time.Now().Unix()))
		if err := store.StoreOPRFOutput(oprfOutput, metadata); err != nil {
			return count, fmt.Errorf("store OPRF output: %w", err)
		}

		count++
		if count%1000 == 0 {
			log.Printf("processed %d hashes...", count)
		}
	}

	if err := scanner.Err(); err != nil {
		return count, fmt.Errorf("scan file: %w", err)
	}

	return count, nil
}
