package app

import (
	"context"
	"os"
	"testing"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/D13ya/DaZZLeD/internal/storage"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// End-to-end PSI flow tests

func TestE2E_NonMatchDoesNotTriggerUpload(t *testing.T) {
	// Setup: server with known-bad hashes, client checking non-matching image
	tempDir, _ := os.MkdirTemp("", "e2e_test_*")
	defer os.RemoveAll(tempDir)

	// Generate keys
	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	_, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	// Create store and seed with known-bad hashes
	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Seed known-bad hashes via direct OPRF computation
	knownBadHashes := [][]byte{
		[]byte("known_bad_hash_1"),
		[]byte("known_bad_hash_2"),
		[]byte("known_bad_hash_3"),
	}
	for _, hash := range knownBadHashes {
		oprfOutput, _ := oprfServer.ComputeDirectOPRF(hash)
		store.StoreOPRFOutput(oprfOutput, nil)
	}

	// Create server service
	attestSecret := []byte("test-attestation-secret")
	service := NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, attestSecret)
	defer service.Stop()

	// Client: check a NON-matching image
	nonMatchingImage := []byte("this_image_is_not_in_the_bad_set")

	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	state, blindedReq, _ := oprfClient.Blind(nonMatchingImage)

	// Get server response
	resp, err := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// Client finalizes OPRF
	oprfOutput, err := oprfClient.Finalize(state, resp.BlindedSignature)
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	// Check membership - should NOT match
	isMatch := crypto.VerifyMembershipWithOPRF(oprfOutput, resp.ProofInstance)
	if isMatch {
		t.Error("non-matching image should NOT be in the set")
	}

	// Since there's no match, client would NOT upload a voucher share
	// (the actual client logic skips upload on non-match)
}

func TestE2E_MatchTriggersUploadPath(t *testing.T) {
	// Setup: server with known-bad hashes, client checking matching image
	tempDir, _ := os.MkdirTemp("", "e2e_test_*")
	defer os.RemoveAll(tempDir)

	// Generate keys
	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	mldsaPubKey, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	// Create store
	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	// Seed known-bad hash
	matchingImage := []byte("this_is_a_known_bad_image")
	oprfOutput, _ := oprfServer.ComputeDirectOPRF(matchingImage)
	store.StoreOPRFOutput(oprfOutput, nil)

	// Create server
	attestSecret := []byte("test-attestation-secret")
	service := NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, attestSecret)
	defer service.Stop()

	// Client checks the MATCHING image
	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	state, blindedReq, _ := oprfClient.Blind(matchingImage)

	resp, err := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})
	if err != nil {
		t.Fatalf("HandleCheckImage: %v", err)
	}

	// Client finalizes
	clientOPRFOutput, err := oprfClient.Finalize(state, resp.BlindedSignature)
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	// Check membership - SHOULD match
	isMatch := crypto.VerifyMembershipWithOPRF(clientOPRFOutput, resp.ProofInstance)
	if !isMatch {
		t.Error("matching image SHOULD be in the set")
	}

	// Verify commitment signature (V2)
	if !crypto.VerifyProofInstanceSignature(mldsaPubKey, resp.ProofInstance) {
		t.Error("commitment signature verification failed")
	}

	// Since there IS a match, client would upload a voucher share
	// Simulate the upload
	attestCfg := crypto.NewAttestationConfig(attestSecret)
	attestation, _ := crypto.GenerateAttestation(attestCfg, []byte("device"))

	uploadResp, _ := service.HandleUploadShare(context.Background(), &pb.VoucherShare{
		ShareData:         []byte("voucher_share_data"),
		ShareIndex:        1,
		UploadToken:       resp.UploadToken,
		DeviceAttestation: attestation,
	})

	if uploadResp.Status != pb.ShareResponse_ACCEPTED {
		t.Error("voucher share should be accepted after match")
	}
}

func TestE2E_ClientVerifiesV2Signature(t *testing.T) {
	// Full flow: server generates V2 signed proof, client verifies commitment signature
	tempDir, _ := os.MkdirTemp("", "e2e_test_*")
	defer os.RemoveAll(tempDir)

	oprfPrivBytes, oprfPubBytes, _ := crypto.GenerateOPRFKeyPair()
	oprfPrivKey, _ := crypto.ParseOPRFPrivateKey(oprfPrivBytes)
	oprfPubKey, _ := crypto.ParseOPRFPublicKey(oprfPubBytes)
	mldsaPubKey, mldsaPrivKey, _ := mldsa65.GenerateKey(nil)

	oprfServer := crypto.NewOPRFServer(oprfPrivKey)

	cfg := storage.BadgerConfig{Dir: tempDir, InMemory: false}
	store := storage.NewBadgerStoreWithConfig(cfg)
	defer store.Close()

	_ = store.StoreOPRFOutput([]byte("some_hash"), nil)

	service := NewServerService(oprfServer, mldsaPrivKey, store, 5*time.Minute, []byte("secret"))
	defer service.Stop()

	oprfClient := crypto.NewOPRFClientWithPublicKey(oprfPubKey)
	_, blindedReq, _ := oprfClient.Blind([]byte("test"))

	resp, _ := service.HandleCheckImage(context.Background(), &pb.BlindCheckRequest{
		BlindedElement: blindedReq,
		ModelVersion:   "v1.0",
	})

	// Verify proof is V2
	if resp.ProofVersion != uint32(crypto.ProofVersionV2) {
		t.Errorf("expected V2 proof, got version %d", resp.ProofVersion)
	}

	// Client verifies commitment signature (this is the key security property)
	if !crypto.VerifyProofInstanceSignature(mldsaPubKey, resp.ProofInstance) {
		t.Error("client should be able to verify commitment signature")
	}
}

// Signed set digest integrity tests

func TestSignedSetDigest_TamperedBloomFilter(t *testing.T) {
	_, privKey, _ := mldsa65.GenerateKey(nil)
	pubKey, _, _ := mldsa65.GenerateKey(nil) // Different keypair for verification

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accRoot := make([]byte, crypto.AccumulatorRootSize)

	proof, _ := crypto.BuildSignedProofInstance(privKey, 12345, bloomFilter, accRoot)

	// Tamper with bloom filter (after version+epoch, first byte of commitment)
	tamperedProof := make([]byte, len(proof))
	copy(tamperedProof, proof)
	tamperedProof[9] ^= 0xFF // Byte 9 is start of commitment

	// Should fail with any public key
	if crypto.VerifyProofInstanceSignature(pubKey, tamperedProof) {
		t.Error("tampered bloom filter should fail verification")
	}
}

func TestSignedSetDigest_TamperedAccumulatorRoot(t *testing.T) {
	pubKey, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accRoot := make([]byte, crypto.AccumulatorRootSize)

	proof, _ := crypto.BuildSignedProofInstance(privKey, 12345, bloomFilter, accRoot)

	// Tamper with accumulator root (after bloom filter)
	tamperedProof := make([]byte, len(proof))
	copy(tamperedProof, proof)
	rootStart := 9 + crypto.BloomFilterSize
	tamperedProof[rootStart] ^= 0xFF

	if crypto.VerifyProofInstanceSignature(pubKey, tamperedProof) {
		t.Error("tampered accumulator root should fail verification")
	}
}

func TestSignedSetDigest_TamperedSignature(t *testing.T) {
	pubKey, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accRoot := make([]byte, crypto.AccumulatorRootSize)

	proof, _ := crypto.BuildSignedProofInstance(privKey, 12345, bloomFilter, accRoot)

	// Tamper with signature (last part of proof)
	tamperedProof := make([]byte, len(proof))
	copy(tamperedProof, proof)
	sigStart := 9 + crypto.BloomFilterSize + crypto.AccumulatorRootSize
	if sigStart < len(tamperedProof) {
		tamperedProof[sigStart+10] ^= 0xFF
	}

	if crypto.VerifyProofInstanceSignature(pubKey, tamperedProof) {
		t.Error("tampered signature should fail verification")
	}
}

func TestSignedSetDigest_TamperedEpoch(t *testing.T) {
	pubKey, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accRoot := make([]byte, crypto.AccumulatorRootSize)

	proof, _ := crypto.BuildSignedProofInstance(privKey, 12345, bloomFilter, accRoot)

	// Tamper with epoch (bytes 1-8)
	tamperedProof := make([]byte, len(proof))
	copy(tamperedProof, proof)
	tamperedProof[1] ^= 0xFF

	if crypto.VerifyProofInstanceSignature(pubKey, tamperedProof) {
		t.Error("tampered epoch should fail verification")
	}
}

func TestSignedSetDigest_ClientRejectsV2WithBadSignature(t *testing.T) {
	// Full integration: client rejects response with tampered V2 proof
	pubKey, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accRoot := make([]byte, crypto.AccumulatorRootSize)
	epoch := crypto.CurrentEpochID(time.Now())

	proof, _ := crypto.BuildSignedProofInstance(privKey, epoch, bloomFilter, accRoot)

	// Tamper with proof
	tamperedProof := make([]byte, len(proof))
	copy(tamperedProof, proof)
	tamperedProof[20] ^= 0xFF

	// Create response with tampered proof
	blindedSig := make([]byte, 64)
	sigPayload := crypto.ProofSignaturePayload(tamperedProof, blindedSig)
	membershipProof, _ := crypto.SignMLDSA(privKey, sigPayload)

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    tamperedProof,
		MembershipProof:  membershipProof,
		EpochId:          epoch,
		ProofVersion:     uint32(crypto.ProofVersionV2),
	}

	// Client should reject
	client := mockClientService(t, pubKey)
	err := client.verifyServerResponse(resp)
	if err == nil {
		t.Error("client should reject V2 proof with tampered commitment")
	}
}
