package app

import (
	"testing"
	"time"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// mockClientService creates a minimal client service for testing verification logic
func mockClientService(t *testing.T, mldsaPubKey *mldsa65.PublicKey) *ClientService {
	return &ClientService{
		mldsaPublicKey: mldsaPubKey,
		epochMaxSkew:   1,
	}
}

func TestVerifyServerResponse_V2SignedProof(t *testing.T) {
	// Generate ML-DSA keypair
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	client := mockClientService(t, pubKey)

	// Create a V2 signed proof instance
	epoch := crypto.CurrentEpochID(time.Now())
	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)

	proofInstance, err := crypto.BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Create membership proof (signature over proof+eval)
	blindedSig := make([]byte, 64) // Dummy OPRF evaluation
	sigPayload := crypto.ProofSignaturePayload(proofInstance, blindedSig)
	membershipProof, err := crypto.SignMLDSA(privKey, sigPayload)
	if err != nil {
		t.Fatalf("SignMLDSA: %v", err)
	}

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    proofInstance,
		MembershipProof:  membershipProof,
		EpochId:          epoch,
		ProofVersion:     uint32(crypto.ProofVersionV2),
	}

	// Should verify successfully
	if err := client.verifyServerResponse(resp); err != nil {
		t.Errorf("verifyServerResponse failed: %v", err)
	}
}

func TestVerifyServerResponse_V2WrongSignerKey(t *testing.T) {
	// Generate two different keypairs
	pubKey1, _, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	_, privKey2, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	// Client has pubKey1, but proof is signed with privKey2
	client := mockClientService(t, pubKey1)

	epoch := crypto.CurrentEpochID(time.Now())
	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)

	// Sign with the WRONG key (privKey2, but client expects pubKey1)
	proofInstance, err := crypto.BuildSignedProofInstance(privKey2, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	blindedSig := make([]byte, 64)
	sigPayload := crypto.ProofSignaturePayload(proofInstance, blindedSig)
	membershipProof, err := crypto.SignMLDSA(privKey2, sigPayload)
	if err != nil {
		t.Fatalf("SignMLDSA: %v", err)
	}

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    proofInstance,
		MembershipProof:  membershipProof,
		EpochId:          epoch,
		ProofVersion:     uint32(crypto.ProofVersionV2),
	}

	// Should fail - commitment signed with wrong key
	if err := client.verifyServerResponse(resp); err == nil {
		t.Error("verifyServerResponse should fail with wrong signer key")
	}
}

func TestVerifyServerResponse_V2TamperedBloomFilter(t *testing.T) {
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	client := mockClientService(t, pubKey)

	epoch := crypto.CurrentEpochID(time.Now())
	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)

	proofInstance, err := crypto.BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Tamper with the Bloom filter (byte 10 is in the Bloom filter area)
	tamperedProof := make([]byte, len(proofInstance))
	copy(tamperedProof, proofInstance)
	tamperedProof[10] ^= 0xFF

	blindedSig := make([]byte, 64)
	sigPayload := crypto.ProofSignaturePayload(tamperedProof, blindedSig)
	membershipProof, err := crypto.SignMLDSA(privKey, sigPayload)
	if err != nil {
		t.Fatalf("SignMLDSA: %v", err)
	}

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    tamperedProof,
		MembershipProof:  membershipProof,
		EpochId:          epoch,
		ProofVersion:     uint32(crypto.ProofVersionV2),
	}

	// Should fail - commitment signature won't verify after tampering
	if err := client.verifyServerResponse(resp); err == nil {
		t.Error("verifyServerResponse should fail with tampered bloom filter")
	}
}

func TestVerifyServerResponse_V1BackwardsCompat(t *testing.T) {
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	client := mockClientService(t, pubKey)

	// Create a V1 unsigned proof instance
	epoch := crypto.CurrentEpochID(time.Now())
	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)
	commitment := append(bloomFilter, accumulatorRoot...)
	proofInstance := crypto.BuildProofInstance(crypto.ProofVersionV1, epoch, commitment)

	blindedSig := make([]byte, 64)
	sigPayload := crypto.ProofSignaturePayload(proofInstance, blindedSig)
	membershipProof, err := crypto.SignMLDSA(privKey, sigPayload)
	if err != nil {
		t.Fatalf("SignMLDSA: %v", err)
	}

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    proofInstance,
		MembershipProof:  membershipProof,
		EpochId:          epoch,
		ProofVersion:     uint32(crypto.ProofVersionV1),
	}

	// V1 should still work (no commitment signature verification)
	if err := client.verifyServerResponse(resp); err != nil {
		t.Errorf("verifyServerResponse V1 failed: %v", err)
	}
}

func TestVerifyServerResponse_StaleEpoch(t *testing.T) {
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	client := mockClientService(t, pubKey)

	// Use an old epoch (2 epochs back, beyond max skew of 1)
	staleEpoch := crypto.CurrentEpochID(time.Now()) - 2

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)
	proofInstance, err := crypto.BuildSignedProofInstance(privKey, staleEpoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	blindedSig := make([]byte, 64)
	sigPayload := crypto.ProofSignaturePayload(proofInstance, blindedSig)
	membershipProof, err := crypto.SignMLDSA(privKey, sigPayload)
	if err != nil {
		t.Fatalf("SignMLDSA: %v", err)
	}

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    proofInstance,
		MembershipProof:  membershipProof,
		EpochId:          staleEpoch,
		ProofVersion:     uint32(crypto.ProofVersionV2),
	}

	// Should fail - stale epoch
	err = client.verifyServerResponse(resp)
	if err == nil {
		t.Error("verifyServerResponse should fail with stale epoch")
	}
	if err != ErrStaleEpoch {
		t.Errorf("expected ErrStaleEpoch, got: %v", err)
	}
}

func TestVerifyServerResponse_EpochMismatch(t *testing.T) {
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	client := mockClientService(t, pubKey)

	// Build proof with one epoch, but return different epoch in response
	proofEpoch := crypto.CurrentEpochID(time.Now())
	responseEpoch := proofEpoch + 1

	bloomFilter := make([]byte, crypto.BloomFilterSize)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)
	proofInstance, err := crypto.BuildSignedProofInstance(privKey, proofEpoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	blindedSig := make([]byte, 64)
	sigPayload := crypto.ProofSignaturePayload(proofInstance, blindedSig)
	membershipProof, err := crypto.SignMLDSA(privKey, sigPayload)
	if err != nil {
		t.Fatalf("SignMLDSA: %v", err)
	}

	resp := &pb.BlindCheckResponse{
		BlindedSignature: blindedSig,
		ProofInstance:    proofInstance,
		MembershipProof:  membershipProof,
		EpochId:          responseEpoch, // Mismatch!
		ProofVersion:     uint32(crypto.ProofVersionV2),
	}

	// Should fail - epoch mismatch
	err = client.verifyServerResponse(resp)
	if err == nil {
		t.Error("verifyServerResponse should fail with epoch mismatch")
	}
}
