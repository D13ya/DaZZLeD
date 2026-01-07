package crypto

import (
	"bytes"
	"testing"

	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

func TestBuildSignedProofInstance(t *testing.T) {
	// Generate ML-DSA keypair
	_, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	epoch := uint64(1234567890)
	bloomFilter := make([]byte, BloomFilterSize)
	bloomFilter[0] = 0xFF // Set some bits
	bloomFilter[100] = 0xAA

	accumulatorRoot := make([]byte, AccumulatorRootSize)
	for i := range accumulatorRoot {
		accumulatorRoot[i] = byte(i)
	}

	// Build signed proof instance
	proofInstance, err := BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Verify structure
	if len(proofInstance) != MinProofInstanceSize {
		t.Errorf("wrong size: got %d, want %d", len(proofInstance), MinProofInstanceSize)
	}

	// Parse and verify version
	version, parsedEpoch, commitment, err := ParseProofInstance(proofInstance)
	if err != nil {
		t.Fatalf("ParseProofInstance: %v", err)
	}

	if version != ProofVersionV2 {
		t.Errorf("wrong version: got %d, want %d", version, ProofVersionV2)
	}

	if parsedEpoch != epoch {
		t.Errorf("wrong epoch: got %d, want %d", parsedEpoch, epoch)
	}

	// Commitment should contain bloom + root + signature
	expectedCommitmentSize := SignedCommitmentSize
	if len(commitment) != expectedCommitmentSize {
		t.Errorf("wrong commitment size: got %d, want %d", len(commitment), expectedCommitmentSize)
	}

	// Verify bloom filter in commitment
	if !bytes.Equal(commitment[:BloomFilterSize], bloomFilter) {
		t.Error("bloom filter mismatch")
	}

	// Verify accumulator root in commitment
	rootStart := BloomFilterSize
	rootEnd := rootStart + AccumulatorRootSize
	if !bytes.Equal(commitment[rootStart:rootEnd], accumulatorRoot) {
		t.Error("accumulator root mismatch")
	}
}

func TestVerifyProofInstanceSignature(t *testing.T) {
	// Generate ML-DSA keypair
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	epoch := uint64(1234567890)
	bloomFilter := make([]byte, BloomFilterSize)
	accumulatorRoot := make([]byte, AccumulatorRootSize)

	// Build signed proof instance
	proofInstance, err := BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Verify with correct public key
	if !VerifyProofInstanceSignature(pubKey, proofInstance) {
		t.Error("VerifyProofInstanceSignature failed with correct key")
	}

	// Verify with wrong public key should fail
	wrongPubKey, _, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	if VerifyProofInstanceSignature(wrongPubKey, proofInstance) {
		t.Error("VerifyProofInstanceSignature should fail with wrong key")
	}
}

func TestVerifyProofInstanceSignature_TamperedData(t *testing.T) {
	pubKey, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	epoch := uint64(1234567890)
	bloomFilter := make([]byte, BloomFilterSize)
	accumulatorRoot := make([]byte, AccumulatorRootSize)

	proofInstance, err := BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Tamper with the bloom filter (byte 10 of proof instance, after version+epoch)
	tamperedProof := make([]byte, len(proofInstance))
	copy(tamperedProof, proofInstance)
	tamperedProof[10] ^= 0xFF

	if VerifyProofInstanceSignature(pubKey, tamperedProof) {
		t.Error("VerifyProofInstanceSignature should fail with tampered bloom filter")
	}

	// Tamper with the epoch
	tamperedProof = make([]byte, len(proofInstance))
	copy(tamperedProof, proofInstance)
	tamperedProof[1] ^= 0xFF

	if VerifyProofInstanceSignature(pubKey, tamperedProof) {
		t.Error("VerifyProofInstanceSignature should fail with tampered epoch")
	}
}

func TestParseSignedProofInstance(t *testing.T) {
	_, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	epoch := uint64(0xDEADBEEF12345678)
	bloomFilter := make([]byte, BloomFilterSize)
	bloomFilter[42] = 0xAB
	accumulatorRoot := make([]byte, AccumulatorRootSize)
	accumulatorRoot[0] = 0xCD

	proofInstance, err := BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Parse with signed function
	version, parsedEpoch, rawCommitment, sig, err := ParseSignedProofInstance(proofInstance)
	if err != nil {
		t.Fatalf("ParseSignedProofInstance: %v", err)
	}

	if version != ProofVersionV2 {
		t.Errorf("wrong version: got %d, want %d", version, ProofVersionV2)
	}

	if parsedEpoch != epoch {
		t.Errorf("wrong epoch: got %d, want %d", parsedEpoch, epoch)
	}

	// Raw commitment should be bloom + root only
	if len(rawCommitment) != ProofCommitmentSize {
		t.Errorf("wrong raw commitment size: got %d, want %d", len(rawCommitment), ProofCommitmentSize)
	}

	// Verify bloom filter preserved
	if !bytes.Equal(rawCommitment[:BloomFilterSize], bloomFilter) {
		t.Error("bloom filter mismatch in parsed raw commitment")
	}

	// Verify accumulator root preserved
	if !bytes.Equal(rawCommitment[BloomFilterSize:], accumulatorRoot) {
		t.Error("accumulator root mismatch in parsed raw commitment")
	}

	// Signature should be present
	if len(sig) != CommitmentSignatureSize {
		t.Errorf("wrong signature size: got %d, want %d", len(sig), CommitmentSignatureSize)
	}
}

func TestV1BackwardsCompatibility(t *testing.T) {
	// Build a V1 proof instance (unsigned)
	epoch := uint64(1234567890)
	bloomFilter := make([]byte, BloomFilterSize)
	accumulatorRoot := make([]byte, AccumulatorRootSize)

	commitment := append(bloomFilter, accumulatorRoot...)
	v1Proof := BuildProofInstance(ProofVersionV1, epoch, commitment)

	// ParseProofInstance should work
	version, parsedEpoch, parsedCommitment, err := ParseProofInstance(v1Proof)
	if err != nil {
		t.Fatalf("ParseProofInstance: %v", err)
	}

	if version != ProofVersionV1 {
		t.Errorf("wrong version: got %d, want %d", version, ProofVersionV1)
	}

	if parsedEpoch != epoch {
		t.Errorf("wrong epoch: got %d, want %d", parsedEpoch, epoch)
	}

	if len(parsedCommitment) != ProofCommitmentSize {
		t.Errorf("wrong commitment size: got %d, want %d", len(parsedCommitment), ProofCommitmentSize)
	}

	// ParseSignedProofInstance should work for V1 (returns empty sig)
	version, parsedEpoch, rawCommitment, sig, err := ParseSignedProofInstance(v1Proof)
	if err != nil {
		t.Fatalf("ParseSignedProofInstance: %v", err)
	}

	if version != ProofVersionV1 {
		t.Errorf("wrong version: got %d, want %d", version, ProofVersionV1)
	}

	if len(sig) != 0 {
		t.Errorf("V1 should have no signature, got %d bytes", len(sig))
	}

	if len(rawCommitment) != ProofCommitmentSize {
		t.Errorf("wrong raw commitment size: got %d, want %d", len(rawCommitment), ProofCommitmentSize)
	}
}

func TestBloomFilterMembershipWithSignedProof(t *testing.T) {
	_, privKey, err := mldsa65.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	// Create some OPRF outputs
	oprfOutputs := [][]byte{
		[]byte("hash1_oprf_output"),
		[]byte("hash2_oprf_output"),
		[]byte("hash3_oprf_output"),
	}

	// Build Bloom filter
	bloomFilter := BuildBloomFilter(oprfOutputs)

	// Build signed proof instance
	epoch := uint64(12345)
	accumulatorRoot := make([]byte, AccumulatorRootSize)
	proofInstance, err := BuildSignedProofInstance(privKey, epoch, bloomFilter, accumulatorRoot)
	if err != nil {
		t.Fatalf("BuildSignedProofInstance: %v", err)
	}

	// Check membership for items in the set
	for i, output := range oprfOutputs {
		if !VerifyMembershipWithOPRF(output, proofInstance) {
			t.Errorf("OPRF output %d should be in set", i)
		}
	}

	// Check membership for item NOT in the set
	notInSet := []byte("definitely_not_in_set_hash_oprf_output")
	if VerifyMembershipWithOPRF(notInSet, proofInstance) {
		t.Log("Note: false positive detected (expected occasionally)")
	}
}

// Edge case tests

func TestBuildSignedProofInstance_NilPrivateKey(t *testing.T) {
	bloomFilter := make([]byte, BloomFilterSize)
	accumulatorRoot := make([]byte, AccumulatorRootSize)

	_, err := BuildSignedProofInstance(nil, 12345, bloomFilter, accumulatorRoot)
	if err == nil {
		t.Error("expected error for nil private key")
	}
}

func TestBuildSignedProofInstance_WrongBloomFilterSize(t *testing.T) {
	_, privKey, _ := mldsa65.GenerateKey(nil)

	wrongSizeBloom := make([]byte, 100) // Should be 256
	accumulatorRoot := make([]byte, AccumulatorRootSize)

	_, err := BuildSignedProofInstance(privKey, 12345, wrongSizeBloom, accumulatorRoot)
	if err == nil {
		t.Error("expected error for wrong bloom filter size")
	}
}

func TestBuildSignedProofInstance_WrongAccumulatorRootSize(t *testing.T) {
	_, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, BloomFilterSize)
	wrongSizeRoot := make([]byte, 16) // Should be 32

	_, err := BuildSignedProofInstance(privKey, 12345, bloomFilter, wrongSizeRoot)
	if err == nil {
		t.Error("expected error for wrong accumulator root size")
	}
}

func TestVerifyProofInstanceSignature_EmptyProof(t *testing.T) {
	pubKey, _, _ := mldsa65.GenerateKey(nil)

	if VerifyProofInstanceSignature(pubKey, nil) {
		t.Error("should return false for nil proof")
	}

	if VerifyProofInstanceSignature(pubKey, []byte{}) {
		t.Error("should return false for empty proof")
	}
}

func TestVerifyProofInstanceSignature_TruncatedProof(t *testing.T) {
	pubKey, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, BloomFilterSize)
	accumulatorRoot := make([]byte, AccumulatorRootSize)
	proof, _ := BuildSignedProofInstance(privKey, 12345, bloomFilter, accumulatorRoot)

	// Truncate the proof (remove last 100 bytes of signature)
	truncated := proof[:len(proof)-100]

	if VerifyProofInstanceSignature(pubKey, truncated) {
		t.Error("should return false for truncated proof")
	}
}

func TestVerifyProofInstanceSignature_NilPublicKey(t *testing.T) {
	_, privKey, _ := mldsa65.GenerateKey(nil)

	bloomFilter := make([]byte, BloomFilterSize)
	accumulatorRoot := make([]byte, AccumulatorRootSize)
	proof, _ := BuildSignedProofInstance(privKey, 12345, bloomFilter, accumulatorRoot)

	if VerifyProofInstanceSignature(nil, proof) {
		t.Error("should return false for nil public key")
	}
}

func TestVerifyMembershipWithOPRF_EdgeCases(t *testing.T) {
	// Nil inputs
	if VerifyMembershipWithOPRF(nil, nil) {
		t.Error("should return false for nil inputs")
	}

	// Empty OPRF output
	validProof := make([]byte, MinProofInstanceSizeV1)
	if VerifyMembershipWithOPRF([]byte{}, validProof) {
		t.Error("should return false for empty OPRF output")
	}

	// Truncated proof instance
	truncated := make([]byte, 10)
	if VerifyMembershipWithOPRF([]byte("test"), truncated) {
		t.Error("should return false for truncated proof")
	}
}

func TestCommitmentSigningPayload_DomainSeparator(t *testing.T) {
	epoch := uint64(12345)
	commitment := make([]byte, ProofCommitmentSize)
	commitment[0] = 0xAB

	payload := CommitmentSigningPayload(epoch, commitment)

	// Should contain domain separator
	if !bytes.Contains(payload, []byte(CommitmentDomainSeparator)) {
		t.Error("payload should contain domain separator")
	}

	// Different epochs should produce different payloads
	payload2 := CommitmentSigningPayload(epoch+1, commitment)
	if bytes.Equal(payload, payload2) {
		t.Error("different epochs should produce different payloads")
	}

	// Different commitments should produce different payloads
	commitment2 := make([]byte, ProofCommitmentSize)
	commitment2[0] = 0xCD
	payload3 := CommitmentSigningPayload(epoch, commitment2)
	if bytes.Equal(payload, payload3) {
		t.Error("different commitments should produce different payloads")
	}
}

func TestParseSignedProofInstance_V1NoSignature(t *testing.T) {
	// V1 proofs should return empty signature
	epoch := uint64(12345)
	commitment := make([]byte, ProofCommitmentSize)
	v1Proof := BuildProofInstance(ProofVersionV1, epoch, commitment)

	version, parsedEpoch, rawCommitment, sig, err := ParseSignedProofInstance(v1Proof)
	if err != nil {
		t.Fatalf("ParseSignedProofInstance: %v", err)
	}

	if version != ProofVersionV1 {
		t.Errorf("wrong version: got %d, want %d", version, ProofVersionV1)
	}

	if parsedEpoch != epoch {
		t.Errorf("wrong epoch: got %d, want %d", parsedEpoch, epoch)
	}

	if len(sig) != 0 {
		t.Errorf("V1 should have no signature, got %d bytes", len(sig))
	}

	if !bytes.Equal(rawCommitment, commitment) {
		t.Error("raw commitment mismatch for V1")
	}
}
