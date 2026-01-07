package crypto

import (
	"testing"
)

func TestOPRFVerifiableMode_CorruptDLEQProof(t *testing.T) {
	// Generate OPRF keypair
	privBytes, pubBytes, err := GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("GenerateOPRFKeyPair: %v", err)
	}

	privKey, _ := ParseOPRFPrivateKey(privBytes)
	pubKey, _ := ParseOPRFPublicKey(pubBytes)

	server := NewOPRFServer(privKey)
	client := NewOPRFClientWithPublicKey(pubKey)

	// Client blinds
	state, blindedReq, err := client.Blind([]byte("test_input"))
	if err != nil {
		t.Fatalf("Blind: %v", err)
	}

	// Server evaluates
	evaluation, err := server.Evaluate(blindedReq)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}

	// Corrupt the evaluation (which contains DLEQ proof in VOPRF mode)
	corruptedEval := make([]byte, len(evaluation))
	copy(corruptedEval, evaluation)
	if len(corruptedEval) > 10 {
		corruptedEval[10] ^= 0xFF // Flip bits
	}

	// Finalize should fail with corrupted evaluation
	_, err = client.Finalize(state, corruptedEval)
	if err == nil {
		t.Error("Finalize should fail with corrupted evaluation")
	}
}

func TestOPRFVerifiableMode_WrongPublicKey(t *testing.T) {
	// Generate two different OPRF keypairs
	privBytes1, _, err := GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("GenerateOPRFKeyPair: %v", err)
	}
	_, pubBytes2, err := GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("GenerateOPRFKeyPair: %v", err)
	}

	privKey1, _ := ParseOPRFPrivateKey(privBytes1)
	pubKey2, _ := ParseOPRFPublicKey(pubBytes2) // Wrong public key!

	server := NewOPRFServer(privKey1)
	client := NewOPRFClientWithPublicKey(pubKey2) // Client has wrong key

	// Client blinds
	state, blindedReq, err := client.Blind([]byte("test_input"))
	if err != nil {
		t.Fatalf("Blind: %v", err)
	}

	// Server evaluates with privKey1
	evaluation, err := server.Evaluate(blindedReq)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}

	// Finalize should fail - DLEQ proof won't verify against wrong public key
	_, err = client.Finalize(state, evaluation)
	if err == nil {
		t.Error("Finalize should fail with wrong public key")
	}
}

func TestOPRFVerifiableMode_MalformedEvaluationBytes(t *testing.T) {
	_, pubBytes, err := GenerateOPRFKeyPair()
	if err != nil {
		t.Fatalf("GenerateOPRFKeyPair: %v", err)
	}

	pubKey, _ := ParseOPRFPublicKey(pubBytes)
	client := NewOPRFClientWithPublicKey(pubKey)

	state, _, err := client.Blind([]byte("test_input"))
	if err != nil {
		t.Fatalf("Blind: %v", err)
	}

	// Test various malformed evaluations
	testCases := []struct {
		name string
		eval []byte
	}{
		{"nil", nil},
		{"empty", []byte{}},
		{"too_short", []byte{0x01, 0x02, 0x03}},
		{"random_garbage", []byte("this is not a valid evaluation")},
		{"wrong_length", make([]byte, 100)}, // Wrong size
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := client.Finalize(state, tc.eval)
			if err == nil {
				t.Errorf("Finalize should fail with %s evaluation", tc.name)
			}
		})
	}
}

func TestOPRFVerifiableMode_NilState(t *testing.T) {
	_, pubBytes, _ := GenerateOPRFKeyPair()
	pubKey, _ := ParseOPRFPublicKey(pubBytes)
	client := NewOPRFClientWithPublicKey(pubKey)

	// Try to finalize with nil state
	_, err := client.Finalize(nil, []byte("some_evaluation"))
	if err == nil {
		t.Error("Finalize should fail with nil state")
	}
}

func TestOPRFVerifiableMode_ReuseState(t *testing.T) {
	privBytes, pubBytes, _ := GenerateOPRFKeyPair()
	privKey, _ := ParseOPRFPrivateKey(privBytes)
	pubKey, _ := ParseOPRFPublicKey(pubBytes)

	server := NewOPRFServer(privKey)
	client := NewOPRFClientWithPublicKey(pubKey)

	state, blindedReq, _ := client.Blind([]byte("test_input"))
	evaluation, _ := server.Evaluate(blindedReq)

	// First finalize should succeed
	_, err := client.Finalize(state, evaluation)
	if err != nil {
		t.Fatalf("First Finalize: %v", err)
	}

	// Second finalize with same state - behavior depends on implementation
	// Some implementations allow reuse, some don't
	_, err = client.Finalize(state, evaluation)
	// Just log the behavior, don't fail
	t.Logf("State reuse result: %v", err)
}

func TestOPRFServerEvaluate_MalformedBlindedElement(t *testing.T) {
	privBytes, _, _ := GenerateOPRFKeyPair()
	privKey, _ := ParseOPRFPrivateKey(privBytes)
	server := NewOPRFServer(privKey)

	testCases := []struct {
		name    string
		blinded []byte
	}{
		{"nil", nil},
		{"empty", []byte{}},
		{"too_short", []byte{0x01, 0x02}},
		{"random_garbage", []byte("not a group element")},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := server.Evaluate(tc.blinded)
			if err == nil {
				t.Errorf("Evaluate should fail with %s blinded element", tc.name)
			}
		})
	}
}

func TestComputeDirectOPRF_Consistency(t *testing.T) {
	privBytes, pubBytes, _ := GenerateOPRFKeyPair()
	privKey, _ := ParseOPRFPrivateKey(privBytes)
	pubKey, _ := ParseOPRFPublicKey(pubBytes)

	server := NewOPRFServer(privKey)
	client := NewOPRFClientWithPublicKey(pubKey)

	input := []byte("test_hash_input")

	// Compute direct OPRF (server-side, for seeding Bloom filter)
	directOutput, err := server.ComputeDirectOPRF(input)
	if err != nil {
		t.Fatalf("ComputeDirectOPRF: %v", err)
	}

	// Compute via blind/evaluate/finalize (client-server protocol)
	state, blindedReq, _ := client.Blind(input)
	evaluation, _ := server.Evaluate(blindedReq)
	protocolOutput, err := client.Finalize(state, evaluation)
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	// Both should produce the same output
	if len(directOutput) != len(protocolOutput) {
		t.Errorf("output length mismatch: direct=%d, protocol=%d", len(directOutput), len(protocolOutput))
	}

	// Compare outputs - they should be equal for deterministic input
	for i := range directOutput {
		if directOutput[i] != protocolOutput[i] {
			t.Errorf("output mismatch at byte %d", i)
			break
		}
	}
}
