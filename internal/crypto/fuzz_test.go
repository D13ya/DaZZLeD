//go:build go1.18

package crypto

import (
	"testing"
)

// FuzzParseProofInstance tests that ParseProofInstance handles arbitrary input.
func FuzzParseProofInstance(f *testing.F) {
	// Seed corpus with valid V1 and V2 formats
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0}) // V1 minimal
	f.Add(make([]byte, 297))                 // V1 expected size

	// V2 seed: version(1) + epoch(8) + bloom(256) + root(32) + sig(3309) = 3606
	v2Seed := make([]byte, 3606)
	v2Seed[0] = ProofVersionV2
	f.Add(v2Seed)

	// Edge cases
	f.Add([]byte{})        // empty
	f.Add([]byte{0})       // invalid version
	f.Add([]byte{255})     // future version
	f.Add([]byte{1})       // truncated
	f.Add([]byte{2})       // truncated V2
	f.Add([]byte{3, 0, 0}) // unknown version

	f.Fuzz(func(t *testing.T, data []byte) {
		// Should never panic, may return error or valid result
		version, epochID, commitment, err := ParseProofInstance(data)
		if err != nil {
			// Expected for malformed input
			return
		}
		// If no error, basic sanity checks
		if version != ProofVersionV1 && version != ProofVersionV2 {
			t.Errorf("unexpected version: %d", version)
		}
		_ = epochID
		_ = commitment
	})
}

// FuzzParseOPRFPublicKey tests OPRF public key parsing with arbitrary input.
func FuzzParseOPRFPublicKey(f *testing.F) {
	// 32-byte valid Ristretto255 point (identity element)
	identity := make([]byte, 32)
	f.Add(identity)

	// Seed with various lengths
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0})
	f.Add(make([]byte, 31))
	f.Add(make([]byte, 32))
	f.Add(make([]byte, 33))
	f.Add(make([]byte, 64))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Should never panic
		_, _ = ParseOPRFPublicKey(data)
	})
}

// FuzzParseOPRFPrivateKey tests OPRF private key parsing with arbitrary input.
func FuzzParseOPRFPrivateKey(f *testing.F) {
	f.Add([]byte{})
	f.Add(make([]byte, 32))
	f.Add(make([]byte, 64))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Should never panic
		_, _ = ParseOPRFPrivateKey(data)
	})
}

// FuzzVerifyMembershipWithOPRF tests membership verification with arbitrary proofs.
func FuzzVerifyMembershipWithOPRF(f *testing.F) {
	oprfOutput := make([]byte, 32)
	validV2 := make([]byte, 3606)
	validV2[0] = ProofVersionV2

	f.Add(oprfOutput, validV2)
	f.Add([]byte{}, []byte{})
	f.Add(make([]byte, 32), []byte{})
	f.Add([]byte{}, make([]byte, 3606))

	f.Fuzz(func(t *testing.T, oprfOutput, proofInstance []byte) {
		// Should never panic
		_ = VerifyMembershipWithOPRF(oprfOutput, proofInstance)
	})
}

// FuzzCheckBloomFilter tests Bloom filter operations with arbitrary data.
func FuzzCheckBloomFilter(f *testing.F) {
	f.Add(make([]byte, 256), make([]byte, 32))
	f.Add([]byte{}, []byte{})
	f.Add(make([]byte, 256), []byte{})
	f.Add([]byte{}, make([]byte, 32))

	f.Fuzz(func(t *testing.T, bloom, element []byte) {
		// Should never panic
		if len(bloom) == BloomFilterSize {
			_ = CheckBloomFilter(bloom, element)
		}
	})
}

// FuzzValidateProofInstance tests proof instance validation with arbitrary data.
func FuzzValidateProofInstance(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1})
	f.Add([]byte{2})
	f.Add(make([]byte, 297))
	f.Add(make([]byte, 3606))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Should never panic
		_ = ValidateProofInstance(data)
	})
}

// FuzzParseSIgnedProofInstance tests signed proof parsing with arbitrary data.
func FuzzParseSignedProofInstance(f *testing.F) {
	v1 := make([]byte, 297)
	v1[0] = ProofVersionV1
	f.Add(v1)

	v2 := make([]byte, 3606)
	v2[0] = ProofVersionV2
	f.Add(v2)

	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		// Should never panic
		_, _, _, _, _ = ParseSignedProofInstance(data)
	})
}
