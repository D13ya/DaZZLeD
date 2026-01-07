package crypto

import "github.com/D13ya/DaZZLeD/internal/crypto/lattice"

// VerifyMLDSA is a placeholder verifier; replace with ML-DSA verification.
func VerifyMLDSA(_ lattice.Vector, signature []byte) bool {
	return len(signature) > 0
}
