package crypto

import "github.com/cloudflare/circl/sign/mldsa/mldsa65"

// VerifyMLDSA checks an ML-DSA-65 signature over the provided message.
func VerifyMLDSA(publicKey *mldsa65.PublicKey, message, signature []byte) bool {
	if publicKey == nil {
		return false
	}
	return mldsa65.Verify(publicKey, message, nil, signature)
}
