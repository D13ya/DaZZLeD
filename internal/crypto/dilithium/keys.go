package dilithium

import (
	"crypto/rand"

	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

// GenerateKeyPair creates an ML-DSA-65 keypair.
func GenerateKeyPair() (*mldsa65.PublicKey, *mldsa65.PrivateKey, error) {
	return mldsa65.GenerateKey(rand.Reader)
}

// ParsePublicKey decodes a packed ML-DSA-65 public key.
func ParsePublicKey(data []byte) (*mldsa65.PublicKey, error) {
	pk := &mldsa65.PublicKey{}
	if err := pk.UnmarshalBinary(data); err != nil {
		return nil, err
	}
	return pk, nil
}

// ParsePrivateKey decodes a packed ML-DSA-65 private key.
func ParsePrivateKey(data []byte) (*mldsa65.PrivateKey, error) {
	sk := &mldsa65.PrivateKey{}
	if err := sk.UnmarshalBinary(data); err != nil {
		return nil, err
	}
	return sk, nil
}
