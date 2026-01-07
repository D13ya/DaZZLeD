package crypto

import (
	"errors"

	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

var errMissingMLDSAKey = errors.New("mldsa key is required")

// SignMLDSA creates an ML-DSA-65 signature for the provided message.
func SignMLDSA(privateKey *mldsa65.PrivateKey, message []byte) ([]byte, error) {
	if privateKey == nil {
		return nil, errMissingMLDSAKey
	}
	sig := make([]byte, mldsa65.SignatureSize)
	if err := mldsa65.SignTo(privateKey, message, nil, true, sig); err != nil {
		return nil, err
	}
	return sig, nil
}
