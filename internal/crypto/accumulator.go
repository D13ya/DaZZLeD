package crypto

import (
	"crypto/rand"
	"encoding/binary"
	"io"
	"time"
)

// VerifySplitAccumulation is a placeholder verifier for proof_instance/membership_proof.
func VerifySplitAccumulation(proofInstance, membershipProof []byte) bool {
	return len(proofInstance) > 0 && len(membershipProof) > 0
}

func NewCommitment() ([]byte, error) {
	commitment := make([]byte, ProofCommitmentSize)
	if _, err := io.ReadFull(rand.Reader, commitment); err != nil {
		return nil, err
	}
	return commitment, nil
}

func BuildProofInstance(version byte, epochID uint64, commitment []byte) []byte {
	out := make([]byte, 1+8+len(commitment))
	out[0] = version
	binary.BigEndian.PutUint64(out[1:9], epochID)
	copy(out[9:], commitment)
	return out
}

func CurrentEpochID(now time.Time) uint64 {
	return uint64(now.Unix() / 86400)
}

func IsEpochFresh(epochID uint64, now time.Time, maxSkew uint64) bool {
	current := CurrentEpochID(now)
	if epochID+maxSkew < current {
		return false
	}
	return true
}

func NewUploadToken() ([]byte, error) {
	token := make([]byte, UploadTokenSize)
	if _, err := io.ReadFull(rand.Reader, token); err != nil {
		return nil, err
	}
	return token, nil
}

func NewDummyShare() (data []byte, index uint32, metadata []byte, err error) {
	data = make([]byte, 32)
	metadata = make([]byte, 32)
	if _, err = io.ReadFull(rand.Reader, data); err != nil {
		return nil, 0, nil, err
	}
	if _, err = io.ReadFull(rand.Reader, metadata); err != nil {
		return nil, 0, nil, err
	}
	index = 1
	return data, index, metadata, nil
}
