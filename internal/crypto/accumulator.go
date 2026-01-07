package crypto

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"io"
	"time"
)

const (
	// MinProofInstanceSize is version(1) + epoch(8) + commitment(ProofCommitmentSize)
	MinProofInstanceSize = 1 + 8 + ProofCommitmentSize
	// ProofVersionV1 is the current proof format version
	ProofVersionV1 byte = 1
)

var (
	ErrProofTooShort     = errors.New("proof instance too short")
	ErrInvalidVersion    = errors.New("unsupported proof version")
	ErrMembershipInvalid = errors.New("membership proof invalid")
)

// VerifySplitAccumulation validates the structure and integrity of the split accumulator.
// This performs structural validation; full cryptographic verification requires the
// ML-DSA signature check which happens separately.
func VerifySplitAccumulation(proofInstance, membershipProof []byte) bool {
	if err := ValidateProofInstance(proofInstance); err != nil {
		return false
	}
	if err := ValidateMembershipProof(membershipProof); err != nil {
		return false
	}
	return true
}

// ValidateProofInstance checks the proof instance format and extracts components.
func ValidateProofInstance(proofInstance []byte) error {
	if len(proofInstance) < MinProofInstanceSize {
		return ErrProofTooShort
	}
	version := proofInstance[0]
	if version != ProofVersionV1 {
		return ErrInvalidVersion
	}
	// Commitment bytes must not be all zeros (sanity check)
	commitment := proofInstance[9:]
	allZero := true
	for _, b := range commitment {
		if b != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return ErrMembershipInvalid
	}
	return nil
}

// ValidateMembershipProof checks the membership proof structure.
// For ML-DSA-65, signatures are 3309 bytes.
func ValidateMembershipProof(membershipProof []byte) error {
	// ML-DSA-65 signature size is 3309 bytes
	const mldsaSignatureSize = 3309
	if len(membershipProof) != mldsaSignatureSize {
		return ErrMembershipInvalid
	}
	return nil
}

// ParseProofInstance extracts version, epoch, and commitment from a proof instance.
func ParseProofInstance(proofInstance []byte) (version byte, epochID uint64, commitment []byte, err error) {
	if err = ValidateProofInstance(proofInstance); err != nil {
		return
	}
	version = proofInstance[0]
	epochID = binary.BigEndian.Uint64(proofInstance[1:9])
	commitment = proofInstance[9:]
	return
}

// ConstantTimeCompare performs constant-time comparison of two byte slices.
func ConstantTimeCompare(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
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

func ProofSignaturePayload(proofInstance, blindedSignature []byte) []byte {
	payload := make([]byte, 0, len(proofInstance)+len(blindedSignature))
	payload = append(payload, proofInstance...)
	payload = append(payload, blindedSignature...)
	return payload
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
