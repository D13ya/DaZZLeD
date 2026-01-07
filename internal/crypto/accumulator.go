package crypto

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"io"
	"time"

	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
)

const (
	// ProofVersionV1 is the unsigned proof format (deprecated)
	ProofVersionV1 byte = 1
	// ProofVersionV2 is the signed commitment format
	ProofVersionV2 byte = 2

	// MinProofInstanceSizeV1 is version(1) + epoch(8) + commitment(288) - legacy unsigned
	MinProofInstanceSizeV1 = 1 + 8 + ProofCommitmentSize

	// MinProofInstanceSize is version(1) + epoch(8) + commitment(288) + signature(3309)
	// This is the current format with signed set digest
	MinProofInstanceSize = 1 + 8 + SignedCommitmentSize
)

var (
	ErrProofTooShort        = errors.New("proof instance too short")
	ErrInvalidVersion       = errors.New("unsupported proof version")
	ErrMembershipInvalid    = errors.New("membership proof invalid")
	ErrCommitmentSigInvalid = errors.New("commitment signature invalid")
	ErrCommitmentSigMissing = errors.New("commitment signature missing")
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
	if len(proofInstance) < MinProofInstanceSizeV1 {
		return ErrProofTooShort
	}
	version := proofInstance[0]
	if version != ProofVersionV1 && version != ProofVersionV2 {
		return ErrInvalidVersion
	}
	// V2 requires the full signed commitment
	if version == ProofVersionV2 && len(proofInstance) < MinProofInstanceSize {
		return ErrCommitmentSigMissing
	}
	return nil
}

// ValidateMembershipProof checks the membership proof structure.
// For ML-DSA-65, signatures are 3309 bytes.
func ValidateMembershipProof(membershipProof []byte) error {
	// ML-DSA-65 signature size is 3309 bytes
	if len(membershipProof) != CommitmentSignatureSize {
		return ErrMembershipInvalid
	}
	return nil
}

// ParseProofInstance extracts version, epoch, and commitment from a proof instance.
// For V2, commitment includes the signature appended.
func ParseProofInstance(proofInstance []byte) (version byte, epochID uint64, commitment []byte, err error) {
	if err = ValidateProofInstance(proofInstance); err != nil {
		return
	}
	version = proofInstance[0]
	epochID = binary.BigEndian.Uint64(proofInstance[1:9])
	commitment = proofInstance[9:]
	return
}

// ParseSignedProofInstance extracts all components including the commitment signature.
// Returns: version, epochID, rawCommitment (bloom+root), commitmentSig
func ParseSignedProofInstance(proofInstance []byte) (version byte, epochID uint64, rawCommitment, commitmentSig []byte, err error) {
	version, epochID, fullCommitment, err := ParseProofInstance(proofInstance)
	if err != nil {
		return
	}

	if version == ProofVersionV1 {
		// V1 has no signature - return empty sig
		rawCommitment = fullCommitment
		commitmentSig = nil
		return
	}

	// V2: commitment = rawCommitment(288) || signature(3309)
	if len(fullCommitment) < SignedCommitmentSize {
		err = ErrCommitmentSigMissing
		return
	}

	rawCommitment = fullCommitment[:ProofCommitmentSize]
	commitmentSig = fullCommitment[ProofCommitmentSize:]
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

// VerifyMembershipWithOPRF checks if the unblinded OPRF output is in the known-bad set.
//
// PSI MEMBERSHIP DESIGN:
// The server CANNOT know if there's a match (that's the "oblivious" part of OPRF).
// Instead, the server provides a SET DIGEST (Bloom filter) of all F(key, bad_hash) values.
// The client unblinds their OPRF output and checks against this digest.
//
// The set digest is embedded in the proof instance commitment:
// - commitment = bloomFilter(256 bytes) || accumulatorRoot(32 bytes)
//
// Note: The membershipProof (ML-DSA signature) is verified separately in verifyServerResponse.
// This function only checks Bloom filter membership.
//
// Returns true if the OPRF output appears to be in the set (may have false positives).
func VerifyMembershipWithOPRF(oprfOutput, proofInstance []byte) bool {
	// Validate inputs
	if len(oprfOutput) == 0 || len(proofInstance) < MinProofInstanceSize {
		return false
	}

	// Parse proof instance to extract commitment (contains set digest)
	_, _, commitment, err := ParseProofInstance(proofInstance)
	if err != nil {
		return false
	}

	// The commitment contains the Bloom filter for set membership
	// Format: bloomFilter(BloomFilterSize) || accumulatorRoot(32)
	if len(commitment) < BloomFilterSize {
		return false
	}

	bloomFilter := commitment[:BloomFilterSize]

	// Check if OPRF output is in the Bloom filter
	return CheckBloomFilter(bloomFilter, oprfOutput)
}

// BloomFilterSize is the size of the Bloom filter in bytes (2048 bits = 256 bytes)
const BloomFilterSize = 256

// BloomHashCount is the number of hash functions used in the Bloom filter
const BloomHashCount = 8

// CheckBloomFilter checks if an item might be in the Bloom filter.
// Returns true if the item is probably in the set (may have false positives).
// Returns false if the item is definitely not in the set.
func CheckBloomFilter(filter, item []byte) bool {
	if len(filter) != BloomFilterSize {
		return false
	}

	// Compute k hash positions using double hashing: h(i) = h1 + i*h2
	h := sha256.Sum256(item)
	h1 := binary.BigEndian.Uint64(h[0:8])
	h2 := binary.BigEndian.Uint64(h[8:16])

	numBits := uint64(BloomFilterSize * 8)

	for i := uint64(0); i < BloomHashCount; i++ {
		bitPos := (h1 + i*h2) % numBits
		bytePos := bitPos / 8
		bitOffset := bitPos % 8

		if filter[bytePos]&(1<<bitOffset) == 0 {
			return false // Definitely not in set
		}
	}

	return true // Probably in set
}

// BuildBloomFilter creates a Bloom filter from a set of OPRF outputs.
// This is used by the server to build the set digest.
func BuildBloomFilter(oprfOutputs [][]byte) []byte {
	filter := make([]byte, BloomFilterSize)

	for _, item := range oprfOutputs {
		// Compute k hash positions using double hashing
		h := sha256.Sum256(item)
		h1 := binary.BigEndian.Uint64(h[0:8])
		h2 := binary.BigEndian.Uint64(h[8:16])

		numBits := uint64(BloomFilterSize * 8)

		for i := uint64(0); i < BloomHashCount; i++ {
			bitPos := (h1 + i*h2) % numBits
			bytePos := bitPos / 8
			bitOffset := bitPos % 8

			filter[bytePos] |= 1 << bitOffset
		}
	}

	return filter
}

// BuildProofInstanceWithBloomFilter creates a proof instance with a Bloom filter set digest.
// DEPRECATED: Use BuildSignedProofInstance for signed commitments.
// This creates an unsigned V1 proof instance for backward compatibility.
func BuildProofInstanceWithBloomFilter(version byte, epochID uint64, bloomFilter, accumulatorRoot []byte) []byte {
	// Commitment = bloomFilter(256) || accumulatorRoot(32)
	if len(bloomFilter) != BloomFilterSize {
		return nil
	}
	// Enforce accumulator root is exactly AccumulatorRootSize (32 bytes)
	if len(accumulatorRoot) != AccumulatorRootSize {
		return nil
	}

	commitment := make([]byte, BloomFilterSize+AccumulatorRootSize)
	copy(commitment[:BloomFilterSize], bloomFilter)
	copy(commitment[BloomFilterSize:], accumulatorRoot)

	return BuildProofInstance(ProofVersionV1, epochID, commitment)
}

// CommitmentSigningPayload creates the message to be signed for a commitment.
// Format: domain_separator || epoch_id(8) || commitment(288)
// The domain separator prevents cross-protocol signature attacks.
func CommitmentSigningPayload(epochID uint64, commitment []byte) []byte {
	domain := []byte(CommitmentDomainSeparator)
	payload := make([]byte, len(domain)+8+len(commitment))
	copy(payload, domain)
	binary.BigEndian.PutUint64(payload[len(domain):], epochID)
	copy(payload[len(domain)+8:], commitment)
	return payload
}

// SignCommitment creates an ML-DSA signature over the commitment.
// This is the core of the "Signed Set Digest" - it proves the authority
// approved this specific Bloom filter for this epoch.
func SignCommitment(privateKey *mldsa65.PrivateKey, epochID uint64, commitment []byte) ([]byte, error) {
	payload := CommitmentSigningPayload(epochID, commitment)
	return SignMLDSA(privateKey, payload)
}

// VerifyCommitmentSignature verifies the authority's signature over a commitment.
// This is what prevents the "opaque list" attack - clients can verify the
// Bloom filter was approved by the trusted authority.
func VerifyCommitmentSignature(publicKey *mldsa65.PublicKey, epochID uint64, commitment, signature []byte) bool {
	payload := CommitmentSigningPayload(epochID, commitment)
	return VerifyMLDSA(publicKey, payload, signature)
}

// BuildSignedProofInstance creates a V2 proof instance with signed commitment.
// This is the production format that addresses the "opaque list" criticism.
//
// Format:
//
//	version(1) || epoch_id(8) || commitment(288) || commitment_sig(3309)
//
// The commitment_sig proves the authority signed this specific Bloom filter.
func BuildSignedProofInstance(privateKey *mldsa65.PrivateKey, epochID uint64, bloomFilter, accumulatorRoot []byte) ([]byte, error) {
	// Validate inputs
	if len(bloomFilter) != BloomFilterSize {
		return nil, errors.New("invalid bloom filter size")
	}
	if len(accumulatorRoot) != AccumulatorRootSize {
		return nil, errors.New("invalid accumulator root size")
	}

	// Build raw commitment
	rawCommitment := make([]byte, ProofCommitmentSize)
	copy(rawCommitment[:BloomFilterSize], bloomFilter)
	copy(rawCommitment[BloomFilterSize:], accumulatorRoot)

	// Sign the commitment
	sig, err := SignCommitment(privateKey, epochID, rawCommitment)
	if err != nil {
		return nil, err
	}

	// Build signed commitment = rawCommitment || signature
	signedCommitment := make([]byte, SignedCommitmentSize)
	copy(signedCommitment[:ProofCommitmentSize], rawCommitment)
	copy(signedCommitment[ProofCommitmentSize:], sig)

	// Build final proof instance
	return BuildProofInstance(ProofVersionV2, epochID, signedCommitment), nil
}

// VerifyProofInstanceSignature verifies the commitment signature in a V2 proof instance.
// Returns true if the signature is valid, false otherwise.
// For V1 instances (unsigned), this returns false.
func VerifyProofInstanceSignature(publicKey *mldsa65.PublicKey, proofInstance []byte) bool {
	version, epochID, rawCommitment, commitmentSig, err := ParseSignedProofInstance(proofInstance)
	if err != nil {
		return false
	}

	// V1 has no signature to verify
	if version == ProofVersionV1 {
		return false
	}

	return VerifyCommitmentSignature(publicKey, epochID, rawCommitment, commitmentSig)
}
