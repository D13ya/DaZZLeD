package crypto

const (
	// AccumulatorRootSize is the size of the Merkle/accumulator root (SHA-256)
	AccumulatorRootSize = 32

	// CommitmentSignatureSize is the size of ML-DSA-65 signature over the commitment
	CommitmentSignatureSize = 3309

	// ProofCommitmentSize is the size of the raw commitment (unsigned)
	// Format: bloomFilter(256) || accumulatorRoot(32) = 288 bytes
	ProofCommitmentSize = BloomFilterSize + AccumulatorRootSize

	// SignedCommitmentSize includes the commitment plus its ML-DSA signature
	// Format: commitment(288) || commitment_signature(3309) = 3597 bytes
	SignedCommitmentSize = ProofCommitmentSize + CommitmentSignatureSize

	// UploadTokenSize is the size of upload tokens
	UploadTokenSize = 32

	// CommitmentDomainSeparator is the domain separator for commitment signatures
	// This ensures signatures are bound to this specific use case
	CommitmentDomainSeparator = "dazzled-set-digest-v1"
)
