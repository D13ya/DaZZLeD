package storage

import (
	"sync"

	"github.com/D13ya/DaZZLeD/internal/crypto"
)

// ProofStore exposes proof retrieval for the current epoch.
type ProofStore interface {
	GetProofBundle(epoch uint64) (proofInstance []byte, membershipProof []byte, err error)
}

// BadgerStore is a placeholder in-memory implementation.
type BadgerStore struct {
	mu              sync.Mutex
	epochID         uint64
	proofInstance   []byte
	membershipProof []byte
}

func NewBadgerStore() *BadgerStore {
	return &BadgerStore{}
}

func (s *BadgerStore) GetProofBundle(epoch uint64) ([]byte, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if epoch == s.epochID && len(s.proofInstance) > 0 {
		return s.proofInstance, s.membershipProof, nil
	}

	commitment, err := crypto.NewCommitment()
	if err != nil {
		return nil, nil, err
	}
	proofInstance := crypto.BuildProofInstance(1, epoch, commitment)

	membershipProof, err := crypto.NewCommitment()
	if err != nil {
		return nil, nil, err
	}

	s.epochID = epoch
	s.proofInstance = proofInstance
	s.membershipProof = membershipProof
	return proofInstance, membershipProof, nil
}
