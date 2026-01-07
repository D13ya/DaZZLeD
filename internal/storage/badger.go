package storage

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/cloudflare/circl/sign/mldsa/mldsa65"
	"github.com/dgraph-io/badger/v4"
)

var (
	ErrStoreClosed    = errors.New("store is closed")
	ErrKeyNotFound    = errors.New("key not found")
	ErrInvalidData    = errors.New("invalid data format")
	ErrSignerRequired = errors.New("ML-DSA signer required for proof generation")
)

// Key prefixes for different data types
var (
	prefixProofInstance   = []byte("pi:") // Proof instances by epoch
	prefixMembershipProof = []byte("mp:") // Membership proofs by epoch
	prefixOPRFOutput      = []byte("op:") // OPRF outputs for known-bad hashes
)

// ProofStore exposes proof retrieval for the current epoch.
type ProofStore interface {
	GetProofBundle(epoch uint64) (proofInstance []byte, membershipProof []byte, err error)
}

// SignedProofStore generates signed proof bundles.
type SignedProofStore interface {
	// GetSignedProofBundle retrieves or generates a signed proof bundle.
	// The signer is used to create the commitment signature (V2 format).
	GetSignedProofBundle(epoch uint64, signer *mldsa65.PrivateKey) (proofInstance []byte, err error)
}

// OPRFOutputStore provides storage for OPRF outputs of known-bad hashes.
// The server pre-computes F(key, hash) for each known-bad hash and stores it.
type OPRFOutputStore interface {
	StoreOPRFOutput(oprfOutput []byte, metadata []byte) error
	GetAllOPRFOutputs() ([][]byte, error)
	DeleteOPRFOutput(oprfOutput []byte) error
}

// Store combines all storage interfaces.
type Store interface {
	ProofStore
	SignedProofStore
	OPRFOutputStore
	Close() error
}

// BadgerStore implements persistent storage using BadgerDB.
type BadgerStore struct {
	db     *badger.DB
	mu     sync.RWMutex
	closed bool

	// Cache for current epoch's proof bundle
	cache struct {
		sync.RWMutex
		epochID         uint64
		proofInstance   []byte
		membershipProof []byte
	}
}

// BadgerConfig holds configuration for BadgerStore.
type BadgerConfig struct {
	// Directory for database files
	Dir string
	// InMemory runs BadgerDB in memory-only mode (for testing)
	InMemory bool
	// SyncWrites ensures durability at cost of performance
	SyncWrites bool
	// GCInterval for value log garbage collection (0 disables)
	GCInterval time.Duration
}

// DefaultBadgerConfig returns sensible defaults for production.
func DefaultBadgerConfig(baseDir string) BadgerConfig {
	return BadgerConfig{
		Dir:        filepath.Join(baseDir, "data", "badger"),
		InMemory:   false,
		SyncWrites: true,
		GCInterval: 5 * time.Minute,
	}
}

// NewBadgerStore creates a new BadgerDB-backed store.
func NewBadgerStore() *BadgerStore {
	// Default to in-memory for backward compatibility
	return NewBadgerStoreWithConfig(BadgerConfig{InMemory: true})
}

// NewBadgerStoreWithConfig creates a store with explicit configuration.
func NewBadgerStoreWithConfig(cfg BadgerConfig) *BadgerStore {
	opts := badger.DefaultOptions(cfg.Dir)
	opts.InMemory = cfg.InMemory
	opts.SyncWrites = cfg.SyncWrites
	opts.Logger = nil // Disable badger's verbose logging

	if cfg.InMemory {
		opts.Dir = ""
		opts.ValueDir = ""
	}

	db, err := badger.Open(opts)
	if err != nil {
		// Fall back to in-memory if disk fails
		opts.InMemory = true
		opts.Dir = ""
		opts.ValueDir = ""
		db, _ = badger.Open(opts)
	}

	store := &BadgerStore{db: db}

	// Start GC goroutine if configured
	if cfg.GCInterval > 0 && !cfg.InMemory {
		go store.runGC(cfg.GCInterval)
	}

	return store
}

// Close closes the database.
func (s *BadgerStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true

	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// runGC periodically runs value log garbage collection.
func (s *BadgerStore) runGC(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		s.mu.RLock()
		if s.closed {
			s.mu.RUnlock()
			return
		}
		s.mu.RUnlock()

		// Run GC until no more cleanup needed
		for {
			err := s.db.RunValueLogGC(0.5)
			if err != nil {
				break
			}
		}
	}
}

// GetProofBundle retrieves or generates the proof bundle for an epoch.
// DEPRECATED: Use GetSignedProofBundle for signed V2 proof instances.
func (s *BadgerStore) GetProofBundle(epoch uint64) ([]byte, []byte, error) {
	if err := s.ensureOpen(); err != nil {
		return nil, nil, err
	}

	if pi, mp, ok := s.getFromCache(epoch); ok {
		return pi, mp, nil
	}

	proofInstance, membershipProof, err := s.loadOrGenerateBundle(epoch)
	if err != nil {
		return nil, nil, err
	}

	s.updateCache(epoch, proofInstance, membershipProof)
	return proofInstance, membershipProof, nil
}

// GetSignedProofBundle retrieves or generates a V2 signed proof bundle.
// The signer is used to create the commitment signature, proving the authority
// approved this specific Bloom filter (addressing the "opaque list" criticism).
func (s *BadgerStore) GetSignedProofBundle(epoch uint64, signer *mldsa65.PrivateKey) ([]byte, error) {
	if signer == nil {
		return nil, ErrSignerRequired
	}
	if err := s.ensureOpen(); err != nil {
		return nil, err
	}

	// Check cache first
	if pi, _, ok := s.getFromCache(epoch); ok && len(pi) > 0 {
		// Verify it's a V2 instance
		if pi[0] == crypto.ProofVersionV2 {
			return pi, nil
		}
		// Cached is V1, need to regenerate as V2
	}

	// Try to load from DB
	proofInstance, _, err := s.loadBundleFromDB(epoch)
	if err != nil {
		return nil, err
	}

	// If we have a V2 instance, return it
	if len(proofInstance) > 0 && proofInstance[0] == crypto.ProofVersionV2 {
		s.updateCache(epoch, proofInstance, nil)
		return proofInstance, nil
	}

	// Generate new signed proof instance
	proofInstance, err = s.generateSignedProofBundle(epoch, signer)
	if err != nil {
		return nil, err
	}

	s.updateCache(epoch, proofInstance, nil)
	return proofInstance, nil
}

func (s *BadgerStore) ensureOpen() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return ErrStoreClosed
	}
	return nil
}

func (s *BadgerStore) getFromCache(epoch uint64) ([]byte, []byte, bool) {
	s.cache.RLock()
	defer s.cache.RUnlock()
	if epoch == s.cache.epochID && len(s.cache.proofInstance) > 0 {
		return s.cache.proofInstance, s.cache.membershipProof, true
	}
	return nil, nil, false
}

func (s *BadgerStore) updateCache(epoch uint64, proofInstance, membershipProof []byte) {
	s.cache.Lock()
	s.cache.epochID = epoch
	s.cache.proofInstance = proofInstance
	s.cache.membershipProof = membershipProof
	s.cache.Unlock()
}

// InvalidateEpochCache clears the cache for a specific epoch, forcing regeneration.
// This is used when OPRF outputs are added/removed and the Bloom filter needs rebuilding.
func (s *BadgerStore) InvalidateEpochCache(epoch uint64) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return ErrStoreClosed
	}
	s.mu.RUnlock()

	// Clear the in-memory cache
	s.cache.Lock()
	if s.cache.epochID == epoch {
		s.cache.epochID = 0
		s.cache.proofInstance = nil
		s.cache.membershipProof = nil
	}
	s.cache.Unlock()

	// Delete the stored proof bundle from DB to force regeneration
	piKey := makeEpochKey(prefixProofInstance, epoch)
	mpKey := makeEpochKey(prefixMembershipProof, epoch)

	return s.db.Update(func(txn *badger.Txn) error {
		if err := txn.Delete(piKey); err != nil && err != badger.ErrKeyNotFound {
			return err
		}
		if err := txn.Delete(mpKey); err != nil && err != badger.ErrKeyNotFound {
			return err
		}
		return nil
	})
}

func (s *BadgerStore) loadOrGenerateBundle(epoch uint64) ([]byte, []byte, error) {
	proofInstance, membershipProof, err := s.loadBundleFromDB(epoch)
	if err != nil {
		return nil, nil, err
	}

	if len(proofInstance) == 0 {
		return s.generateAndStoreProofBundle(epoch)
	}
	return proofInstance, membershipProof, nil
}

func (s *BadgerStore) loadBundleFromDB(epoch uint64) ([]byte, []byte, error) {
	piKey := makeEpochKey(prefixProofInstance, epoch)
	mpKey := makeEpochKey(prefixMembershipProof, epoch)

	var proofInstance, membershipProof []byte

	err := s.db.View(func(txn *badger.Txn) error {
		if item, err := txn.Get(piKey); err == nil {
			if val, err := item.ValueCopy(nil); err == nil {
				proofInstance = val
			} else {
				return err
			}
		}

		if item, err := txn.Get(mpKey); err == nil {
			if val, err := item.ValueCopy(nil); err == nil {
				membershipProof = val
			} else {
				return err
			}
		}

		return nil
	})

	if err != nil && !errors.Is(err, badger.ErrKeyNotFound) {
		return nil, nil, fmt.Errorf("load proof bundle: %w", err)
	}

	return proofInstance, membershipProof, nil
}

// generateAndStoreProofBundle creates a new proof bundle with Bloom filter set digest.
func (s *BadgerStore) generateAndStoreProofBundle(epoch uint64) ([]byte, []byte, error) {
	// Collect all OPRF outputs for known-bad hashes
	oprfOutputs, err := s.GetAllOPRFOutputs()
	if err != nil {
		return nil, nil, fmt.Errorf("collect OPRF outputs: %w", err)
	}

	// Build Bloom filter from all OPRF outputs
	bloomFilter := crypto.BuildBloomFilter(oprfOutputs)

	// Generate accumulator root (fixed 32 bytes - in production this would be a Merkle root)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)
	if _, err := rand.Read(accumulatorRoot); err != nil {
		return nil, nil, fmt.Errorf("generate accumulator root: %w", err)
	}

	// Build proof instance with Bloom filter embedded in commitment
	proofInstance := crypto.BuildProofInstanceWithBloomFilter(
		crypto.ProofVersionV1,
		epoch,
		bloomFilter,
		accumulatorRoot,
	)
	if proofInstance == nil {
		return nil, nil, fmt.Errorf("build proof instance with bloom filter")
	}

	// For now, membership proof is a placeholder signature
	// In production, this would be the ML-DSA signature over the proofInstance
	membershipProof := make([]byte, 3309) // ML-DSA-65 signature size
	if _, err := rand.Read(membershipProof); err != nil {
		return nil, nil, fmt.Errorf("generate membership proof placeholder: %w", err)
	}

	// Persist to database
	piKey := makeEpochKey(prefixProofInstance, epoch)
	mpKey := makeEpochKey(prefixMembershipProof, epoch)

	err = s.db.Update(func(txn *badger.Txn) error {
		if err := txn.Set(piKey, proofInstance); err != nil {
			return err
		}
		return txn.Set(mpKey, membershipProof)
	})

	if err != nil {
		return nil, nil, fmt.Errorf("persist proof bundle: %w", err)
	}

	return proofInstance, membershipProof, nil
}

// generateSignedProofBundle creates a V2 proof bundle with signed commitment.
// This addresses the "opaque list" criticism by having the authority sign
// the Bloom filter commitment, allowing clients to verify the set is legitimate.
func (s *BadgerStore) generateSignedProofBundle(epoch uint64, signer *mldsa65.PrivateKey) ([]byte, error) {
	// Collect all OPRF outputs for known-bad hashes
	oprfOutputs, err := s.GetAllOPRFOutputs()
	if err != nil {
		return nil, fmt.Errorf("collect OPRF outputs: %w", err)
	}

	// Build Bloom filter from all OPRF outputs
	bloomFilter := crypto.BuildBloomFilter(oprfOutputs)

	// Generate accumulator root (fixed 32 bytes - in production this would be a Merkle root)
	accumulatorRoot := make([]byte, crypto.AccumulatorRootSize)
	if _, err := rand.Read(accumulatorRoot); err != nil {
		return nil, fmt.Errorf("generate accumulator root: %w", err)
	}

	// Build V2 signed proof instance with ML-DSA signature over commitment
	proofInstance, err := crypto.BuildSignedProofInstance(
		signer,
		epoch,
		bloomFilter,
		accumulatorRoot,
	)
	if err != nil {
		return nil, fmt.Errorf("build signed proof instance: %w", err)
	}

	// Persist to database
	piKey := makeEpochKey(prefixProofInstance, epoch)

	err = s.db.Update(func(txn *badger.Txn) error {
		return txn.Set(piKey, proofInstance)
	})

	if err != nil {
		return nil, fmt.Errorf("persist signed proof bundle: %w", err)
	}

	return proofInstance, nil
}

// StoreOPRFOutput stores an OPRF output (F(key, hash)) for a known-bad hash.
func (s *BadgerStore) StoreOPRFOutput(oprfOutput []byte, metadata []byte) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return ErrStoreClosed
	}
	s.mu.RUnlock()

	key := makeOPRFOutputKey(oprfOutput)
	return s.db.Update(func(txn *badger.Txn) error {
		return txn.Set(key, metadata)
	})
}

// GetAllOPRFOutputs retrieves all stored OPRF outputs for building Bloom filter.
func (s *BadgerStore) GetAllOPRFOutputs() ([][]byte, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, ErrStoreClosed
	}
	s.mu.RUnlock()

	var outputs [][]byte

	err := s.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.Prefix = prefixOPRFOutput
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			key := item.Key()
			// Extract OPRF output from key (strip prefix)
			oprfOutput := make([]byte, len(key)-len(prefixOPRFOutput))
			copy(oprfOutput, key[len(prefixOPRFOutput):])
			outputs = append(outputs, oprfOutput)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return outputs, nil
}

// DeleteOPRFOutput removes an OPRF output from the database.
func (s *BadgerStore) DeleteOPRFOutput(oprfOutput []byte) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return ErrStoreClosed
	}
	s.mu.RUnlock()

	key := makeOPRFOutputKey(oprfOutput)
	return s.db.Update(func(txn *badger.Txn) error {
		return txn.Delete(key)
	})
}

// makeEpochKey creates a key for epoch-indexed data.
func makeEpochKey(prefix []byte, epoch uint64) []byte {
	key := make([]byte, len(prefix)+8)
	copy(key, prefix)
	binary.BigEndian.PutUint64(key[len(prefix):], epoch)
	return key
}

// makeOPRFOutputKey creates a key for OPRF output entries.
func makeOPRFOutputKey(oprfOutput []byte) []byte {
	key := make([]byte, len(prefixOPRFOutput)+len(oprfOutput))
	copy(key, prefixOPRFOutput)
	copy(key[len(prefixOPRFOutput):], oprfOutput)
	return key
}
