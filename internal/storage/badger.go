package storage

import (
	"encoding/binary"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/D13ya/DaZZLeD/internal/crypto"
	"github.com/dgraph-io/badger/v4"
)

var (
	ErrStoreClosed = errors.New("store is closed")
	ErrKeyNotFound = errors.New("key not found")
	ErrInvalidData = errors.New("invalid data format")
)

// Key prefixes for different data types
var (
	prefixProofInstance   = []byte("pi:") // Proof instances by epoch
	prefixMembershipProof = []byte("mp:") // Membership proofs by epoch
	prefixHashEntry       = []byte("he:") // Hash entries (blinded hashes)
	prefixMetadata        = []byte("md:") // Store metadata
)

// ProofStore exposes proof retrieval for the current epoch.
type ProofStore interface {
	GetProofBundle(epoch uint64) (proofInstance []byte, membershipProof []byte, err error)
}

// HashStore provides hash database operations.
type HashStore interface {
	StoreHash(hash []byte, metadata []byte) error
	LookupHash(hash []byte) (found bool, metadata []byte, err error)
	DeleteHash(hash []byte) error
}

// Store combines all storage interfaces.
type Store interface {
	ProofStore
	HashStore
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

// generateAndStoreProofBundle creates a new proof bundle and persists it.
func (s *BadgerStore) generateAndStoreProofBundle(epoch uint64) ([]byte, []byte, error) {
	commitment, err := crypto.NewCommitment()
	if err != nil {
		return nil, nil, fmt.Errorf("generate commitment: %w", err)
	}

	proofInstance := crypto.BuildProofInstance(crypto.ProofVersionV1, epoch, commitment)

	// For now, membership proof is a placeholder
	// In production, this would be the ML-DSA signature over the accumulator
	membershipProof, err := crypto.NewCommitment()
	if err != nil {
		return nil, nil, fmt.Errorf("generate membership proof: %w", err)
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

// StoreHash stores a hash entry in the database.
func (s *BadgerStore) StoreHash(hash []byte, metadata []byte) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return ErrStoreClosed
	}
	s.mu.RUnlock()

	key := makeHashKey(hash)
	return s.db.Update(func(txn *badger.Txn) error {
		return txn.Set(key, metadata)
	})
}

// LookupHash checks if a hash exists in the database.
func (s *BadgerStore) LookupHash(hash []byte) (bool, []byte, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return false, nil, ErrStoreClosed
	}
	s.mu.RUnlock()

	key := makeHashKey(hash)
	var metadata []byte

	err := s.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get(key)
		if err != nil {
			return err
		}
		metadata, err = item.ValueCopy(nil)
		return err
	})

	if errors.Is(err, badger.ErrKeyNotFound) {
		return false, nil, nil
	}
	if err != nil {
		return false, nil, err
	}

	return true, metadata, nil
}

// DeleteHash removes a hash entry from the database.
func (s *BadgerStore) DeleteHash(hash []byte) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return ErrStoreClosed
	}
	s.mu.RUnlock()

	key := makeHashKey(hash)
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

// makeHashKey creates a key for hash entries.
func makeHashKey(hash []byte) []byte {
	key := make([]byte, len(prefixHashEntry)+len(hash))
	copy(key, prefixHashEntry)
	copy(key[len(prefixHashEntry):], hash)
	return key
}
