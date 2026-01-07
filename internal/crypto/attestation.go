package crypto

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"time"
)

// Attestation errors
var (
	ErrAttestationExpired   = errors.New("attestation has expired")
	ErrAttestationInvalid   = errors.New("attestation signature invalid")
	ErrAttestationMalformed = errors.New("attestation data malformed")
	ErrDeviceIDMissing      = errors.New("device ID is required")
)

const (
	// AttestationVersion is the current attestation format version
	AttestationVersion byte = 1

	// AttestationNonceSize is the size of the random nonce
	AttestationNonceSize = 16

	// AttestationSignatureSize is HMAC-SHA256 output size
	AttestationSignatureSize = 32

	// MinAttestationSize is version(1) + timestamp(8) + nonce(16) + epochToken(32) + sig(32)
	MinAttestationSize = 1 + 8 + AttestationNonceSize + 32 + AttestationSignatureSize

	// DefaultAttestationTTL is how long an attestation is valid
	DefaultAttestationTTL = 5 * time.Minute
)

// DeviceAttestation represents a signed device attestation blob.
// Format: version(1) | timestamp(8) | nonce(16) | epochToken(32) | signature(32)
// Note: We use an unlinkable epoch-bound pseudonymous token instead of a stable device ID
// to prevent cross-scan device tracking (addressing Apple-PSI style surveillance concerns).
type DeviceAttestation struct {
	Version    byte
	Timestamp  time.Time
	Nonce      []byte
	EpochToken []byte // Unlinkable per-epoch pseudonymous token (NOT a stable device ID)
	Signature  []byte
}

// AttestationConfig holds attestation generation/verification settings.
type AttestationConfig struct {
	// Secret key for HMAC signing (should be derived from device secure enclave)
	Secret []byte
	// TTL for attestation validity
	TTL time.Duration
}

// NewAttestationConfig creates a config with sensible defaults.
func NewAttestationConfig(secret []byte) AttestationConfig {
	return AttestationConfig{
		Secret: secret,
		TTL:    DefaultAttestationTTL,
	}
}

// GenerateAttestation creates a new device attestation blob.
// Instead of embedding a stable device ID (which enables tracking), we generate
// an unlinkable pseudonymous token that rotates each epoch.
func GenerateAttestation(cfg AttestationConfig, deviceSecret []byte) ([]byte, error) {
	if len(deviceSecret) == 0 {
		return nil, ErrDeviceIDMissing
	}

	// Generate random nonce
	nonce := make([]byte, AttestationNonceSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	timestamp := time.Now().UTC()

	// Generate unlinkable epoch token: HMAC(deviceSecret, epoch || nonce)
	// This token is unique per attestation but cannot be linked across scans
	epochID := uint64(timestamp.Unix() / 86400) // Daily epoch
	epochToken := deriveEpochToken(deviceSecret, epochID, nonce)

	// Build unsigned payload (fixed size, no variable-length device ID)
	// Format: version(1) | timestamp(8) | nonce(16) | epochToken(32)
	const payloadSize = 1 + 8 + AttestationNonceSize + 32
	payload := make([]byte, payloadSize)

	offset := 0
	payload[offset] = AttestationVersion
	offset++

	binary.BigEndian.PutUint64(payload[offset:], uint64(timestamp.UnixMilli()))
	offset += 8

	copy(payload[offset:], nonce)
	offset += AttestationNonceSize

	copy(payload[offset:], epochToken)

	// Sign the payload
	signature := signAttestation(cfg.Secret, payload)

	// Combine payload + signature
	result := make([]byte, len(payload)+len(signature))
	copy(result, payload)
	copy(result[len(payload):], signature)

	return result, nil
}

// VerifyAttestation validates an attestation blob.
func VerifyAttestation(cfg AttestationConfig, attestation []byte) (*DeviceAttestation, error) {
	if len(attestation) < MinAttestationSize {
		return nil, ErrAttestationMalformed
	}

	// Parse the attestation
	offset := 0

	version := attestation[offset]
	if version != AttestationVersion {
		return nil, ErrAttestationMalformed
	}
	offset++

	timestampMs := binary.BigEndian.Uint64(attestation[offset:])
	timestamp := time.UnixMilli(int64(timestampMs))
	offset += 8

	nonce := make([]byte, AttestationNonceSize)
	copy(nonce, attestation[offset:])
	offset += AttestationNonceSize

	// New format: fixed 32-byte epoch token (no variable-length device ID)
	const epochTokenSize = 32
	expectedLen := 1 + 8 + AttestationNonceSize + epochTokenSize + AttestationSignatureSize
	if len(attestation) != expectedLen {
		return nil, ErrAttestationMalformed
	}

	epochToken := make([]byte, epochTokenSize)
	copy(epochToken, attestation[offset:])
	offset += epochTokenSize

	signature := attestation[offset:]

	// Verify signature
	payload := attestation[:offset]
	expectedSig := signAttestation(cfg.Secret, payload)
	if !hmac.Equal(signature, expectedSig) {
		return nil, ErrAttestationInvalid
	}

	// Check expiration
	if time.Since(timestamp) > cfg.TTL {
		return nil, ErrAttestationExpired
	}

	// Check for future timestamps (clock skew tolerance: 1 minute)
	if timestamp.After(time.Now().Add(1 * time.Minute)) {
		return nil, ErrAttestationInvalid
	}

	return &DeviceAttestation{
		Version:    version,
		Timestamp:  timestamp,
		Nonce:      nonce,
		EpochToken: epochToken,
		Signature:  signature,
	}, nil
}

// signAttestation creates an HMAC-SHA256 signature over the payload.
func signAttestation(secret, payload []byte) []byte {
	mac := hmac.New(sha256.New, secret)
	mac.Write(payload)
	return mac.Sum(nil)
}

// deriveEpochToken generates an unlinkable pseudonymous token for the current epoch.
// This token cannot be linked across epochs or to the device identity.
func deriveEpochToken(deviceSecret []byte, epochID uint64, nonce []byte) []byte {
	h := hmac.New(sha256.New, deviceSecret)
	h.Write([]byte("dazzled-epoch-token-v1"))
	epochBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(epochBytes, epochID)
	h.Write(epochBytes)
	h.Write(nonce)
	return h.Sum(nil)
}

// GenerateDeviceSecret creates a random device secret for attestation.
// In production, this MUST come from device secure enclave or HSM.
// This secret should be stored securely and never transmitted.
func GenerateDeviceSecret() ([]byte, error) {
	secret := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, secret); err != nil {
		return nil, err
	}
	return secret, nil
}

// GenerateDeviceID is deprecated - use GenerateDeviceSecret instead.
// Kept for backward compatibility during migration.
func GenerateDeviceID() ([]byte, error) {
	return GenerateDeviceSecret()
}
