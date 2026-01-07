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

	// MinAttestationSize is version(1) + timestamp(8) + nonce(16) + deviceID length(2) + sig(32)
	MinAttestationSize = 1 + 8 + AttestationNonceSize + 2 + AttestationSignatureSize

	// DefaultAttestationTTL is how long an attestation is valid
	DefaultAttestationTTL = 5 * time.Minute
)

// DeviceAttestation represents a signed device attestation blob.
// Format: version(1) | timestamp(8) | nonce(16) | deviceIDLen(2) | deviceID(var) | signature(32)
type DeviceAttestation struct {
	Version   byte
	Timestamp time.Time
	Nonce     []byte
	DeviceID  []byte
	Signature []byte
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
func GenerateAttestation(cfg AttestationConfig, deviceID []byte) ([]byte, error) {
	if len(deviceID) == 0 {
		return nil, ErrDeviceIDMissing
	}
	if len(deviceID) > 0xFFFF {
		return nil, ErrAttestationMalformed
	}

	// Generate random nonce
	nonce := make([]byte, AttestationNonceSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	timestamp := time.Now().UTC()

	// Build unsigned payload
	payloadSize := 1 + 8 + AttestationNonceSize + 2 + len(deviceID)
	payload := make([]byte, payloadSize)

	offset := 0
	payload[offset] = AttestationVersion
	offset++

	binary.BigEndian.PutUint64(payload[offset:], uint64(timestamp.UnixMilli()))
	offset += 8

	copy(payload[offset:], nonce)
	offset += AttestationNonceSize

	binary.BigEndian.PutUint16(payload[offset:], uint16(len(deviceID)))
	offset += 2

	copy(payload[offset:], deviceID)

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

	deviceIDLen := binary.BigEndian.Uint16(attestation[offset:])
	offset += 2

	expectedLen := 1 + 8 + AttestationNonceSize + 2 + int(deviceIDLen) + AttestationSignatureSize
	if len(attestation) != expectedLen {
		return nil, ErrAttestationMalformed
	}

	deviceID := make([]byte, deviceIDLen)
	copy(deviceID, attestation[offset:])
	offset += int(deviceIDLen)

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
		Version:   version,
		Timestamp: timestamp,
		Nonce:     nonce,
		DeviceID:  deviceID,
		Signature: signature,
	}, nil
}

// signAttestation creates an HMAC-SHA256 signature over the payload.
func signAttestation(secret, payload []byte) []byte {
	mac := hmac.New(sha256.New, secret)
	mac.Write(payload)
	return mac.Sum(nil)
}

// GenerateDeviceID creates a random device identifier.
// In production, this should come from device secure enclave or hardware ID.
func GenerateDeviceID() ([]byte, error) {
	id := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, id); err != nil {
		return nil, err
	}
	return id, nil
}

// DeriveAttestationSecret derives an attestation secret from a master key and device ID.
// Uses HKDF-like construction for key derivation.
func DeriveAttestationSecret(masterKey, deviceID []byte) []byte {
	h := hmac.New(sha256.New, masterKey)
	h.Write([]byte("dazzled-attestation-v1"))
	h.Write(deviceID)
	return h.Sum(nil)
}
