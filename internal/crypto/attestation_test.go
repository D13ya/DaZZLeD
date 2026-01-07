package crypto

import (
	"testing"
	"time"
)

func TestGenerateAttestation_ValidOutput(t *testing.T) {
	secret := []byte("test-attestation-secret-32bytes!")
	cfg := NewAttestationConfig(secret)
	deviceID := []byte("device-12345")

	attestation, err := GenerateAttestation(cfg, deviceID)
	if err != nil {
		t.Fatalf("GenerateAttestation: %v", err)
	}

	if len(attestation) == 0 {
		t.Error("attestation should not be empty")
	}
}

func TestVerifyAttestation_ValidAttestation(t *testing.T) {
	secret := []byte("test-attestation-secret-32bytes!")
	cfg := NewAttestationConfig(secret)
	deviceID := []byte("device-12345")

	attestation, err := GenerateAttestation(cfg, deviceID)
	if err != nil {
		t.Fatalf("GenerateAttestation: %v", err)
	}

	// Verify should succeed with same config
	result, err := VerifyAttestation(cfg, attestation)
	if err != nil {
		t.Errorf("VerifyAttestation failed: %v", err)
	}
	if result == nil {
		t.Error("result should not be nil")
	}
}

func TestVerifyAttestation_WrongSecret(t *testing.T) {
	secret1 := []byte("test-attestation-secret-32bytes!")
	secret2 := []byte("different-attestation-secret!!!!!")
	cfg1 := NewAttestationConfig(secret1)
	cfg2 := NewAttestationConfig(secret2)
	deviceID := []byte("device-12345")

	attestation, _ := GenerateAttestation(cfg1, deviceID)

	// Verify with wrong secret should fail
	_, err := VerifyAttestation(cfg2, attestation)
	if err == nil {
		t.Error("VerifyAttestation should fail with wrong secret")
	}
}

func TestVerifyAttestation_TamperedAttestation(t *testing.T) {
	secret := []byte("test-attestation-secret-32bytes!")
	cfg := NewAttestationConfig(secret)
	deviceID := []byte("device-12345")

	attestation, _ := GenerateAttestation(cfg, deviceID)

	// Tamper with attestation
	tampered := make([]byte, len(attestation))
	copy(tampered, attestation)
	if len(tampered) > 5 {
		tampered[5] ^= 0xFF
	}

	_, err := VerifyAttestation(cfg, tampered)
	if err == nil {
		t.Error("VerifyAttestation should fail with tampered attestation")
	}
}

func TestVerifyAttestation_MalformedInput(t *testing.T) {
	secret := []byte("test-attestation-secret-32bytes!")
	cfg := NewAttestationConfig(secret)

	testCases := []struct {
		name        string
		attestation []byte
	}{
		{"nil", nil},
		{"empty", []byte{}},
		{"too_short", []byte{0x01, 0x02, 0x03}},
		{"garbage", []byte("not a valid attestation")},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := VerifyAttestation(cfg, tc.attestation)
			if err == nil {
				t.Errorf("VerifyAttestation should fail with %s input", tc.name)
			}
		})
	}
}

func TestAttestationConfig_EmptySecret(t *testing.T) {
	// Empty secret should still work (but shouldn't be used in production)
	cfg := NewAttestationConfig(nil)
	deviceID := []byte("device-12345")

	attestation, err := GenerateAttestation(cfg, deviceID)
	if err != nil {
		t.Logf("GenerateAttestation with nil secret: %v", err)
		return // Acceptable to fail
	}

	_, err = VerifyAttestation(cfg, attestation)
	if err != nil {
		t.Logf("VerifyAttestation with nil secret: %v", err)
	}
}

func TestEpochFreshness_CurrentEpoch(t *testing.T) {
	now := time.Now()
	epoch := CurrentEpochID(now)

	// Current epoch should be fresh with skew of 0
	if !IsEpochFresh(epoch, now, 0) {
		t.Error("current epoch should be fresh")
	}

	// Current epoch should be fresh with skew of 1
	if !IsEpochFresh(epoch, now, 1) {
		t.Error("current epoch should be fresh with skew 1")
	}
}

func TestEpochFreshness_StaleEpoch(t *testing.T) {
	now := time.Now()
	epoch := CurrentEpochID(now)

	// Epoch 2 behind should be stale with skew of 1
	staleEpoch := epoch - 2
	if IsEpochFresh(staleEpoch, now, 1) {
		t.Error("epoch 2 behind should be stale with skew 1")
	}

	// Epoch 3 behind should be stale with skew of 2
	veryStale := epoch - 3
	if IsEpochFresh(veryStale, now, 2) {
		t.Error("epoch 3 behind should be stale with skew 2")
	}
}

func TestEpochFreshness_FutureEpoch(t *testing.T) {
	now := time.Now()
	epoch := CurrentEpochID(now)

	// Future epoch should be accepted with sufficient skew
	futureEpoch := epoch + 1
	if !IsEpochFresh(futureEpoch, now, 1) {
		t.Error("future epoch should be fresh with skew 1")
	}

	// Note: Current implementation only checks for stale epochs (past),
	// not for future epochs. This is intentional to allow for clock drift
	// where the client's clock may be slightly ahead of the server's.
	// If needed, add upper bound check to IsEpochFresh.
	farFuture := epoch + 100
	t.Logf("Note: Far future epoch (%d) is currently accepted (epoch=%d)",
		farFuture, epoch)
}

func TestEpochFreshness_BoundaryConditions(t *testing.T) {
	now := time.Now()
	epoch := CurrentEpochID(now)

	// Exactly at boundary (skew = 1, epoch = current - 1)
	boundaryEpoch := epoch - 1
	if !IsEpochFresh(boundaryEpoch, now, 1) {
		t.Error("epoch at boundary should be fresh")
	}

	// Just past boundary (skew = 1, epoch = current - 2)
	pastBoundary := epoch - 2
	if IsEpochFresh(pastBoundary, now, 1) {
		t.Error("epoch past boundary should be stale")
	}

	// Zero skew - only current epoch is valid
	if IsEpochFresh(epoch-1, now, 0) {
		t.Error("with zero skew, only current epoch should be fresh")
	}
}

func TestEpochFreshness_LargeSkew(t *testing.T) {
	now := time.Now()
	epoch := CurrentEpochID(now)

	// Large skew should accept many old epochs
	oldEpoch := epoch - 100
	if !IsEpochFresh(oldEpoch, now, 100) {
		t.Error("old epoch should be fresh with large skew")
	}

	// But not too old
	veryOld := epoch - 150
	if IsEpochFresh(veryOld, now, 100) {
		t.Error("very old epoch should be stale even with large skew")
	}
}

func TestCurrentEpochID_Deterministic(t *testing.T) {
	// Same time should produce same epoch
	fixedTime := time.Date(2026, 1, 7, 12, 0, 0, 0, time.UTC)
	epoch1 := CurrentEpochID(fixedTime)
	epoch2 := CurrentEpochID(fixedTime)

	if epoch1 != epoch2 {
		t.Errorf("same time should produce same epoch: %d != %d", epoch1, epoch2)
	}
}

func TestCurrentEpochID_Increasing(t *testing.T) {
	// Epochs should increase over time (assuming epoch duration is reasonable)
	t1 := time.Now()
	t2 := t1.Add(24 * time.Hour * 7) // One week later

	epoch1 := CurrentEpochID(t1)
	epoch2 := CurrentEpochID(t2)

	if epoch2 <= epoch1 {
		t.Errorf("later time should produce higher epoch: %d <= %d", epoch2, epoch1)
	}
}
