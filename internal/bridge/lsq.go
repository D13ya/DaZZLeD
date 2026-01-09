package bridge

import (
	"math"

	"github.com/D13ya/DaZZLeD/internal/crypto/lattice"
)

// MapToLattice quantizes the hash vector into k polynomials over R_q.
// Expects sigmoid outputs in [0,1] and maps them to [-1,1] before quantization.
func MapToLattice(hash []float32) lattice.Vector {
	coeffs := make([]uint32, lattice.RingN*lattice.ModuleK)
	if len(hash) == 0 {
		return lattice.Vector{Coeffs: coeffs}
	}
	for i := range coeffs {
		scaled := scaleSigmoidToSigned(hash[i%len(hash)])
		coeffs[i] = quantize(scaled)
	}
	return lattice.Vector{Coeffs: coeffs}
}

// BinarizeHash thresholds sigmoid outputs into {0,1}.
// If the input appears signed (outside [0,1]), thresholds at 0 instead.
func BinarizeHash(hash []float32) []float32 {
	out := make([]float32, len(hash))
	if len(hash) == 0 {
		return out
	}
	minVal := hash[0]
	maxVal := hash[0]
	for _, v := range hash[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	threshold := float32(0.5)
	if minVal < 0 || maxVal > 1 {
		threshold = 0
	}
	for i, v := range hash {
		if v > threshold {
			out[i] = 1
		}
	}
	return out
}

func scaleSigmoidToSigned(x float32) float32 {
	if x < 0 || x > 1 {
		if x < -1 {
			return -1
		}
		if x > 1 {
			return 1
		}
		return x
	}
	return (2 * x) - 1
}

func quantize(x float32) uint32 {
	if x > 1 {
		x = 1
	}
	if x < -1 {
		x = -1
	}
	scaled := float64(x) * float64(lattice.ModulusQ) / 2.0
	rounded := int64(math.Round(scaled))
	if rounded < 0 {
		rounded += int64(lattice.ModulusQ)
	}
	return uint32(uint64(rounded) % uint64(lattice.ModulusQ))
}
