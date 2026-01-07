package bridge

import (
	"math"

	"github.com/D13ya/DaZZLeD/internal/crypto/lattice"
)

// MapToLattice quantizes the hash vector into k polynomials over R_q.
func MapToLattice(hash []float32) lattice.Vector {
	coeffs := make([]uint32, lattice.RingN*lattice.ModuleK)
	if len(hash) == 0 {
		return lattice.Vector{Coeffs: coeffs}
	}
	for i := range coeffs {
		coeffs[i] = quantize(hash[i%len(hash)])
	}
	return lattice.Vector{Coeffs: coeffs}
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
