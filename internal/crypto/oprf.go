package crypto

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"io"

	"github.com/D13ya/DaZZLeD/internal/crypto/lattice"
)

type Signer struct {
	PrivateKey uint32
}

// Blind multiplies the lattice vector by a scalar in R_q.
func Blind(v lattice.Vector, scalar uint32) lattice.Vector {
	return lattice.MulScalar(v, scalar)
}

// SignBlinded applies the server's private scalar to the blinded point.
func (s Signer) SignBlinded(v lattice.Vector) []byte {
	return lattice.MulScalar(v, s.PrivateKey).Marshal()
}

// Unblind removes the scalar from a blinded signature. Placeholder logic only.
func Unblind(signature []byte, scalar uint32) ([]byte, error) {
	v, err := lattice.Deserialize(signature)
	if err != nil {
		return nil, err
	}
	inv, err := modInverse(uint64(scalar), uint64(lattice.ModulusQ))
	if err != nil {
		return nil, err
	}
	return lattice.MulScalar(v, uint32(inv)).Marshal(), nil
}

// NewRandomScalar returns a non-zero scalar in R_q.
func NewRandomScalar() (uint32, error) {
	for {
		var buf [4]byte
		if _, err := io.ReadFull(rand.Reader, buf[:]); err != nil {
			return 0, err
		}
		val := binary.BigEndian.Uint32(buf[:]) % lattice.ModulusQ
		if val != 0 {
			return val, nil
		}
	}
}

func modInverse(a, m uint64) (uint64, error) {
	var t, newT int64 = 0, 1
	var r, newR int64 = int64(m), int64(a)
	for newR != 0 {
		quotient := r / newR
		t, newT = newT, t-quotient*newT
		r, newR = newR, r-quotient*newR
	}
	if r > 1 {
		return 0, errors.New("not invertible")
	}
	if t < 0 {
		t += int64(m)
	}
	return uint64(t), nil
}
