package lattice

import (
	"encoding/binary"
	"errors"
)

type Vector struct {
	Coeffs []uint32
}

func NewVector(size int) Vector {
	return Vector{Coeffs: make([]uint32, size)}
}

func (v Vector) Marshal() []byte {
	out := make([]byte, len(v.Coeffs)*4)
	for i, c := range v.Coeffs {
		binary.BigEndian.PutUint32(out[i*4:], c)
	}
	return out
}

func Deserialize(in []byte) (Vector, error) {
	if len(in)%4 != 0 {
		return Vector{}, errors.New("invalid lattice encoding")
	}
	count := len(in) / 4
	coeffs := make([]uint32, count)
	for i := 0; i < count; i++ {
		coeffs[i] = binary.BigEndian.Uint32(in[i*4 : (i+1)*4])
		if coeffs[i] >= ModulusQ {
			return Vector{}, errors.New("coefficient out of range")
		}
	}
	return Vector{Coeffs: coeffs}, nil
}

func MulScalar(v Vector, scalar uint32) Vector {
	out := make([]uint32, len(v.Coeffs))
	for i, c := range v.Coeffs {
		out[i] = uint32((uint64(c) * uint64(scalar)) % uint64(ModulusQ))
	}
	return Vector{Coeffs: out}
}
