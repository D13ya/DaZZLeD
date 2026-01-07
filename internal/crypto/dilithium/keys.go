package dilithium

import (
	"crypto/rand"
	"encoding/binary"
	"io"

	"github.com/D13ya/DaZZLeD/internal/crypto/lattice"
)

// GeneratePrivateScalar produces a placeholder private scalar in R_q.
func GeneratePrivateScalar() (uint32, error) {
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
