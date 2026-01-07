package bridge

import (
	"crypto/sha256"
	"errors"
	"os"
)

const HashDim = 96

// LoadImage reads raw bytes from disk. Real implementations should decode and preprocess.
func LoadImage(path string) ([]byte, error) {
	if path == "" {
		return nil, errors.New("image path is required")
	}
	return os.ReadFile(path)
}

// RecursiveInference is a deterministic placeholder. Replace with ONNX runtime inference.
func RecursiveInference(image []byte, _ int) []float32 {
	sum := sha256.Sum256(image)
	out := make([]float32, HashDim)
	for i := range out {
		b := sum[i%len(sum)]
		out[i] = (float32(b) / 127.5) - 1.0
	}
	return out
}
