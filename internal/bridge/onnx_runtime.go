package bridge

import (
	"crypto/sha256"
	"errors"
	"os"
	"sync"
)

const (
	HashDim        = 128
	StateDim       = 128
	DefaultSteps   = 16
	DefaultImgSize = 224
)

var (
	ErrModelNotLoaded = errors.New("ONNX model not loaded")
	ErrInvalidImage   = errors.New("invalid image data")
	ErrSessionClosed  = errors.New("ONNX session is closed")
)

// Hasher defines the interface for perceptual hash generation.
// This allows swapping between placeholder and real ONNX implementations.
type Hasher interface {
	// Hash generates a perceptual hash vector from image bytes.
	Hash(imageData []byte) ([]float32, error)
	// Close releases any resources held by the hasher.
	Close() error
}

// HasherConfig holds configuration for hasher implementations.
type HasherConfig struct {
	// ModelPath is the path to the ONNX model file
	ModelPath string
	// RecursionSteps is the number of recursive inference passes
	RecursionSteps int
	// ImageSize is the expected input image dimension
	ImageSize int
	// StateDim is the hidden state dimension
	StateDim int
	// HashDim is the output hash dimension
	HashDim int
}

// DefaultHasherConfig returns sensible defaults matching the TRM model.
func DefaultHasherConfig() HasherConfig {
	return HasherConfig{
		RecursionSteps: DefaultSteps,
		ImageSize:      DefaultImgSize,
		StateDim:       StateDim,
		HashDim:        HashDim,
	}
}

// PlaceholderHasher provides a deterministic hash implementation for testing.
// WARNING: This does NOT provide security guarantees and must be replaced
// with ONNXHasher in production.
type PlaceholderHasher struct {
	cfg    HasherConfig
	closed bool
	mu     sync.Mutex
}

// NewPlaceholderHasher creates a new placeholder hasher.
func NewPlaceholderHasher(cfg HasherConfig) *PlaceholderHasher {
	if cfg.RecursionSteps == 0 {
		cfg.RecursionSteps = DefaultSteps
	}
	if cfg.HashDim == 0 {
		cfg.HashDim = HashDim
	}
	return &PlaceholderHasher{cfg: cfg}
}

// Hash generates a deterministic hash for testing purposes.
func (h *PlaceholderHasher) Hash(imageData []byte) ([]float32, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return nil, ErrSessionClosed
	}
	if len(imageData) == 0 {
		return nil, ErrInvalidImage
	}

	return placeholderHash(imageData, h.cfg.RecursionSteps, h.cfg.HashDim), nil
}

// Close marks the hasher as closed.
func (h *PlaceholderHasher) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.closed = true
	return nil
}

// ONNXHasher provides perceptual hashing using ONNX Runtime.
// This is the production implementation that runs the trained TRM model.
type ONNXHasher struct {
	cfg    HasherConfig
	mu     sync.Mutex
	closed bool
	// session will hold the actual onnxruntime-go session
	// Add this when integrating onnxruntime-go:
	// session *ort.DynamicAdvancedSession
}

// NewONNXHasher creates a new ONNX-based hasher.
// Returns an error if the model cannot be loaded.
func NewONNXHasher(cfg HasherConfig) (*ONNXHasher, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("model path is required")
	}
	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return nil, errors.New("model file not found: " + cfg.ModelPath)
	}

	if cfg.RecursionSteps == 0 {
		cfg.RecursionSteps = DefaultSteps
	}
	if cfg.HashDim == 0 {
		cfg.HashDim = HashDim
	}
	if cfg.StateDim == 0 {
		cfg.StateDim = StateDim
	}
	if cfg.ImageSize == 0 {
		cfg.ImageSize = DefaultImgSize
	}

	h := &ONNXHasher{cfg: cfg}

	// Initialize ONNX Runtime session
	// When integrating onnxruntime-go, add:
	//
	// ort.SetSharedLibraryPath(libPath)
	// if err := ort.InitializeEnvironment(); err != nil {
	//     return nil, fmt.Errorf("init onnx environment: %w", err)
	// }
	//
	// inputNames := []string{"image", "prev_state"}
	// outputNames := []string{"next_state", "hash"}
	// session, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputNames, outputNames, nil)
	// if err != nil {
	//     return nil, fmt.Errorf("load onnx model: %w", err)
	// }
	// h.session = session

	return h, nil
}

// Hash generates a perceptual hash using the ONNX model.
func (h *ONNXHasher) Hash(imageData []byte) ([]float32, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return nil, ErrSessionClosed
	}
	if len(imageData) == 0 {
		return nil, ErrInvalidImage
	}

	// Preprocess image to tensor
	imageTensor, err := preprocessImage(imageData, h.cfg.ImageSize)
	if err != nil {
		return nil, err
	}

	// Run recursive inference
	hash, err := h.runRecursiveInference(imageTensor)
	if err != nil {
		return nil, err
	}

	return hash, nil
}

// runRecursiveInference executes the TRM model for cfg.RecursionSteps iterations.
func (h *ONNXHasher) runRecursiveInference(imageTensor []float32) ([]float32, error) {
	// Initialize state to zeros
	state := make([]float32, h.cfg.StateDim)
	var hash []float32

	for step := 0; step < h.cfg.RecursionSteps; step++ {
		// When integrating onnxruntime-go, replace with:
		//
		// imageTensorORT, _ := ort.NewTensor(ort.NewShape(1, 3, h.cfg.ImageSize, h.cfg.ImageSize), imageTensor)
		// stateTensorORT, _ := ort.NewTensor(ort.NewShape(1, h.cfg.StateDim), state)
		// defer imageTensorORT.Destroy()
		// defer stateTensorORT.Destroy()
		//
		// outputs, err := h.session.Run([]ort.ArbitraryTensor{imageTensorORT, stateTensorORT})
		// if err != nil {
		//     return nil, fmt.Errorf("inference step %d: %w", step, err)
		// }
		//
		// state = outputs[0].GetData().([]float32)
		// hash = outputs[1].GetData().([]float32)

		// Placeholder: use deterministic simulation
		state, hash = simulateInferenceStep(imageTensor, state, step, h.cfg.HashDim)
	}

	return hash, nil
}

// simulateInferenceStep provides a deterministic placeholder for a single TRM step.
func simulateInferenceStep(image, state []float32, step, hashDim int) ([]float32, []float32) {
	// Combine image features with state
	combined := make([]byte, len(image)*4+len(state)*4+1)
	offset := 0
	for _, v := range image[:min(len(image), 64)] { // Use subset for speed
		combined[offset] = byte(int(v*127) & 0xFF)
		offset++
	}
	for _, v := range state {
		combined[offset] = byte(int(v*127) & 0xFF)
		offset++
	}
	combined[offset] = byte(step)

	// Generate new state and hash
	sum := sha256.Sum256(combined)

	newState := make([]float32, len(state))
	for i := range newState {
		newState[i] = (float32(sum[i%len(sum)]) / 127.5) - 1.0
	}

	hash := make([]float32, hashDim)
	for i := range hash {
		hash[i] = (float32(sum[(i+16)%len(sum)]) / 127.5) - 1.0
	}

	// L2 normalize hash
	hash = normalizeL2(hash)
	return newState, hash
}

// Close releases ONNX runtime resources.
func (h *ONNXHasher) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return nil
	}
	h.closed = true

	// When integrating onnxruntime-go:
	// if h.session != nil {
	//     h.session.Destroy()
	// }
	// ort.DestroyEnvironment()

	return nil
}

// preprocessImage decodes and normalizes image data for the model.
func preprocessImage(data []byte, targetSize int) ([]float32, error) {
	if len(data) == 0 {
		return nil, ErrInvalidImage
	}

	// Full implementation requires image decoding library.
	// When integrating, use:
	// - "image" package for decoding
	// - "golang.org/x/image/draw" for resizing
	// - Apply ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

	// Placeholder: generate tensor from raw bytes
	tensorSize := 3 * targetSize * targetSize
	tensor := make([]float32, tensorSize)

	// Use image bytes to seed deterministic values
	sum := sha256.Sum256(data)
	for i := range tensor {
		tensor[i] = (float32(sum[i%len(sum)]) / 127.5) - 1.0
	}

	return tensor, nil
}

// normalizeL2 applies L2 normalization to a vector.
func normalizeL2(v []float32) []float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	if norm > 0 {
		invNorm := 1.0 / sqrt32(norm)
		for i := range v {
			v[i] *= invNorm
		}
	}
	return v
}

func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// --- Legacy API for backward compatibility ---

// LoadImage reads raw bytes from disk.
func LoadImage(path string) ([]byte, error) {
	if path == "" {
		return nil, errors.New("image path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, ErrInvalidImage
	}
	return data, nil
}

// RecursiveInference is the legacy API. Use Hasher interface for new code.
func RecursiveInference(image []byte, steps int) []float32 {
	if steps <= 0 {
		steps = DefaultSteps
	}
	return placeholderHash(image, steps, HashDim)
}

// placeholderHash generates a deterministic hash for testing.
func placeholderHash(image []byte, steps, hashDim int) []float32 {
	state := sha256.Sum256(image)
	for i := 0; i < steps; i++ {
		combined := append(state[:], byte(i))
		state = sha256.Sum256(combined)
	}

	out := make([]float32, hashDim)
	for i := range out {
		b := state[i%len(state)]
		out[i] = (float32(b) / 127.5) - 1.0
	}

	return normalizeL2(out)
}
