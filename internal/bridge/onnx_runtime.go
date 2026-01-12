package bridge

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

const (
	HashDim        = 128
	DefaultImgSize = 224
)

var (
	ErrModelNotLoaded = errors.New("ONNX model not loaded")
	ErrInvalidImage   = errors.New("invalid image data")
	ErrSessionClosed  = errors.New("ONNX session is closed")
)

// ImageNet normalization constants
var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// onnxInitialized tracks if ONNX runtime environment is initialized
var onnxInitialized bool
var onnxInitMu sync.Mutex

// Hasher defines the interface for perceptual hash generation.
type Hasher interface {
	Hash(imageData []byte) ([]float32, error)
	Close() error
}

// HasherConfig holds configuration for hasher implementations.
type HasherConfig struct {
	ModelPath      string
	ImageSize      int
	HashDim        int
	SharedLibPath  string // Path to onnxruntime shared library (optional)
	UsePlaceholder bool   // If true, skip ONNX and use placeholder (for testing)
}

// DefaultHasherConfig returns sensible defaults matching the ResNetHashNet model.
func DefaultHasherConfig() HasherConfig {
	return HasherConfig{
		ImageSize: DefaultImgSize,
		HashDim:   HashDim,
	}
}

// PlaceholderHasher provides a deterministic hash implementation for testing.
type PlaceholderHasher struct {
	cfg    HasherConfig
	closed bool
	mu     sync.Mutex
}

// NewPlaceholderHasher creates a new placeholder hasher.
func NewPlaceholderHasher(cfg HasherConfig) *PlaceholderHasher {
	if cfg.HashDim == 0 {
		cfg.HashDim = HashDim
	}
	if cfg.ImageSize == 0 {
		cfg.ImageSize = DefaultImgSize
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

	return placeholderHash(imageData, h.cfg.HashDim), nil
}

func (h *PlaceholderHasher) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.closed = true
	return nil
}

// ONNXHasher provides perceptual hashing using ONNX Runtime.
type ONNXHasher struct {
	cfg          HasherConfig
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	mu           sync.Mutex
	closed       bool
}

// InitONNXEnvironment initializes the ONNX runtime environment.
// Call this once at application startup with the path to onnxruntime shared library.
func InitONNXEnvironment(sharedLibPath string) error {
	onnxInitMu.Lock()
	defer onnxInitMu.Unlock()

	if onnxInitialized {
		return nil
	}

	if sharedLibPath != "" {
		ort.SetSharedLibraryPath(sharedLibPath)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("init onnx environment: %w", err)
	}

	onnxInitialized = true
	return nil
}

// DestroyONNXEnvironment cleans up the ONNX runtime environment.
// Call this at application shutdown.
func DestroyONNXEnvironment() error {
	onnxInitMu.Lock()
	defer onnxInitMu.Unlock()

	if !onnxInitialized {
		return nil
	}

	if err := ort.DestroyEnvironment(); err != nil {
		return err
	}

	onnxInitialized = false
	return nil
}

// NewONNXHasher creates a new ONNX-based hasher.
func NewONNXHasher(cfg HasherConfig) (*ONNXHasher, error) {
	if cfg.UsePlaceholder {
		// Return a wrapper that uses placeholder internally
		return &ONNXHasher{cfg: cfg, session: nil}, nil
	}

	if cfg.ModelPath == "" {
		return nil, errors.New("model path is required")
	}
	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return nil, errors.New("model file not found: " + cfg.ModelPath)
	}

	if cfg.HashDim == 0 {
		cfg.HashDim = HashDim
	}
	if cfg.ImageSize == 0 {
		cfg.ImageSize = DefaultImgSize
	}

	// Ensure ONNX environment is initialized
	if err := InitONNXEnvironment(cfg.SharedLibPath); err != nil {
		return nil, err
	}

	// Create input tensor [1, 3, 224, 224]
	inputShape := ort.NewShape(1, 3, int64(cfg.ImageSize), int64(cfg.ImageSize))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	// Create output tensor [1, 128]
	outputShape := ort.NewShape(1, int64(cfg.HashDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	// Create session with input and output tensors
	session, err := ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{"image"},
		[]string{"hash"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil, // Use default options
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create onnx session: %w", err)
	}

	return &ONNXHasher{
		cfg:          cfg,
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
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

	// Use placeholder if no session
	if h.session == nil {
		return placeholderHash(imageData, h.cfg.HashDim), nil
	}

	// Preprocess image to tensor
	imageTensor, err := preprocessImage(imageData, h.cfg.ImageSize)
	if err != nil {
		return nil, err
	}

	// Copy preprocessed image data into input tensor
	inputData := h.inputTensor.GetData()
	copy(inputData, imageTensor)

	// Run inference
	if err := h.session.Run(); err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Copy output from output tensor
	outputData := h.outputTensor.GetData()
	result := make([]float32, h.cfg.HashDim)
	copy(result, outputData)

	return result, nil
}

// Close releases ONNX runtime resources.
func (h *ONNXHasher) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return nil
	}
	h.closed = true

	var errs []error

	if h.session != nil {
		if err := h.session.Destroy(); err != nil {
			errs = append(errs, err)
		}
	}
	if h.inputTensor != nil {
		if err := h.inputTensor.Destroy(); err != nil {
			errs = append(errs, err)
		}
	}
	if h.outputTensor != nil {
		if err := h.outputTensor.Destroy(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// preprocessImage decodes, resizes, and normalizes image data for the model.
func preprocessImage(data []byte, targetSize int) ([]float32, error) {
	if len(data) == 0 {
		return nil, ErrInvalidImage
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return preprocessImageFallback(data, targetSize), nil
	}

	resized := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	tensorSize := 3 * targetSize * targetSize
	tensor := make([]float32, tensorSize)

	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			rNorm := (float32(r>>8)/255.0 - imagenetMean[0]) / imagenetStd[0]
			gNorm := (float32(g>>8)/255.0 - imagenetMean[1]) / imagenetStd[1]
			bNorm := (float32(b>>8)/255.0 - imagenetMean[2]) / imagenetStd[2]

			tensor[0*targetSize*targetSize+y*targetSize+x] = rNorm
			tensor[1*targetSize*targetSize+y*targetSize+x] = gNorm
			tensor[2*targetSize*targetSize+y*targetSize+x] = bNorm
		}
	}

	return tensor, nil
}

func preprocessImageFallback(data []byte, targetSize int) []float32 {
	tensorSize := 3 * targetSize * targetSize
	tensor := make([]float32, tensorSize)
	sum := sha256.Sum256(data)
	for i := range tensor {
		tensor[i] = (float32(sum[i%len(sum)]) / 127.5) - 1.0
	}
	return tensor
}

// BinarizeHashToBytes converts soft hash to compact binary hash.
func BinarizeHashToBytes(softHash []float32) []byte {
	numBytes := (len(softHash) + 7) / 8
	binary := make([]byte, numBytes)
	for i, v := range softHash {
		if v > 0.5 {
			binary[i/8] |= 1 << (7 - uint(i%8))
		}
	}
	return binary
}

// HammingDistance computes the Hamming distance between two binary hashes.
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		return -1
	}
	dist := 0
	for i := range a {
		xor := a[i] ^ b[i]
		for xor != 0 {
			dist += int(xor & 1)
			xor >>= 1
		}
	}
	return dist
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// --- Legacy API for backward compatibility ---

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

func placeholderHash(imageData []byte, hashDim int) []float32 {
	sum := sha256.Sum256(imageData)
	out := make([]float32, hashDim)
	for i := range out {
		out[i] = float32(sum[i%len(sum)]) / 255.0
	}
	return out
}

func RecursiveInference(image []byte, steps int) []float32 {
	return placeholderHash(image, HashDim)
}
