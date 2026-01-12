package main

import (
	"fmt"
	"os"

	"github.com/D13ya/DaZZLeD/internal/bridge"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: hashtest <image_path>")
		fmt.Println("Example: hashtest test.jpg")
		os.Exit(1)
	}

	imagePath := os.Args[1]

	// Initialize ONNX Runtime with the DLL
	if err := bridge.InitONNXEnvironment("configs/models/onnxruntime.dll"); err != nil {
		fmt.Printf("Error initializing ONNX: %v\n", err)
		os.Exit(1)
	}
	defer bridge.DestroyONNXEnvironment()

	// Create hasher with real ONNX inference
	cfg := bridge.HasherConfig{
		ModelPath:      "configs/models/hashnet.onnx",
		ImageSize:      224,
		HashDim:        128,
		UsePlaceholder: false, // Use real ONNX!
	}

	hasher, err := bridge.NewONNXHasher(cfg)
	if err != nil {
		fmt.Printf("Error creating hasher: %v\n", err)
		os.Exit(1)
	}
	defer hasher.Close()

	// Load and hash the image
	imgBytes, err := bridge.LoadImage(imagePath)
	if err != nil {
		fmt.Printf("Error loading image: %v\n", err)
		os.Exit(1)
	}

	hash, err := hasher.Hash(imgBytes)
	if err != nil {
		fmt.Printf("Error hashing: %v\n", err)
		os.Exit(1)
	}

	// Binarize the hash
	binaryHash := bridge.BinarizeHashToBytes(hash)

	fmt.Printf("Image: %s\n", imagePath)
	fmt.Printf("Hash (first 10 floats): %.4f\n", hash[:10])
	fmt.Printf("Binary hash (hex): %x\n", binaryHash)
	fmt.Printf("Binary hash (bits): ")
	for _, b := range binaryHash {
		fmt.Printf("%08b", b)
	}
	fmt.Println()
}
