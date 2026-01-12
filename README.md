# **DaZZLeD: Privacy-Preserving Content Detection**

### *A Clean-Room Implementation of Apple's PSI Protocol with Post-Quantum Improvements*

[![Go](https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go)](https://golang.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx)](https://onnxruntime.ai)

---

## **🎯 Overview**

DaZZLeD is a research project that **reconstructs and improves** upon the [Apple CSAM Detection Protocol](https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf).

It implements the core "Sandwich" privacy architecture:
1.  **Client-Side AI:** Perceptual hashing to detect content.
2.  **Blind PSI:** Checking hashes against a server without revealing user data.

### **Our Improvements**
We address specific weaknesses in the original design using modern techniques:

| Feature | Apple NeuralHash (Original) | DaZZLeD (This Project) |
|---------|-----------------------------|------------------------|
| **Hash Robustness** | Vulnerable to collision attacks | **ResNetHashNet:** Contrastive learning + adversarial training for robust per-image hashing. |
| **Cryptography** | Elliptic Curve PSI (Pre-Quantum) | **Lattice PSI:** Post-Quantum ML-DSA + OPRF (Module-Lattices). |
| **Auditability** | Opaque Database | **Signed Commitment:** The server signs the database state (Bloom Filter) to prevent split-view attacks. |
| **Runtime** | iOS CoreML only | Cross-platform ONNX Runtime (Windows/Linux/Mac). |

> **⚠️ Note on Database:** This is a **clean-room implementation**. We do NOT possess or distribute real CSAM hashes. The system is designed to verify the *protocol*, and users must ingest their own dummy hashes for testing.

---

## **🏗 Architecture**

```mermaid
graph TD
    subgraph Client [Client Device]
        Img["Image (JPEG)"] -->|HashNet ONNX| Hash["128-bit Perceptual Hash"]
        Hash -->|LSQ Quantization| Lattice["Lattice Ring Element R_q"]
        
        subgraph Crypto Core
            Lattice -->|Blind w/ Randomness| Blinded["Blinded Element P'"]
        end
    end

    Blinded -->|gRPC: BlindCheckRequest| Server
    
    subgraph Server [Authority Node]
        Server -->|Sign Blindly| Signature["Blinded Signature S'"]
        Server -->|Sign Bloom Filter| Proof["Signed Set Commitment"]
    end
    
    Signature -->|gRPC: BlindCheckResponse| ValidDB
    Proof -->|gRPC: BlindCheckResponse| ValidDB
    
    subgraph VerifyGroup [Verification]
        ValidDB{"Valid Commitment?"}
        ValidDB -- Yes --> Unblind
        Unblind -->|Unblind Signature| FinalSig
        FinalSig -->|Check Membership| Result{"Match Found?"}
    end
    
    style Client fill:#f9f,stroke:#333
    style Server fill:#bbf,stroke:#333
    style VerifyGroup fill:#efe,stroke:#333
```

---

## **🚀 Quick Start**

### Prerequisites

- **Go 1.24+**
- **Python 3.10+** (for training)
- **ONNX Runtime** (for inference)

### 1. Clone & Build

```bash
git clone https://github.com/D13ya/DaZZLeD.git
cd DaZZLeD
go mod tidy
go build ./...
```

### 2. Test the Hash Generator

```bash
# Build the hash test tool
go build -o hashtest.exe ./cmd/hashtest

# Hash an image
./hashtest.exe path/to/image.jpg
```

**Output:**
```
Image: test.jpg
Hash (first 10 floats): [0.1234 0.8765 0.3456 ...]
Binary hash (hex): a1b2c3d4e5f6789012345678...
Binary hash (bits): 10100001101100101100...
```

---

## **🧠 ML Core: HashNet Training**

The perceptual hasher is a **ResNet50-based contrastive model** trained to produce unique 128-bit hashes for each image.

### Training Features

| Component | Description |
|-----------|-------------|
| **Backbone** | ResNet50 (ImageNet pretrained) |
| **Hash Dim** | 128 bits |
| **Losses** | NT-Xent contrastive + DHD + Quantization |
| **Augmentations** | Random crop, flip, color jitter, blur |
| **Memory Optimization** | Gradient checkpointing (~50% savings) |
| **Training Time** | ~2 hours on T4 GPU (55k images) |

### Train Your Own Model

```python
# Google Colab (T4 GPU recommended)
!python training/train_hashnet.py \
  --data-list /path/to/manifest.txt \
  --backbone resnet50 \
  --epochs 10 \
  --batch-size 256 \
  --grad-checkpoint \
  --label-mode none \
  --hash-contrastive-weight 1.0 \
  --dhd-weight 0.5 \
  --quant-weight 0.1 \
  --counterfactual-mode aug \
  --lr 5e-4 \
  --amp
```

### Export to ONNX

```python
import torch
from training.train_hashnet import ResNetHashNet
import safetensors.torch

model = ResNetHashNet("resnet50", hash_dim=128, proj_dim=512, pretrained=False)
safetensors.torch.load_model(model, "student_final.safetensors")
model.eval()

torch.onnx.export(
    model,
    torch.randn(1, 3, 224, 224),
    "hashnet.onnx",
    input_names=["image"],
    output_names=["hash"],
    dynamic_axes={"image": {0: "batch"}, "hash": {0: "batch"}},
    opset_version=14
)
```

---

## **⚙️ Go Integration**

### ONNX Runtime Setup

1. **Download ONNX Runtime** from [GitHub Releases](https://github.com/microsoft/onnxruntime/releases)
2. **Place files** in `configs/models/`:
   - `hashnet.onnx` (your trained model)
   - `hashnet.onnx.data` (model weights)
   - `onnxruntime.dll` (runtime library)

### Using the Hasher API

```go
package main

import (
    "fmt"
    "github.com/D13ya/DaZZLeD/internal/bridge"
)

func main() {
    // Initialize ONNX Runtime
    bridge.InitONNXEnvironment("configs/models/onnxruntime.dll")
    defer bridge.DestroyONNXEnvironment()

    // Create hasher
    cfg := bridge.HasherConfig{
        ModelPath: "configs/models/hashnet.onnx",
        ImageSize: 224,
        HashDim:   128,
    }
    hasher, _ := bridge.NewONNXHasher(cfg)
    defer hasher.Close()

    // Hash an image
    imgBytes, _ := bridge.LoadImage("photo.jpg")
    hash, _ := hasher.Hash(imgBytes)

    // Binarize for comparison
    binaryHash := bridge.BinarizeHashToBytes(hash)
    fmt.Printf("Hash: %x\n", binaryHash)

    // Compare two images
    hash2, _ := hasher.Hash(otherImageBytes)
    distance := bridge.HammingDistance(
        bridge.BinarizeHashToBytes(hash),
        bridge.BinarizeHashToBytes(hash2),
    )
    fmt.Printf("Hamming distance: %d bits\n", distance)
}
```

---

## **🔐 Crypto Core: Post-Quantum Security**

### Lattice-Based OPRF

The hash is mapped to a lattice ring element before cryptographic operations:

```go
// Map float hash to lattice point
latticePoint := bridge.MapToLattice(hashVec)

// Blind for OPRF
state, blindedRequest := oprfClient.Blind(latticePoint.Marshal())

// Server signs blindly (doesn't see the hash)
// Client unblinds to verify membership
```

### ML-DSA Signatures

All proofs are signed with [ML-DSA (Dilithium)](https://pq-crystals.org/dilithium/), a post-quantum digital signature algorithm.

---

## **📊 Performance**

| Metric | Value |
|--------|-------|
| Hash generation | ~15ms (GPU) / ~100ms (CPU) |
| Model size (ONNX) | ~95 MB |
| Hash size | 128 bits (16 bytes) |
| Collision resistance | 2^64 (birthday bound) |

---

## **📁 Project Structure**

```
DaZZLeD/
├── cmd/
│   ├── client/          # Client binary
│   ├── server/          # Server binary
│   ├── hashtest/        # Hash testing tool
│   └── setup/           # Key generation
├── configs/
│   └── models/          # ONNX model + runtime
├── internal/
│   ├── bridge/          # ONNX Runtime wrapper
│   │   ├── onnx_runtime.go
│   │   └── lsq.go       # Lattice quantization
│   ├── crypto/          # Post-quantum crypto
│   └── app/             # Client/server logic
├── ml-core/
│   ├── training/
│   │   └── train_hashnet.py  # HashNet training
│   └── notebooks/       # Colab notebooks
└── api/
    └── proto/           # gRPC definitions
```

---

## **🔬 Research Background**

This project implements concepts from:

1. **[Black-box Collision Attacks on NeuralHash](https://eprint.iacr.org/2024/1869.pdf)** - Why we need adversarial robustness
2. **[Split Accumulation for Relations](https://eprint.iacr.org/2020/1618.pdf)** - Our ZK verification approach
3. **[Contrastive Learning for Perceptual Hashes](https://arxiv.org/abs/2002.05709)** - NT-Xent loss for per-image discrimination

---

## **⚠️ Legal & Ethical Notice**

**Research Only:** This is an educational implementation for studying privacy-preserving AI systems.

- No real illegal content is used for training or testing
- All datasets are public and non-sensitive (FFHQ, OpenImages)
- This is a clean-room implementation based on public papers

---

## **📜 License**

MIT License - See [LICENSE](LICENSE) for details.

---

## **🤝 Contributing**

Contributions welcome! Please see:
1. Open an issue to discuss changes
2. Fork and create a PR
3. Ensure tests pass: `go test ./...`
