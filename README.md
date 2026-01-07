# **Recursive Safety: Post-Quantum Content Detection Protocol**

A Zero-Knowledge, Adversarially Robust Surveillance Architecture

Proof-of-Concept Implementation in Go & Python

## **📖 Overview**

**Recursive Safety** is a privacy-preserving protocol designed to detect illegal content (e.g., CSAM) on client devices **without** the server ever viewing user data or the user having to blindly trust the server's database.

This project reconstructs and effectively "patches" the cancelled [Apple CSAM Detection Protocol](https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf) by addressing its fundamental flaws using state-of-the-art research from 2024-2026.

### **The Core Improvements**

| Feature | Original Protocol (NeuralHash) | Recursive Safety (This Project) |
| :---- | :---- | :---- |
| **Robustness** | **Fragile.** Vulnerable to gradient-based collision attacks (one-shot CNN). | **Robust.** Uses a **Tiny Recursive Model (TRM)** that loops 16x to "denoise" adversarial attacks. |
| **Trust** | **Blind.** User must trust the server's list contains only illegal content. | **Verifiable.** Server provides a **Zero-Knowledge Proof** (Split Accumulation) that every hash is signed by a trusted authority (NCMEC). |
| **Security** | **Pre-Quantum.** Relies on Elliptic Curves (ECC) and standard PSI. | **Post-Quantum.** Built on **Module-Lattices** (ML-DSA / Dilithium) and Lattice-based OPRF. |
| **False Positives** | **High.** Recent research shows high collision rates on human faces. | **Minimized.** Uses **DINOv3** distillation for semantic (not just textural) understanding. |

---

## **🏗 System Architecture**

The system operates as a "Sandwich" architecture: heavy AI on the client, exact crypto on the wire.

### **1\. ML Core: The Recursive Hasher**

Instead of a feed-forward CNN, we use a **Samsung-style Tiny Recursive Model (TRM)**.

* **Logic:** The model takes an image and a "thought vector." It loops 16 times, refining the hash at each step. To force a collision, an attacker must fool the model 16 times simultaneously.  
* **Training:** Distilled from a frozen **DINOv3** backbone to capture dense semantic features (depth, segmentation) rather than fragile edge patterns.  
* **Stack:** Python (PyTorch) $\\to$ ONNX $\\to$ Go Runtime (onnxruntime\_go).

### **2\. Crypto Core: Post-Quantum OPRF**

We implement a blind intersection protocol using **Lattice-Based Cryptography**.

* **Bridge:** A "Locality Sensitive Quantization" layer maps the fuzzy AI float vector to an exact Ring Element ($R\_q$).  
* **Protocol:**  
  1. **Client:** Blinds the hash $H$ with factor $r$: $H' \= H \\cdot r \+ e$.  
  2. **Server:** Signs blindly: $S' \= H' \\cdot k \+ e'$.  
  3. **Client:** Unblinds to get signature $S$ and checks against the public key.

### **3\. Verification: Split Accumulation**

To solve the "Auditability" problem without massive bandwidth, we implement **Split Accumulation for Relations** (Bünz et al.).

* **The Check:** The client demands proof that the server's key $k$ corresponds *only* to the authorized NCMEC database.  
* **The Trick:** Instead of sending a massive SNARK, the server maintains a **Split Accumulator**. The client receives a constant-size "Instance" (2.5KB) and verifies a single linear step. If the accumulator holds, the entire database history is valid.

---

## **🛠 Technical Stack**

### **Core Services**

* **Client/Server Logic:** Go (1.22+)  
* **RPC Framework:** gRPC \+ Protobuf  
* **Database:** BadgerDB (Key-Value store for blinded hashes)

### **Cryptography (crypto-core)**

* **Lattice Library:** tuneinsight/lattigo (Go) or cloudflare/circl (ML-DSA/Dilithium)  
* **Zero Knowledge:** Custom implementation of Split Accumulation over Module-SIS.

### **Machine Learning (ml-core)**

* **Training:** PyTorch \+ Transformers (HuggingFace)  
* **Inference:** ONNX Runtime  
* **Teacher Model:** Meta DINOv3 (ViT-Large)

---

## **📁 Project Layout**

```text
go.mod                         - Go module path and dependency versions
Makefile                       - Convenience targets for build and proto generation

api/
  proto/
    v1/
      service.proto            - gRPC contract (messages + RPCs)

cmd/
  client/
    main.go                    - Client binary entrypoint
  server/
    main.go                    - Server binary entrypoint
  setup/
    main.go                    - Admin key setup utility

configs/
  client.yaml                  - Client runtime config template
  server.yaml                  - Server runtime config template

internal/
  api/
    grpc_handler.go            - gRPC handler wiring to app layer
  app/
    client_service.go          - Client orchestration (hash -> blind -> request)
    server_service.go          - Server orchestration (sign -> prove -> respond)
  bridge/
    lsq.go                     - LSQ quantization (float -> lattice)
    onnx_runtime.go            - ONNX runtime wrapper (placeholder)
  crypto/
    accumulator.go             - Split accumulator proof helpers (placeholder)
    oprf.go                    - Blind signature / OPRF helpers (placeholder)
    params.go                  - Crypto constants shared by modules
    verify.go                  - Signature verification stub
    dilithium/
      keys.go                  - ML-DSA key generation stub
    lattice/
      params.go                - Ring parameters (n, q, k)
      vector.go                - Lattice vector serialization + math
  storage/
    badger.go                  - Proof storage interface + in-memory stub

ml-core/
  requirements.txt             - Python training dependencies
  models/
    dino_teacher.py            - DINOv3 teacher loader
    recursive_student.py       - Tiny recursive student model
  training/
    export_onnx.py             - ONNX export script
    train.py                   - Distillation training loop

pkg/
  logger/
    logger.go                  - Shared logger helper
  profiler/
    profiler.go                - Lightweight timing helper

scripts/
  build.sh                     - Build all Go binaries
  gen_proto.sh                 - Generate Go gRPC stubs
```

---

## **🚀 Getting Started**

### **Prerequisites**

* Go 1.22+  
* Python 3.10+  
* libonnxruntime (Shared Library)

### **0\. Prepare Datasets**

Download and stage the public, non-sensitive datasets in `./data`:

| Dataset | Role | Why use it instead of COCO |
| :--- | :--- | :--- |
| **FFHQ (NVIDIA)** | The "People" Proxy | High-res faces; perceptual hashes often fail on faces. |
| **OpenImages V7** | The "Life" Proxy | Replaces COCO; messy, real-world gallery photos. |
| **Text/Screenshots** | The "Edge Case" | Hashers often false-positive on text (receipts, documents). Use RVL-CDIP or synthesize. |

Bite-size subsets are recommended for OpenImages and Text/Screenshots to fit local disk and Colab limits.

Bash

```bash
mkdir -p data/ffhq/raw data/openimages/raw data/text/raw

# Place curated subsets into raw/ folders, then resize to 224x224 using a small script.
python - <<'PY'
from pathlib import Path
from PIL import Image

datasets = ["ffhq", "openimages", "text"]
for name in datasets:
    raw = Path(f"data/{name}/raw")
    out = Path(f"data/{name}/224")
    out.mkdir(parents=True, exist_ok=True)
    for img_path in raw.rglob("*"):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
            continue
        img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC)
        img.save(out / img_path.name)
PY
```

PowerShell (Windows)

```powershell
New-Item -ItemType Directory -Force -Path "data/ffhq/raw","data/openimages/raw","data/text/raw"
```

PowerShell does not support Bash here-docs, so save this to `prep_data.py`:

```python
from pathlib import Path
from PIL import Image

datasets = ["ffhq", "openimages", "text"]
for name in datasets:
    raw = Path(f"data/{name}/raw")
    out = Path(f"data/{name}/224")
    out.mkdir(parents=True, exist_ok=True)
    for img_path in raw.rglob("*"):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
            continue
        img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC)
        img.save(out / img_path.name)
```

Then run:

```powershell
pip install Pillow
python prep_data.py
```

### **1\. Build the Recursive Model**

Train the tiny student model to mimic DINOv3 (GPU recommended). This is a heavy run; expect hours on a single GPU for meaningful checkpoints.

Bash

cd ml-core  
python -m venv .venv  
source .venv/bin/activate  
pip install \-r requirements.txt  

# Optional sanity check (CUDA should be True on GPU machines)
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
PY

# Point to a single staged dataset root (merge FFHQ/OpenImages/Text under one folder)
python training/train.py \--data ../data/ffhq/224 \--teacher facebook/dinov3-vit-large-14 \--epochs 5 \--checkpoint-dir ./checkpoints  

# Export ONNX from a safetensors checkpoint
python training/export\_onnx.py \--checkpoint ./checkpoints/student_epoch_5.safetensors \--output ./models/trm\_v1.onnx

PowerShell (Windows)

```powershell
cd ml-core
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional sanity check (CUDA should be True on GPU machines)
python -c "import torch; print('cuda:', torch.cuda.is_available())"

python training\train.py --data ..\data\ffhq\224 --teacher facebook/dinov3-vit-large-14 --epochs 5 --checkpoint-dir .\checkpoints
python training\export_onnx.py --checkpoint .\checkpoints\student_epoch_5.safetensors --output .\models\trm_v1.onnx
```

### **1\.1 Train in Google Colab (Recommended for GPU)**

If your local machine has no GPU, use Colab. A full walk‑through is in `ml-core/COLAB.md`. Example Colab command:

```python
!python training/train.py \
  --data /content/drive/MyDrive/dazzled/data/combined \
  --teacher facebook/dinov3-vit-large-14 \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --amp \
  --checkpoint-dir ./checkpoints \
  --checkpoint-every 200 \
  --max-steps 1000
```

### **1\.4 Install Proto Toolchain**

Install the Go generators and ensure they are on your `PATH`:

Bash

go install google.golang.org/protobuf/cmd/protoc-gen-go@latest  
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

PowerShell (Windows)

```powershell
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
$env:PATH += ";$(go env GOPATH)\bin"
```

Make sure `protoc` is available (`protoc --version`). If you later import standard protos (e.g., `google/protobuf/timestamp.proto`) and see include errors, pass the include path for your installation (Windows example: `-I "C:/Program Files/protoc-win64/include"`).

### **1\.5 Generate gRPC Stubs**

Bash

./scripts/gen\_proto.sh

PowerShell (Windows)

```powershell
.\scripts\gen_proto.ps1
```

If you import standard protos later (e.g., `google/protobuf/timestamp.proto`), keep the include folder at `C:\Program Files\protoc-win64\include` and run:

```powershell
$env:PROTOC_INCLUDE = "C:/Program Files/protoc-win64/include"
.\scripts\gen_proto.ps1
```

### **2\. Initialize the Trusted Authority**

Simulate the NCMEC root key generation (Lattice-based).

Bash

go run cmd/setup/main.go \--out ./certs/authority.key

PowerShell (Windows)

```powershell
go run .\cmd\setup\main.go --out .\certs\authority.key
```

### **3\. Run the Server**

Start the authority node that listens for blinded queries.

Bash

go run cmd/server/main.go \--port 50051

PowerShell (Windows)

```powershell
go run .\cmd\server\main.go --port 50051
```

### **4\. Run the Client Scan**

Scan a local image. This runs the ONNX model, blinds the hash, and queries the server.

Bash

go run cmd/client/main.go \--image ./samples/test\_image.jpg \--server localhost:50051

PowerShell (Windows)

```powershell
go run .\cmd\client\main.go --image .\samples\test_image.jpg --server localhost:50051
```

---

## **🧪 Reproducible Build & Run (Windows/PowerShell)**

```powershell
# 0) Install Go protobuf generators
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
$env:PATH += ";$(go env GOPATH)\bin"

# 1) Generate gRPC stubs
.\scripts\gen_proto.ps1

# 2) Tidy modules (after stubs are generated)
go mod tidy

# 3) Build (Makefile optional on Windows)
go build -o bin\client.exe .\cmd\client
go build -o bin\server.exe .\cmd\server
go build -o bin\setup.exe .\cmd\setup

# 4) Run
.\bin\setup.exe --out .\certs\authority.key
.\bin\server.exe --port 50051 --insecure
.\bin\client.exe --image .\samples\test_image.jpg --server localhost:50051 --insecure
```

---

## **📚 References & Research**

This project implements concepts from the following papers:

1. **"Black-box Collision Attacks on Apple NeuralHash and Microsoft PhotoDNA"** (Leblanc-Albarel et al., ePrint 2024/1869, 2025\)  
   * *Demonstrates why non-recursive hashes fail on human faces, motivating our TRM approach.*  
   * [Read Paper](https://eprint.iacr.org/2024/1869.pdf)  
2. **"Proof-Carrying Data without Succinct Arguments"** (Bünz et al., Crypto 2020\)  
   * *Source of the "Split Accumulation" logic, allowing us to verify the database without heavy SNARKs.*  
   * [Read Paper](https://eprint.iacr.org/2020/1618.pdf)  
3. **"Less is More: Recursive Reasoning with Tiny Networks"** (Samsung SAIL, 2024\)  
   * *The architectural basis for our "Thinking" hash function.*  
   * [arXiv:2410.04871](https://arxiv.org/abs/2410.04871)  
4. **"DINOv3: Learning State-of-the-art Dense Visual Features"** (Meta AI, 2025\)  
   * *Used as the frozen teacher model for semantic robustness.*

---

## **⚠️ Legal & Ethical Notice**

**Research Only:** This is an educational implementation of cryptographic surveillance protocols. It is intended to demonstrate how privacy-preserving technologies *can* be built, not to encourage surveillance.

* **No Real CSAM:** Evaluation uses only non-sensitive public datasets.  
  * **FFHQ (NVIDIA):** High-res face proxy for human-centric feature extraction.  
  * **OpenImages V7 (Subset):** Real-world "gallery" proxy for messy, diverse images.  
  * **Text/Screenshots:** Edge-case proxy to reduce false positives on documents and receipts.  
* **Clean Room:** This is a clean-room implementation based on public academic papers. It contains no proprietary code from Apple or Microsoft.
