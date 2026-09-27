# PaliGemma from Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat&logo=googlecolab)](https://colab.research.google.com/)
[![HuggingFace](https://img.shields.io/badge/Weights-google%2Fpaligemma--3b--mix--224-yellow?style=flat&logo=huggingface)](https://huggingface.co/google/paligemma-3b-mix-224)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A clean, modular implementation of Google's PaliGemma Vision-Language Model (VLM) written completely from scratch in PyTorch.

This project implements the complete multimodal pipeline without relying on high-level modeling wrappers:
- **SigLIP Vision Tower**: Vision Transformer extracting spatial patch embeddings from raw images.
- **Multimodal Projector**: Linear adapter projecting visual representations into the language model's hidden dimension.
- **Gemma 2B Causal Language Model**: Transformer decoder featuring RMSNorm with unit-offset scaling, Rotary Position Embeddings (RoPE), GeGLU feed-forward networks, and dynamic Key-Value (KV) caching for autoregressive text and coordinate generation.
- **Streaming Shard Loader**: Direct in-place weight streaming from safetensors into GPU VRAM in `torch.bfloat16`, avoiding host RAM double-buffering.

---

## Architecture Overview

```
Input Image (224x224x3)            Input Text Prompt ("caption en", "detect person", etc.)
          │                                              │
          ▼                                              ▼
┌───────────────────────────┐                  ┌───────────────────────────┐
│   SigLIP Vision Tower     │                  │  Gemma Tokenizer (256k)   │
│  - 27 Transformer Blocks  │                  │  - Prepends 256 <image>   │
│  - Patch Size: 14x14      │                  │    placeholder tokens     │
│  - 256 visual tokens      │                  └─────────────┬─────────────┘
└─────────────┬─────────────┘                                │
              │ (Batch, 256, 1152)                           │
              ▼                                              │
┌───────────────────────────┐                                │
│ Multimodal Linear Adapter │                                │
│    Linear(1152 -> 2048)   │                                │
└─────────────┬─────────────┘                                │
              │ (Batch, 256, 2048)                           │
              └───────────────────────┬──────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Unified Token Embeddings │
                        │   (256 Image + Text Tokens)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Gemma 2B Decoder       │
                        │  - 18 Transformer Layers  │
                        │  - RoPE Positional Embed  │
                        │  - Unit-Offset RMSNorm    │
                        │  - GeGLU Feedforward      │
                        │  - Fast KV Caching        │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Autoregressive Generation │
                        │   Text / Coordinates      │
                        └───────────────────────────┘
```

---

## Supported Tasks

PaliGemma is trained with task-specific prefix prompts to perform diverse multimodal operations:

1. **Image Captioning**: Detailed visual descriptions (`caption en`).
2. **Visual Question Answering (VQA)**: Context-aware answers grounded directly in image pixels.
3. **Document Understanding & OCR**: Text reading from images, receipts, and signs (`ocr`).
4. **Object Detection**: Identifies target objects and predicts normalized spatial bounding boxes (`<loc0000>` to `<loc1023>`), with automated coordinate de-normalization and visual rendering.
5. **Visual Segmentation**: Predicts boundary coordinates for designated categories (`segment <target>`).
6. **Spatial Reasoning**: Positional and relational query answering across complex multi-object scenes.

---

## Directory Structure

```
gemma_new/
├── config.py                   # Model configurations (SigLIP, Gemma, PaliGemma)
├── siglip.py                   # SigLIP Vision Transformer implementation
├── model.py                    # Gemma LM, Projector, and PaliGemmaForConditionalGeneration
├── processor.py                # Image normalization, prompt formatting, tokenization
├── pretrained_weight_loader.py # In-place streaming shard loader for safetensors
├── tasks.py                    # Task pipelines, prompt templates, bounding box visualizer
├── run_tasks.py                # CLI entry point for running inference
├── paligemma_demo.ipynb        # Interactive notebook with inline visualizations
├── test_loader.py              # Shard loading and parameter assignment tests
└── test_paligemma.py           # Integration and forward pass validation tests
```

---

## Requirements & Setup

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mahmudulhasanrabbidu/paligemma-from-scratch.git
cd paligemma-from-scratch

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install torch torchvision transformers safetensors accelerate pillow matplotlib huggingface_hub
```

### 2. Download Pretrained Weights

PaliGemma weights are gated on Hugging Face. Ensure your account has access accepted on [google/paligemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224).

Download the weights via Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/paligemma-3b-mix-224",
    local_dir="paligemma-weights/paligemma-3b-mix-224",
    token="YOUR_HF_TOKEN"
)
```

---

## Usage

### Command Line Interface

Run all multimodal tasks or target a specific task on an image:

```bash
# Run all tasks sequentially:
python run_tasks.py --image_path test_images/pic1.jpeg --task all --device cuda

# Image Captioning:
python run_tasks.py --image_path test_images/pic1.jpeg --task caption --device cuda

# Visual Question Answering:
python run_tasks.py --image_path test_images/pic1.jpeg --task vqa \
    --question "Where is the photographer resting?" --device cuda

# Object Detection (draws bounding boxes and saves to outputs/):
python run_tasks.py --image_path test_images/pic1.jpeg --task detect \
    --target person --device cuda

# Document OCR:
python run_tasks.py --image_path test_images/receipt.jpg --task ocr --device cuda
```

### Python API

```python
import torch
from PIL import Image
from transformers import AutoTokenizer
from gemma_new import (
    PretrainedPaliGemmaModel,
    PaliGemmaProcessor,
    PaliGemmaPipeline,
    draw_bounding_boxes
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
model_path = "paligemma-weights/paligemma-3b-mix-224"

# 1. Load model onto GPU
model, _ = PretrainedPaliGemmaModel.from_pretrained(
    model_path=model_path,
    device=device,
    dtype=dtype,
)
model.eval()

# 2. Initialize tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = PaliGemmaProcessor(
    tokenizer=tokenizer,
    num_image_tokens=model.config.vision_config.num_image_tokens,
    image_size=model.config.vision_config.image_size,
)

# 3. Create pipeline
pipeline = PaliGemmaPipeline(model=model, processor=processor, device=device)

# 4. Generate image caption
image = Image.open("test_images/pic1.jpeg").convert("RGB")
caption = pipeline.caption(image)
print("Caption:", caption)

# 5. Detect objects and render bounding boxes
detections, raw_output = pipeline.detect(image, target="person")
annotated_image = draw_bounding_boxes(image, detections)
annotated_image.save("detected_person.jpg")
```

### Interactive Notebook

The notebook `paligemma_demo.ipynb` provides an end-to-end interactive workflow for both local environments and Google Colab GPU runtimes:
- **Environment Bootstrap**: Auto-installs missing dependencies and sets up module paths.
- **Hands-On Shape Inspection**: Step-by-step tensor inspection across SigLIP, projector, embedding merger, and Gemma logits.
- **Visual Output Display**: High-resolution matplotlib rendering of annotated detection boxes, OCR text, and VQA responses.

---

## Technical Highlights

### 1. In-Place Tensor Streaming
Standard checkpoint loading often loads entire state dictionaries into system RAM before moving weights to the GPU, causing out-of-memory errors on modest setups. The loader in `pretrained_weight_loader.py` iterates over safetensors shards and assigns weights directly to their target module parameters in `bfloat16`:
```python
with safe_open(shard_path, framework="pt", device=device) as f:
    for hf_key in f.keys():
        tensor = f.get_tensor(hf_key).to(dtype=dtype)
        # In-place parameter assignment without RAM duplicate
```

### 2. KV-Caching for Generative Decoding
To avoid recomputing key-value states for the 256 visual tokens and prompt tokens at each autoregressive step, `GemmaAttention` implements an efficient Key-Value cache. The cache is pre-filled during the prompt evaluation phase and updated with a single token per decode iteration.

### 3. Spatial Grounding Coordinate De-Normalization
PaliGemma outputs spatial coordinates as discrete vocabulary tokens `<loc0000>` to `<loc1023>`. These represent normalized values in the range [0, 1000]. The post-processor extracts coordinate tokens via regular expressions and rescales them to original image pixel coordinates:

$$
x = \left(\frac{\text{coord}}{1000}\right) \times W
$$

$$
y = \left(\frac{\text{coord}}{1000}\right) \times H
$$

Or in Python:
```python
x_pixel = int((coord / 1000.0) * image_width)
y_pixel = int((coord / 1000.0) * image_height)
```

---

## Uploading Only `gemma_new` to GitHub

To publish only the `gemma_new` folder contents to your GitHub repository:

```bash
# Navigate to the gemma_new directory
cd "/home/rabbi/Desktop/LLM/umar_jamil/Multimodal (Vision) Language Model from scratch in PyTorch/gemma_new"

# Initialize git repository
git init

# Add remote repository
git remote add origin https://github.com/mahmudulhasanrabbidu/paligemma-from-scratch.git

# Stage all files inside gemma_new
git add .

# Create initial commit and push
git commit -m "Initial commit: PaliGemma from scratch implementation"
git branch -M main
git push -u origin main --force
```

---

## References

- Beyer et al., "PaliGemma: A versatile 3B VLM for transfer", Google DeepMind, 2024. [arXiv:2405.04419](https://arxiv.org/abs/2405.04419)
- Zhai et al., "Sigmoid Loss for Language Image Pre-Training", 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)

---

## License

This project is licensed under the MIT License.
