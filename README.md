# HuggingFace VRAM Estimator

A lightweight Python tool to estimate VRAM requirements for running inference on any HuggingFace model. It fetches only metadata (safetensors headers) without downloading the full model weights, making it fast and bandwidth-efficient.

## Features

- **Lightweight**: Only depends on `httpx` - no PyTorch, Transformers, or other heavy dependencies
- **Fast**: Downloads only ~KB of metadata instead of GB of model weights
- **Universal**: Works with any safetensors repository (LLMs, diffusion models, vision models, etc.)
- **Accurate**: Parses actual tensor shapes and dtypes from safetensors headers
- **KV Cache Aware**: Calculates memory for context window with GQA (Grouped Query Attention) support
- **Quantization Support**: Estimates for float32, float16, bfloat16, int8, and int4
- **Component Breakdown**: Shows VRAM usage per model component (text_encoder, transformer, vae, etc.)

## Requirements

- Python 3.8+
- httpx

## Installation

### 1. Clone or Download

```bash
# Clone the repository or download the script
git clone <repository-url>
cd hf-vram-estimator

# Or simply download the single file
curl -O https://raw.githubusercontent.com/<repo>/hf_vram_estimator.py
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install httpx
```

Or if you use `uv`:

```bash
uv pip install httpx
```

### 4. (Optional) Set HuggingFace Token for Gated Models

For gated models like Llama, you need a HuggingFace token:

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here

# On Windows:
set HF_TOKEN=hf_your_token_here
```

## Usage

### Basic Usage

```bash
python hf_vram_estimator.py <model_id>
```

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `model_id` | | HuggingFace model ID (required) |
| `--context_length` | `-c` | Override default context length |
| `--batch_size` | `-b` | Batch size for inference (default: 1) |
| `--quantization` | `-q` | Target precision: float32, float16, bfloat16, int8, int4 (default: float16) |
| `--json` | | Output results as JSON |

## Examples

### Example 1: Estimate VRAM for Mistral-7B

```bash
python hf_vram_estimator.py mistralai/Mistral-7B-v0.1
```

**Output:**
```
Fetching metadata for: mistralai/Mistral-7B-v0.1 ...
Found 2 safetensors file(s)

============================================================
VRAM Estimation Report: mistralai/Mistral-7B-v0.1
============================================================

Model Components:
  - root: 13.49 GB

Model Architecture (mistral):
  - Layers:          32
  - Hidden Size:     4096
  - Attention Heads: 32
  - KV Heads (GQA):  8
  - Head Dimension:  128
  - Max Context:     32,768 tokens

Dtype Distribution:
  - BF16: 291 tensors

Inference Configuration:
  - Quantization:    float16
  - Batch Size:      1
  - Context Length:  32,768 tokens

------------------------------------------------------------
VRAM Breakdown:
------------------------------------------------------------
  Model Weights:           13.49 GB
  KV Cache:                 4.00 GB
  Activations (est):        1.35 GB
  CUDA Overhead:          500.00 MB
------------------------------------------------------------
  TOTAL REQUIRED:          19.33 GB
============================================================

GPU Recommendations:
  - RTX 3090/4090 (24GB) or better
```

### Example 2: Estimate with INT4 Quantization

```bash
python hf_vram_estimator.py Qwen/Qwen2-7B --quantization int4
```

**Output:**
```
...
VRAM Breakdown:
------------------------------------------------------------
  Model Weights:            3.55 GB
  KV Cache:                32.00 GB
  Activations (est):      363.14 MB
  CUDA Overhead:          500.00 MB
------------------------------------------------------------
  TOTAL REQUIRED:          36.38 GB
```

### Example 3: Custom Context Length

```bash
python hf_vram_estimator.py Qwen/Qwen2-7B --context_length 8192 --quantization int4
```

**Output:**
```
...
VRAM Breakdown:
------------------------------------------------------------
  Model Weights:            3.55 GB
  KV Cache:               448.00 MB
  Activations (est):      363.14 MB
  CUDA Overhead:          500.00 MB
------------------------------------------------------------
  TOTAL REQUIRED:           4.83 GB
============================================================

GPU Recommendations:
  - RTX 3070/4070 (8GB) or better
```

### Example 4: Multi-Component Diffusion Model

```bash
python hf_vram_estimator.py Qwen/Qwen-Image-Edit-2511
```

**Output:**
```
Fetching metadata for: Qwen/Qwen-Image-Edit-2511 ...
Found 10 safetensors file(s)

============================================================
VRAM Estimation Report: Qwen/Qwen-Image-Edit-2511
============================================================

Model Components:
  - text_encoder: 15.45 GB
  - transformer: 38.05 GB
  - vae: 242.03 MB

Model Type: QwenImageTransformer2DModel
  (Non-transformer or diffusion model - KV cache N/A)

Dtype Distribution:
  - BF16: 2856 tensors

...
------------------------------------------------------------
VRAM Breakdown:
------------------------------------------------------------
  Model Weights:           53.74 GB
  Activations (est):        5.37 GB
  CUDA Overhead:          500.00 MB
------------------------------------------------------------
  TOTAL REQUIRED:          59.60 GB
============================================================

GPU Recommendations:
  - A100 (80GB) or multi-GPU setup
```

### Example 5: JSON Output for Scripting

```bash
python hf_vram_estimator.py gpt2 --json
```

**Output:**
```json
{
  "model_id": "gpt2",
  "architecture": {
    "num_layers": 12,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_kv_heads": 12,
    "head_dim": 64,
    "max_context": 1024,
    "model_type": "gpt2"
  },
  "dtype_distribution": {
    "F32": 160
  },
  "component_sizes": {
    "root": "522.81 MB"
  },
  "quantization": "float16",
  "batch_size": 1,
  "context_length": 1024,
  "estimates": {
    "model_weights": "522.81 MB",
    "model_weights_raw": "522.81 MB",
    "kv_cache": "36.00 MB",
    "activation_overhead": "52.28 MB",
    "cuda_overhead": "500.00 MB",
    "total": "1.09 GB"
  },
  "estimates_bytes": {
    "model_weights": 548090880,
    "model_weights_raw": 548090880,
    "kv_cache": 37748736,
    "activation_overhead": 54809088,
    "cuda_overhead": 524288000,
    "total": 1164936704
  }
}
```

### Example 6: Batch Size for Throughput Estimation

```bash
python hf_vram_estimator.py mistralai/Mistral-7B-v0.1 --batch_size 8 --context_length 4096
```

This estimates VRAM needed to run 8 concurrent requests with 4K context each.

### Example 7: Gated Model (Requires Token)

```bash
export HF_TOKEN=hf_your_token_here
python hf_vram_estimator.py meta-llama/Llama-2-70b-hf --quantization int8
```

## Understanding the Output

### VRAM Components

| Component | Description |
|-----------|-------------|
| **Model Weights** | Memory to store model parameters (adjusted for quantization) |
| **KV Cache** | Memory for Key-Value cache during autoregressive generation |
| **Activations** | Intermediate computation memory (~10% of model weights) |
| **CUDA Overhead** | Base CUDA runtime and kernel memory (~500MB) |

### KV Cache Formula

```
KV Cache = 2 × layers × kv_heads × head_dim × context_length × batch_size × bytes_per_param
```

For GQA models (like Mistral, Llama 2), `kv_heads` < `attention_heads`, which significantly reduces KV cache size.

### Quantization Impact

| Precision | Bytes per Param | Relative Size |
|-----------|-----------------|---------------|
| float32 | 4 | 2x |
| float16/bfloat16 | 2 | 1x (baseline) |
| int8 | 1 | 0.5x |
| int4 | 0.5 | 0.25x |

## Limitations

- Only works with safetensors format (not .bin or .gguf)
- Activation memory is estimated (~10% overhead)
- Does not account for framework-specific optimizations (Flash Attention, etc.)
- Actual VRAM may vary based on inference framework (vLLM, TGI, llama.cpp, etc.)

## Troubleshooting

### "No safetensors files found"

The model might only have PyTorch .bin files. This tool only supports safetensors format.

### "Access denied"

The model is gated. Set your HuggingFace token:
```bash
export HF_TOKEN=hf_your_token_here
```

### Timeout errors

Increase timeout for slow connections or try again later.

## License

MIT License

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
