#!/usr/bin/env python3
"""
HuggingFace VRAM Estimator
Lightweight tool to estimate VRAM requirements for model inference.
Only depends on httpx.
"""

import argparse
import json
import os
import struct
import sys
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Dtype sizes in bytes
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "Q8_0": 1, "Q4_0": 0.5, "Q4_1": 0.5,
}

# Quantization multipliers (relative to fp16)
QUANT_MULTIPLIERS = {
    "float32": 2.0,
    "float16": 1.0,
    "bfloat16": 1.0,
    "int8": 0.5,
    "int4": 0.25,
}

HF_BASE_URL = "https://huggingface.co"
HF_API_URL = "https://huggingface.co/api/models"


def get_headers() -> Dict[str, str]:
    """Get HTTP headers with optional HF token for gated models."""
    headers = {"User-Agent": "hf-vram-estimator/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(client: httpx.Client, url: str) -> Optional[Dict]:
    """Fetch and parse JSON from URL."""
    try:
        resp = client.get(url)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code in (401, 403):
            print(f"Access denied. Set HF_TOKEN env var for gated models.")
            return None
        return None
    except Exception:
        return None


def fetch_safetensors_header(client: httpx.Client, url: str) -> Optional[Dict]:
    """
    Fetch only the metadata header from a safetensors file using range requests.
    Safetensors format: 8 bytes (header size) + N bytes (JSON header) + tensor data
    """
    try:
        # First, fetch the header size (first 8 bytes)
        resp = client.get(url, headers={"Range": "bytes=0-7"})
        if resp.status_code not in (200, 206):
            return None
        
        header_size = struct.unpack("<Q", resp.content)[0]
        
        # Sanity check - header shouldn't be larger than 100MB
        if header_size > 100 * 1024 * 1024:
            print(f"Warning: Unusually large header size: {header_size}")
            return None
        
        # Fetch the JSON header
        resp = client.get(url, headers={"Range": f"bytes=8-{8 + header_size - 1}"})
        if resp.status_code not in (200, 206):
            return None
        
        return json.loads(resp.content.decode("utf-8"))
    except Exception as e:
        print(f"Error fetching safetensors header: {e}")
        return None


def get_repo_files(client: httpx.Client, model_id: str) -> List[str]:
    """Get all files in the repository using HF API."""
    url = f"{HF_API_URL}/{model_id}"
    data = fetch_json(client, url)
    if data and "siblings" in data:
        return [f["rfilename"] for f in data["siblings"]]
    return []


def get_safetensors_files(client: httpx.Client, model_id: str) -> List[str]:
    """Get list of all safetensors files in the repository."""
    # Get all repo files via API
    all_files = get_repo_files(client, model_id)
    
    # Filter for safetensors files (excluding index files)
    safetensors_files = [
        f for f in all_files 
        if f.endswith(".safetensors") and not f.endswith(".index.json")
    ]
    
    if safetensors_files:
        return sorted(safetensors_files)
    
    # Fallback: try common locations
    common_paths = [
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
    ]
    
    for path in common_paths:
        url = f"{HF_BASE_URL}/{model_id}/resolve/main/{path}"
        resp = client.head(url)
        if resp.status_code == 200:
            return [path]
    
    return []


def get_all_configs(client: httpx.Client, model_id: str) -> Dict[str, Dict]:
    """Get all config.json files from the repository."""
    all_files = get_repo_files(client, model_id)
    configs = {}
    
    # Find all config.json files
    config_files = [f for f in all_files if f.endswith("config.json")]
    
    for config_file in config_files:
        url = f"{HF_BASE_URL}/{model_id}/resolve/main/{config_file}"
        config = fetch_json(client, url)
        if config:
            configs[config_file] = config
    
    return configs


def calculate_tensor_size(tensor_info: Dict) -> Tuple[int, str]:
    """Calculate size of a tensor in bytes. Returns (size, dtype)."""
    shape = tensor_info.get("shape", [])
    dtype = tensor_info.get("dtype", "F16")
    
    if not shape:
        return 0, dtype
    
    num_elements = reduce(lambda x, y: x * y, shape, 1)
    bytes_per_element = DTYPE_SIZES.get(dtype, 2)  # Default to 2 bytes (fp16)
    
    return int(num_elements * bytes_per_element), dtype


def get_model_weights_info(client: httpx.Client, model_id: str) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    """
    Get total model weights size, dtype distribution, and per-component breakdown.
    Returns: (total_bytes, {dtype: count}, {component: bytes})
    """
    files = get_safetensors_files(client, model_id)
    
    if not files:
        raise ValueError(f"No safetensors files found in {model_id}")
    
    total_bytes = 0
    dtype_counts: Dict[str, int] = {}
    component_sizes: Dict[str, int] = {}
    
    print(f"Found {len(files)} safetensors file(s)")
    
    for file in files:
        url = f"{HF_BASE_URL}/{model_id}/resolve/main/{file}"
        header = fetch_safetensors_header(client, url)
        
        if not header:
            print(f"Warning: Could not fetch header for {file}")
            continue
        
        file_bytes = 0
        for key, tensor_info in header.items():
            if key == "__metadata__":
                continue
            
            size, dtype = calculate_tensor_size(tensor_info)
            file_bytes += size
            total_bytes += size
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        # Track component sizes (by directory)
        component = file.split("/")[0] if "/" in file else "root"
        component_sizes[component] = component_sizes.get(component, 0) + file_bytes
    
    return total_bytes, dtype_counts, component_sizes


def extract_model_architecture(configs: Dict[str, Dict]) -> Dict[str, Any]:
    """Extract architecture info from configs, prioritizing transformer/LLM configs."""
    # Priority order for finding the main model config
    priority_keys = ["config.json", "transformer/config.json", "text_encoder/config.json"]
    
    main_config = None
    for key in priority_keys:
        if key in configs:
            main_config = configs[key]
            break
    
    if not main_config:
        # Use first available config
        main_config = next(iter(configs.values())) if configs else {}
    
    # Extract architecture parameters
    num_layers = (
        main_config.get("num_hidden_layers") or 
        main_config.get("n_layer") or 
        main_config.get("n_layers") or
        main_config.get("num_layers") or
        0
    )
    hidden_size = (
        main_config.get("hidden_size") or 
        main_config.get("n_embd") or 
        main_config.get("d_model") or
        0
    )
    num_attention_heads = (
        main_config.get("num_attention_heads") or 
        main_config.get("n_head") or
        main_config.get("num_heads") or
        0
    )
    num_kv_heads = (
        main_config.get("num_key_value_heads") or 
        main_config.get("num_kv_heads") or 
        num_attention_heads
    )
    max_context = (
        main_config.get("max_position_embeddings") or
        main_config.get("max_seq_len") or
        main_config.get("n_positions") or
        main_config.get("seq_length") or
        main_config.get("max_sequence_length") or
        2048
    )
    
    head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else 0
    
    model_type = main_config.get("model_type") or main_config.get("_class_name") or "unknown"
    
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_context": max_context,
        "model_type": model_type,
    }


def calculate_kv_cache_size(
    arch_info: Dict[str, Any],
    context_length: Optional[int] = None,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """
    Calculate KV cache size for transformer models.
    KV Cache = 2 * num_layers * num_kv_heads * head_dim * context_length * batch_size * dtype_bytes
    """
    num_layers = arch_info["num_layers"]
    num_kv_heads = arch_info["num_kv_heads"]
    head_dim = arch_info["head_dim"]
    max_ctx = context_length or arch_info["max_context"]
    
    if num_layers == 0 or num_kv_heads == 0 or head_dim == 0:
        return 0
    
    return 2 * num_layers * num_kv_heads * head_dim * max_ctx * batch_size * dtype_bytes


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human readable string."""
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def estimate_vram(
    model_id: str,
    context_length: Optional[int] = None,
    batch_size: int = 1,
    quantization: str = "float16",
) -> Dict[str, Any]:
    """
    Estimate VRAM requirements for a HuggingFace model.
    """
    with httpx.Client(headers=get_headers(), timeout=60.0, follow_redirects=True) as client:
        # Get all configs
        configs = get_all_configs(client, model_id)
        
        # Extract architecture info
        arch_info = extract_model_architecture(configs)
        
        # Get weights size
        weights_bytes, dtype_dist, component_sizes = get_model_weights_info(client, model_id)
        
        # Apply quantization multiplier
        quant_mult = QUANT_MULTIPLIERS.get(quantization, 1.0)
        adjusted_weights = int(weights_bytes * quant_mult)
        
        # Calculate KV cache
        dtype_bytes = 2 if quantization in ("float16", "bfloat16") else 4 if quantization == "float32" else 2
        effective_context = context_length or arch_info["max_context"]
        kv_cache_bytes = calculate_kv_cache_size(arch_info, effective_context, batch_size, dtype_bytes)
        
        # Activation memory overhead (rough estimate)
        activation_overhead = int(adjusted_weights * 0.1)
        
        # CUDA overhead
        cuda_overhead = 500 * 1024 * 1024
        
        total_vram = adjusted_weights + kv_cache_bytes + activation_overhead + cuda_overhead
        
        # Format component sizes for readability
        component_sizes_formatted = {k: format_bytes(v) for k, v in component_sizes.items()}
        
        return {
            "model_id": model_id,
            "architecture": arch_info,
            "dtype_distribution": dtype_dist,
            "component_sizes": component_sizes_formatted,
            "quantization": quantization,
            "batch_size": batch_size,
            "context_length": effective_context,
            "estimates": {
                "model_weights": format_bytes(adjusted_weights),
                "model_weights_raw": format_bytes(weights_bytes),
                "kv_cache": format_bytes(kv_cache_bytes),
                "activation_overhead": format_bytes(activation_overhead),
                "cuda_overhead": format_bytes(cuda_overhead),
                "total": format_bytes(total_vram),
            },
            "estimates_bytes": {
                "model_weights": adjusted_weights,
                "model_weights_raw": weights_bytes,
                "kv_cache": kv_cache_bytes,
                "activation_overhead": activation_overhead,
                "cuda_overhead": cuda_overhead,
                "total": total_vram,
            }
        }


def print_report(result: Dict[str, Any]):
    """Print formatted VRAM estimation report."""
    arch = result["architecture"]
    est = result["estimates"]
    est_bytes = result["estimates_bytes"]
    
    print(f"\n{'='*60}")
    print(f"VRAM Estimation Report: {result['model_id']}")
    print(f"{'='*60}")
    
    # Component breakdown
    if result["component_sizes"]:
        print(f"\nModel Components:")
        for comp, size in sorted(result["component_sizes"].items(), key=lambda x: x[0]):
            print(f"  - {comp}: {size}")
    
    # Architecture (only if detected)
    if arch["num_layers"] > 0:
        print(f"\nModel Architecture ({arch['model_type']}):")
        print(f"  - Layers:          {arch['num_layers']}")
        print(f"  - Hidden Size:     {arch['hidden_size']}")
        print(f"  - Attention Heads: {arch['num_attention_heads']}")
        print(f"  - KV Heads (GQA):  {arch['num_kv_heads']}")
        print(f"  - Head Dimension:  {arch['head_dim']}")
        print(f"  - Max Context:     {arch['max_context']:,} tokens")
    else:
        print(f"\nModel Type: {arch['model_type']}")
        print("  (Non-transformer or diffusion model - KV cache N/A)")
    
    print(f"\nDtype Distribution:")
    for dtype, count in result["dtype_distribution"].items():
        print(f"  - {dtype}: {count} tensors")
    
    print(f"\nInference Configuration:")
    print(f"  - Quantization:    {result['quantization']}")
    print(f"  - Batch Size:      {result['batch_size']}")
    if arch["num_layers"] > 0:
        print(f"  - Context Length:  {result['context_length']:,} tokens")
    
    print(f"\n{'-'*60}")
    print(f"VRAM Breakdown:")
    print(f"{'-'*60}")
    print(f"  Model Weights:       {est['model_weights']:>12}")
    if est_bytes['kv_cache'] > 0:
        print(f"  KV Cache:            {est['kv_cache']:>12}")
    print(f"  Activations (est):   {est['activation_overhead']:>12}")
    print(f"  CUDA Overhead:       {est['cuda_overhead']:>12}")
    print(f"{'-'*60}")
    print(f"  TOTAL REQUIRED:      {est['total']:>12}")
    print(f"{'='*60}")
    
    # GPU recommendations
    total_gb = est_bytes['total'] / (1024**3)
    print(f"\nGPU Recommendations:")
    if total_gb <= 8:
        print("  - RTX 3070/4070 (8GB) or better")
    elif total_gb <= 12:
        print("  - RTX 3080/4080 (12GB) or better")
    elif total_gb <= 16:
        print("  - RTX 4080 Super (16GB) or better")
    elif total_gb <= 24:
        print("  - RTX 3090/4090 (24GB) or better")
    elif total_gb <= 48:
        print("  - A6000 (48GB) or 2x 24GB GPUs")
    elif total_gb <= 80:
        print("  - A100 (80GB) or multi-GPU setup")
    else:
        print(f"  - Multi-GPU setup required ({total_gb:.1f}GB needed)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate VRAM requirements for HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hf_vram_estimator.py mistralai/Mistral-7B-v0.1
  python hf_vram_estimator.py Qwen/Qwen-Image-Edit-2511
  python hf_vram_estimator.py meta-llama/Llama-2-70b-hf --quantization int8
  python hf_vram_estimator.py Qwen/Qwen2-7B --context_length 32768 --batch_size 4

Environment Variables:
  HF_TOKEN: Set for accessing gated models (e.g., Llama)
        """
    )
    
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., Qwen/Qwen2-7B)")
    parser.add_argument("--context_length", "-c", type=int, help="Override context length")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument(
        "--quantization", "-q",
        choices=["float32", "float16", "bfloat16", "int8", "int4"],
        default="float16",
        help="Target quantization (default: float16)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    print(f"Fetching metadata for: {args.model_id} ...")
    
    try:
        result = estimate_vram(
            model_id=args.model_id,
            context_length=args.context_length,
            batch_size=args.batch_size,
            quantization=args.quantization,
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_report(result)
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError as e:
        print(f"HTTP Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
