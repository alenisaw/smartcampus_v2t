# scripts/runtime/benchmark_vlm_batch.py
"""Synthetic CUDA allocation probe.

This does not load or execute the VLM and must never be reported as model
validation. It only rejects obviously unsafe candidate allocation sizes.
"""

import argparse
import json
import sys
import time
import torch

PROBE_TYPE = "synthetic_cuda_memory_probe"


def benchmark_batch(batch_size, max_batch_frames, resolution, device):
    print(f"Running synthetic memory probe: batch_size={batch_size}, max_batch_frames={max_batch_frames}, resolution={resolution} on {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but requested for benchmark")
        
    t0 = time.time()
    try:
        # Simulate loading the model and allocating activations memory.
        # For Qwen VL 2B, activations and model weights occupy significant space.
        # Let's allocate tensors to simulate typical model GPU memory overhead.
        # 2B parameter model in float16 uses ~4GB of VRAM.
        # A batch of frames of size (batch_size, 3, resolution[0], resolution[1])
        w, h = resolution
        
        # Allocate dummy weight tensors to simulate model parameters (e.g. 2 GB allocation)
        simulated_weights = torch.empty((1024 * 1024 * 256,), dtype=torch.float16, device=device)
        
        # Allocate activations for batch_size * max_batch_frames
        # Typically a frame might be encoded into a grid of tokens (e.g. 256 tokens of dim 2048)
        simulated_inputs = torch.empty((batch_size, max_batch_frames, 256, 2048), dtype=torch.float16, device=device)
        
        # Do dummy matrix multiplications to simulate forward pass
        for _ in range(5):
            res = torch.matmul(simulated_inputs, simulated_inputs.transpose(-1, -2))
            torch.cuda.synchronize()
            
        # Free memory
        del simulated_weights
        del simulated_inputs
        del res
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        duration = time.time() - t0
        return {
            "ok": True,
            "probe_type": PROBE_TYPE,
            "vlm_loaded": False,
            "vlm_inference_validated": False,
            "duration_sec": round(duration, 3),
            "allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024**2), 2)
        }
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM occurred: {e}")
        return {"ok": False, "probe_type": PROBE_TYPE, "vlm_loaded": False, "vlm_inference_validated": False, "error": "OOM"}
    except Exception as e:
        print(f"Unexpected benchmark error: {e}")
        return {"ok": False, "probe_type": PROBE_TYPE, "vlm_loaded": False, "vlm_inference_validated": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Synthetic CUDA memory probe (does not load or validate the VLM)")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-batch-frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    res = benchmark_batch(args.batch_size, args.max_batch_frames, (args.width, args.height), args.device)
    if res["ok"]:
        print("MEMORY_PROBE_PASSED (NOT VLM VALIDATION): " + json.dumps(res, sort_keys=True))
        sys.exit(0)
    else:
        print("MEMORY_PROBE_FAILED (NOT VLM VALIDATION): " + json.dumps(res, sort_keys=True))
        sys.exit(1)

if __name__ == "__main__":
    main()
