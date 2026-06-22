# scripts/runtime/autotune_hardware.py
import argparse
import json
import os
import subprocess
import sys
import yaml

def load_hardware_report():
    report_path = "data/research/v08/hardware.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Run detect_hardware inline or return fallback
        print("Hardware report not found, running detection...")
        try:
            subprocess.check_call(f"{sys.executable} scripts/runtime/detect_hardware.py", shell=True)
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to detect hardware inline: {e}")
            return None

def run_memory_probe(batch_size, max_batch_frames, width, height, device):
    cmd = (
        f"{sys.executable} scripts/runtime/benchmark_vlm_batch.py "
        f"--batch-size {batch_size} --max-batch-frames {max_batch_frames} "
        f"--width {width} --height {height} --device {device}"
    )
    print(f"Running synthetic CUDA memory probe (not VLM validation): {cmd}")
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print("Memory probe passed; real VLM inference remains unvalidated.")
            return True, res.stdout
        else:
            print("Memory probe failed.")
            return False, res.stdout + res.stderr
    except Exception as e:
        print(f"Failed to execute benchmark subprocess: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Autotune pipeline parameters based on hardware")
    parser.add_argument("--profile-template", type=str, default="configs/profiles/v08_base.yaml")
    parser.add_argument("--out", type=str, default="configs/generated/v08_auto_server.yaml")
    parser.add_argument("--target", type=str, default="balanced")
    args = parser.parse_args()
    
    # Load template config
    if not os.path.exists(args.profile_template):
        print(f"Error: Profile template not found: {args.profile_template}")
        sys.exit(1)
        
    with open(args.profile_template, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    hw = load_hardware_report()
    vram = 0.0
    gpu_name = "CPU only"
    cuda_available = False
    
    if hw and hw.get("gpu"):
        gpu_name = hw["gpu"].get("name", "N/A")
        vram = hw["gpu"].get("vram_total_gb", 0.0)
        cuda_available = hw["gpu"].get("cuda_available", False)
        
    print(f"Detected GPU: {gpu_name} ({vram} GB VRAM)")
    
    # Determine base values based on VRAM
    if not cuda_available or vram < 6.0:
        tier = "Low / CPU-only"
        start_batch_size = 2
        start_max_batch_frames = 48
        device = "cpu"
    elif vram < 12.0:
        tier = "Low VRAM GPU"
        start_batch_size = 2
        start_max_batch_frames = 48
        device = "cuda"
    elif vram < 24.0:
        tier = "Medium VRAM GPU"
        start_batch_size = 4
        start_max_batch_frames = 72
        device = "cuda"
    else:
        tier = "High VRAM GPU (RTX 5000 Ada class)"
        start_batch_size = 6
        start_max_batch_frames = 96
        device = "cuda"
        
    print(f"Tier selection: {tier}")
    
    # Tuning loop
    batch_size = start_batch_size
    max_batch_frames = start_max_batch_frames
    
    tuned_bs = batch_size
    tuned_mbf = max_batch_frames
    benchmark_log = []
    
    width, height = config.get("video", {}).get("decode_resolution", [1024, 576])
    
    if device == "cuda":
        # Let's try to scale UP if on a very powerful GPU, or adapt down if we encounter OOM
        # We start with the base tier values
        options = []
        if start_batch_size == 6:
            # We can try to upscale to 8 or 10 if 6 succeeds
            options = [(6, 96), (8, 128), (10, 160)]
        elif start_batch_size == 4:
            options = [(4, 72), (6, 96)]
        else:
            options = [(2, 48), (4, 72)]
            
        # We will benchmark sequentially
        for bs, mbf in options:
            ok, output = run_memory_probe(bs, mbf, width, height, device)
            benchmark_log.append({
                "batch_size": bs,
                "max_batch_frames": mbf,
                "probe_type": "synthetic_cuda_memory_probe",
                "vlm_inference_validated": False,
                "status": "MEMORY_PROBE_PASSED" if ok else "MEMORY_PROBE_FAILED",
                "details": output
            })
            if ok:
                tuned_bs = bs
                tuned_mbf = mbf
            else:
                # If a benchmark fails, we do not try higher options
                break
    else:
        benchmark_log.append({
            "batch_size": batch_size,
            "max_batch_frames": max_batch_frames,
            "probe_type": "hardware_tier_only",
            "vlm_inference_validated": False,
            "status": "NOT_RUN",
            "details": "No model inference or synthetic CUDA probe was run on CPU."
        })
        tuned_bs = batch_size
        tuned_mbf = max_batch_frames
        
    print(f"Tuned parameters: batch_size={tuned_bs}, max_batch_frames={tuned_mbf}")
    
    # Apply tuned parameters
    if "model" not in config:
        config["model"] = {}
    config["model"]["batch_size"] = tuned_bs
    config["model"]["max_batch_clips"] = tuned_bs
    config["model"]["max_batch_frames"] = tuned_mbf
    config["model"]["device"] = device
    
    # Save tuned config
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
        
    print(f"Autotuned config written to {args.out}")
    
    # Save autotune report
    autotune_report = {
        "validation": {
            "type": "synthetic_cuda_memory_probe" if device == "cuda" else "hardware_tier_only",
            "vlm_loaded": False,
            "vlm_inference_validated": False,
            "claim": "Provisional allocation sizing only; run a real video/VLM smoke test before experiments."
        },
        "tier": tier,
        "hardware": {
            "gpu": gpu_name,
            "vram_gb": vram
        },
        "tuned_parameters": {
            "batch_size": tuned_bs,
            "max_batch_frames": tuned_mbf,
            "device": device
        },
        "benchmark_log": benchmark_log
    }
    
    report_path = "data/research/v08/autotune_report.json"
    report_dir = os.path.dirname(report_path)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)
        
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(autotune_report, f, indent=2, ensure_ascii=False)
        
    print(f"Autotune report written to {report_path}")

if __name__ == "__main__":
    main()
