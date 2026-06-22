# scripts/runtime/detect_hardware.py
import os
import sys
import json
import argparse
import platform
import shutil
import subprocess
import torch
import psutil

def get_cpu_info():
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output("wmic cpu get name", shell=True).decode().split('\n')
            if len(out) > 1:
                cpu_name = out[1].strip()
        except Exception:
            pass
    cores = psutil.cpu_count(logical=True)
    return {"name": cpu_name, "cores": cores}

def get_gpu_info():
    cuda_available = torch.cuda.is_available()
    gpu_name = "N/A"
    vram_total_gb = 0.0
    driver_version = "N/A"
    cuda_version = "N/A"
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_total_gb = round(vram_bytes / (1024**3), 2)
        
    # Get driver and CUDA version from nvidia-smi
    try:
        smi_out = subprocess.check_output("nvidia-smi", shell=True).decode()
        for line in smi_out.split('\n'):
            if "Driver Version" in line:
                parts = line.split()
                # nvidia-smi typical layout:
                # | NVIDIA-SMI 539.41                 Driver Version: 539.41       CUDA Version: 12.2     |
                for i, part in enumerate(parts):
                    if "Version:" in part:
                        driver_version = parts[i+1]
                    if part == "CUDA" and i+1 < len(parts) and parts[i+1] == "Version:":
                        cuda_version = parts[i+2]
                break
    except Exception:
        pass
        
    return {
        "name": gpu_name,
        "vram_total_gb": vram_total_gb,
        "driver_version": driver_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version if cuda_version != "N/A" else (torch.version.cuda if cuda_available else "N/A")
    }

def get_git_info():
    commit = "N/A"
    dirty = False
    try:
        commit = subprocess.check_output("git rev-parse HEAD", shell=True).decode().strip()
        diff = subprocess.check_output("git status --porcelain", shell=True).decode().strip()
        dirty = len(diff) > 0
    except Exception:
        pass
    return {"commit": commit, "dirty": dirty}

def get_docker_info():
    docker_version = "N/A"
    try:
        docker_version = subprocess.check_output("docker --version", shell=True).decode().strip()
    except Exception:
        pass
    return docker_version

def main():
    parser = argparse.ArgumentParser(description="Detect server hardware and export report")
    parser.add_argument("--out", type=str, default="data/research/v08/hardware.json", help="Path to write JSON output")
    args = parser.parse_args()
    
    # Calculate free disk space
    total, used, free = shutil.disk_usage(os.getcwd())
    free_gb = round(free / (1024**3), 2)
    
    # OS Info
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    
    report = {
        "gpu": get_gpu_info(),
        "cpu": get_cpu_info(),
        "memory": {
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "software": {
            "os": os_name,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "docker": get_docker_info()
        },
        "disk": {
            "free_space_gb": free_gb
        },
        "git": get_git_info()
    }
    
    # Ensure directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Hardware report written to {args.out}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
