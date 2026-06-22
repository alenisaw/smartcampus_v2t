# scripts/runtime/export_hardware_report.py
import argparse
import json
import os

def export_report(json_path, out_path):
    if not os.path.exists(json_path):
        print(f"Error: JSON report not found at {json_path}")
        return False
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    gpu = data.get("gpu", {})
    cpu = data.get("cpu", {})
    mem = data.get("memory", {})
    soft = data.get("software", {})
    disk = data.get("disk", {})
    git_info = data.get("git", {})
    
    md = f"""# SmartCampus V2T v0.8 Hardware & Environment Report

## System Specification

### Hardware
- **GPU Name**: {gpu.get("name", "N/A")}
- **Dedicated VRAM**: {gpu.get("vram_total_gb", 0.0)} GB
- **Driver Version**: {gpu.get("driver_version", "N/A")}
- **CUDA Version**: {gpu.get("cuda_version", "N/A")}
- **CPU Model**: {cpu.get("name", "N/A")}
- **CPU Logical Cores**: {cpu.get("cores", 0)}
- **System Memory (RAM)**: {mem.get("ram_total_gb", 0.0)} GB
- **Available Disk Space**: {disk.get("free_space_gb", 0.0)} GB

### Software Environment
- **Operating System**: {soft.get("os", "N/A")}
- **Python Version**: {soft.get("python", "N/A")}
- **PyTorch Version**: {soft.get("torch", "N/A")}
- **CUDA Runtime (PyTorch)**: {soft.get("cuda_runtime", "N/A")}
- **Docker Version**: {soft.get("docker", "N/A")}

### Code Registry Status
- **Git Commit Hash**: {git_info.get("commit", "N/A")}
- **Git Repository Dirty State**: {git_info.get("dirty", False)}
"""
    
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"Markdown hardware report written to {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export hardware JSON report to markdown")
    parser.add_argument("--json", type=str, default="data/research/v08/hardware.json")
    parser.add_argument("--out", type=str, default="data/research/v08/hardware_report.md")
    args = parser.parse_args()
    
    export_report(args.json, args.out)

if __name__ == "__main__":
    main()
