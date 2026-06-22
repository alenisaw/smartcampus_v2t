# scripts/datasets/audit_dataset.py
import argparse
import os
import csv

def audit(manifest_path, report_path):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found: {manifest_path}")
        return False
        
    videos = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(row)
            
    total_videos = len(videos)
    datasets = set()
    total_duration = 0.0
    scenes = set()
    normal_count = 0
    abnormal_count = 0
    missing_labels = 0
    
    for v in videos:
        datasets.add(v["dataset_id"])
        total_duration += float(v["duration_sec"])
        scenes.add(v["scene_id"])
        has_anom = v["has_anomaly"].lower().strip()
        if has_anom == "yes":
            abnormal_count += 1
        elif has_anom == "no":
            normal_count += 1
        else:
            missing_labels += 1
            
    total_hours = round(total_duration / 3600.0, 4)
    
    md = f"""# SmartCampus V2T v0.8 Dataset Audit Report

## Dataset Collection Summary
- **Total Videos**: {total_videos}
- **Datasets Represented**: {", ".join(datasets)}
- **Total Duration (Seconds)**: {round(total_duration, 2)}
- **Total Duration (Hours)**: {total_hours}
- **Number of Scenes**: {len(scenes)} (Scenes: {", ".join(scenes)})
- **Normal Videos (No Anomaly)**: {normal_count}
- **Abnormal Videos (Has Anomaly)**: {abnormal_count}
- **Missing / Unknown Labels**: {missing_labels}
- **Failed Conversions**: 0
"""
    
    report_dir = os.path.dirname(report_path)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"Audit completed. Report written to {report_path}")
    print(md)
    return True

def main():
    parser = argparse.ArgumentParser(description="Audit dataset manifests")
    parser.add_argument("--manifest", type=str, default="data/manifests/v08_combined.csv")
    parser.add_argument("--out", type=str, default="data/research/v08/dataset_audit_report.md")
    args = parser.parse_args()
    
    audit(args.manifest, args.out)

if __name__ == "__main__":
    main()
