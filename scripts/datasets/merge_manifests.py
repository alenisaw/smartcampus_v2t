# scripts/datasets/merge_manifests.py
import argparse
import csv
import os

REQUIRED_COLUMNS = [
    "dataset_id", "video_id", "source_path", "prepared_video_path", "split",
    "scene_id", "label_type", "has_anomaly", "duration_sec", "fps",
    "width", "height", "num_frames", "notes",
]

def merge(inputs, output):
    merged_rows = []
    headers = []
    
    for ip in inputs:
        if not os.path.isfile(ip):
            print(f"Error: Input manifest not found: {ip}")
            return False
        print(f"Reading {ip}...")
        with open(ip, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if header is None:
                print(f"Error: Input manifest is empty: {ip}")
                return False
            if header != REQUIRED_COLUMNS:
                print(f"Error: Schema mismatch in {ip}. Expected {REQUIRED_COLUMNS}, got {header}")
                return False
            if not headers:
                headers = header
            elif header != headers:
                print(f"Error: Schema mismatch in {ip}")
                return False
            for row in reader:
                merged_rows.append(row)
                
    if not headers:
        print("Error: No manifests merged.")
        return False
        
    if not merged_rows:
        print("Error: Input manifests contain no data rows.")
        return False
    ids = [row["video_id"] for row in merged_rows]
    paths = [os.path.normcase(os.path.normpath(row["prepared_video_path"])) for row in merged_rows]
    if len(ids) != len(set(ids)):
        print("Error: Duplicate video_id values across input manifests.")
        return False
    if len(paths) != len(set(paths)):
        print("Error: Duplicate prepared_video_path values across input manifests.")
        return False

    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(merged_rows)
        
    print(f"Merged manifest saved to {output} (Total rows: {len(merged_rows)})")
    return True

def main():
    parser = argparse.ArgumentParser(description="Merge multiple dataset manifests")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV files")
    parser.add_argument("--out", type=str, required=True, help="Output merged CSV file")
    args = parser.parse_args()
    
    return 0 if merge(args.inputs, args.out) else 1

if __name__ == "__main__":
    raise SystemExit(main())
