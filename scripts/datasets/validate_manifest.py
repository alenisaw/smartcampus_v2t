# scripts/datasets/validate_manifest.py
import argparse
import os
import json
import csv
import cv2

REQUIRED_COLS = [
    "dataset_id", "video_id", "source_path", "prepared_video_path", "split",
    "scene_id", "label_type", "has_anomaly", "duration_sec", "fps",
    "width", "height", "num_frames",
]

def validate(manifest_path, out_path, allow_mock=False):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest path does not exist: {manifest_path}")
        return False
        
    validation_errors = []
    video_ids = set()
    prepared_paths = set()
    total_duration_sec = 0.0
    valid_rows = 0
    duplicate_rows = 0
    row_count = 0
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_headers = [col for col in REQUIRED_COLS if col not in (reader.fieldnames or [])]
        if missing_headers:
            validation_errors.append(f"Missing required columns: {missing_headers}")
        for idx, row in enumerate(reader):
            row_count += 1
            line_no = idx + 2
            
            # Check empty columns
            empty_cols = [col for col in REQUIRED_COLS if not row.get(col)]
            if empty_cols:
                validation_errors.append(f"Row {line_no}: Empty required columns: {empty_cols}")
                continue
                
            video_id = row["video_id"]
            prepared_path = row["prepared_video_path"]
            if row["source_path"].startswith("mock://") and not allow_mock:
                validation_errors.append(f"Row {line_no}: Mock source requires explicit --allow-mock")
            
            # Check unique video_id
            if video_id in video_ids:
                validation_errors.append(f"Row {line_no}: Duplicate video_id: '{video_id}'")
                duplicate_rows += 1
                continue
            video_ids.add(video_id)
            
            # Check duplicate prepared path
            if prepared_path in prepared_paths:
                validation_errors.append(f"Row {line_no}: Duplicate prepared path: '{prepared_path}'")
                continue
            prepared_paths.add(prepared_path)
            
            # Check prepared video file exists
            if not os.path.isfile(prepared_path):
                validation_errors.append(f"Row {line_no}: Prepared video file not found: '{prepared_path}'")
                continue
                
            # Try to read video using OpenCV to validate it is readable and check dimensions
            try:
                cap = cv2.VideoCapture(prepared_path)
                if not cap.isOpened():
                    validation_errors.append(f"Row {line_no}: Prepared video file is not readable by OpenCV: '{prepared_path}'")
                    cap.release()
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Check resolution matches manifest
                expected_w = int(row["width"])
                expected_h = int(row["height"])
                if w != expected_w or h != expected_h:
                    validation_errors.append(f"Row {line_no}: Resolution mismatch. Manifest: {row['width']}x{row['height']}, Video: {w}x{h}")
            except (ValueError, TypeError) as e:
                validation_errors.append(f"Row {line_no}: Invalid width/height values: {e}")
                continue
            except Exception as e:
                validation_errors.append(f"Row {line_no}: OpenCV capture crash: {e}")
                continue
                
            # Check positive duration and FPS
            try:
                dur = float(row["duration_sec"])
                fps = float(row["fps"])
                num_frames = int(row["num_frames"])
                if dur <= 0:
                    validation_errors.append(f"Row {line_no}: Duration is not positive: {dur}")
                if fps <= 0:
                    validation_errors.append(f"Row {line_no}: FPS is not positive: {fps}")
                if num_frames <= 0:
                    validation_errors.append(f"Row {line_no}: Frame count is not positive: {num_frames}")
                total_duration_sec += dur
            except ValueError:
                validation_errors.append(f"Row {line_no}: Non-numeric metrics values: duration={row['duration_sec']}, fps={row['fps']}, frames={row['num_frames']}")
                
            if not any(error.startswith(f"Row {line_no}:") for error in validation_errors):
                valid_rows += 1

    if row_count == 0:
        validation_errors.append("Manifest contains no data rows")
            
    report = {
        "manifest_file": manifest_path,
        "valid": len(validation_errors) == 0,
        "total_rows": row_count,
        "valid_rows": valid_rows,
        "duplicate_rows": duplicate_rows,
        "total_duration_sec": round(total_duration_sec, 2),
        "total_duration_hours": round(total_duration_sec / 3600.0, 4),
        "errors": validation_errors
    }
    
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Manifest validation complete. Report written to {out_path}")
    print(f"Status: {'VALID' if report['valid'] else 'INVALID'} ({len(validation_errors)} errors)")
    return report["valid"]

def main():
    parser = argparse.ArgumentParser(description="Validate dataset manifests")
    parser.add_argument("--manifest", type=str, default="data/manifests/v08_combined.csv")
    parser.add_argument("--out", type=str, default="data/research/v08/manifest_validation_report.json")
    parser.add_argument("--allow-mock", action="store_true", help="Allow explicit mock:// sources")
    args = parser.parse_args()
    
    return 0 if validate(args.manifest, args.out, allow_mock=args.allow_mock) else 1

if __name__ == "__main__":
    raise SystemExit(main())
