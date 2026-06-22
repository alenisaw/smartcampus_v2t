# scripts/datasets/import_manifest_to_library.py
import argparse
import csv
import os
import shutil

REQUIRED_COLUMNS = {"video_id", "prepared_video_path"}

def import_to_library(manifest_path, target_dir, copy_mode):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest path does not exist: {manifest_path}")
        return False
        
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            print(f"Error: Missing required manifest columns: {sorted(missing)}")
            return False
        rows = list(reader)

    if not rows:
        print("Error: Manifest contains no data rows")
        return False
    ids = [row["video_id"].strip() for row in rows]
    if any(not video_id for video_id in ids) or len(ids) != len(set(ids)):
        print("Error: video_id values must be non-empty and unique")
        return False
    missing_sources = [row["prepared_video_path"] for row in rows if not os.path.isfile(row["prepared_video_path"])]
    if missing_sources:
        print(f"Error: {len(missing_sources)} prepared video(s) are missing; nothing imported")
        return False

    os.makedirs(target_dir, exist_ok=True)
    imported_count = 0
    all_ok = True
    for row in rows:
        src = row["prepared_video_path"]
        video_id = row["video_id"]
        ext = os.path.splitext(src)[1] or ".mp4"
        dest = os.path.join(target_dir, f"{video_id}{ext}")

        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError as exc:
                print(f"Error replacing existing destination {dest}: {exc}")
                all_ok = False
                continue

        success = False
        if copy_mode == "symlink":
            try:
                os.symlink(os.path.abspath(src), dest)
                print(f"Symlinked {src} -> {dest}")
                success = True
            except PermissionError:
                print(f"Warning: Lack of privilege for symlink, falling back to copy for: {src}")
            except OSError as exc:
                print(f"Warning: Symlink failed ({exc}), falling back to copy for: {src}")

        if not success:
            try:
                os.link(os.path.abspath(src), dest)
                print(f"Hard-linked {src} -> {dest}")
                success = True
            except OSError:
                pass

        if not success:
            try:
                shutil.copy2(src, dest)
                print(f"Copied {src} -> {dest}")
                success = True
            except OSError as exc:
                print(f"Error copying {src} to {dest}: {exc}")
                all_ok = False

        if success:
            imported_count += 1
                
    print(f"Import complete. Ingested {imported_count} videos into {target_dir}")
    return all_ok and imported_count == len(rows)

def main():
    parser = argparse.ArgumentParser(description="Import manifest videos into local video library")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--videos-dir", type=str, default="data/videos")
    parser.add_argument("--copy-mode", type=str, default="symlink", choices=["symlink", "copy"])
    args = parser.parse_args()
    
    return 0 if import_to_library(args.manifest, args.videos_dir, args.copy_mode) else 1

if __name__ == "__main__":
    raise SystemExit(main())
