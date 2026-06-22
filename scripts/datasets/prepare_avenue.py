# scripts/datasets/prepare_avenue.py
import argparse
import os
import sys
import csv
import cv2
import numpy as np

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv')

def generate_dummy_video(path, duration=5, fps=25, width=640, height=480):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    num_frames = int(duration * fps)
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Create visual variation
        cv2.rectangle(img, (20 + i*2 % 100, 50), (120 + i*2 % 100, 150), (0, 255, 0), -1)
        cv2.putText(img, f"Avenue Mock Video {os.path.basename(path)} Frame {i}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(img)
    out.release()

def process_raw_dataset(root, out_dir, fps):
    # If the root folder exists, scan it. Let's find video files.
    # Typical Avenue folder structure has a 'testing_videos' directory.
    found_videos = []
    for r, d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VIDEO_EXTENSIONS):
                found_videos.append(os.path.join(r, f))
    return sorted(found_videos)

def make_video_id(root, source):
    relative = os.path.relpath(source, root)
    stem = os.path.splitext(relative)[0]
    safe = "_".join(part for part in stem.replace("\\", "/").split("/") if part)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe)
    return f"avenue_{safe.lower()}"

def main():
    parser = argparse.ArgumentParser(description="Prepare CUHK Avenue dataset")
    parser.add_argument("--root", type=str, default="/datasets/CUHK_Avenue")
    parser.add_argument("--out", type=str, default="data/imports/avenue_full")
    parser.add_argument("--manifest", type=str, default="data/manifests/avenue_full.csv")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--mock", action="store_true", help="Explicitly generate synthetic test videos")
    parser.add_argument("--video-mode", choices=("reference", "transcode"), default="reference",
                        help="Reference real videos in place (default) or explicitly transcode them to --out")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    
    os.makedirs(args.out, exist_ok=True)
    manifest_parent = os.path.dirname(args.manifest)
    if manifest_parent:
        os.makedirs(manifest_parent, exist_ok=True)
    
    videos = []
    if args.mock:
        is_mock = True
    elif os.path.isdir(args.root):
        is_mock = False
        print(f"Scanning raw CUHK Avenue from: {args.root}")
        raw_videos = process_raw_dataset(args.root, args.out, args.fps)
        if raw_videos:
            seen_ids = set()
            for src in raw_videos:
                video_id = make_video_id(args.root, src)
                if video_id in seen_ids:
                    raise RuntimeError(f"Duplicate generated video_id: {video_id}")
                seen_ids.add(video_id)
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    raise RuntimeError(f"Unreadable source video: {src}")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if args.video_mode == "reference":
                    dest = src
                    print(f"Referencing real video in place: {src}")
                else:
                    dest = os.path.join(args.out, f"{video_id}.mp4")
                    print(f"Transcoding {src} -> {dest}")
                    output_fps = float(args.fps)
                    out_writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
                    num_frames = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out_writer.write(frame)
                        num_frames += 1
                    out_writer.release()
                cap.release()
                duration = num_frames / output_fps
                videos.append({
                    "video_id": video_id,
                    "source_path": src,
                    "prepared_video_path": dest,
                    "duration_sec": round(duration, 2),
                    "fps": round(output_fps, 6),
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "split": "test" if "test" in src.lower() else "train",
                    "has_anomaly": "yes" if "abnormal" in src.lower() or "anomaly" in src.lower() else "no"
                })
        else:
            print("Error: no raw videos found. Use --mock only for explicit synthetic test data.", file=sys.stderr)
            return 2
    else:
        print(f"Error: CUHK Avenue root directory '{args.root}' not found. Use --mock only for synthetic test data.", file=sys.stderr)
        return 2
        
    if is_mock:
        # Generate full scale mock videos (16 train, 21 test)
        mock_definitions = []
        # 16 train videos
        for i in range(1, 17):
            name = f"train_{i:02d}.mp4"
            mock_definitions.append({
                "name": name,
                "duration": 2.0,
                "split": "train",
                "has_anomaly": "no",
                "scene": "cuhk_avenue_1"
            })
        # 21 test videos
        for i in range(1, 22):
            name = f"test_{i:02d}.mp4"
            # Mark some test videos abnormal (e.g. even ones)
            has_anomaly = "yes" if i % 2 == 0 else "no"
            mock_definitions.append({
                "name": name,
                "duration": 2.0,
                "split": "test",
                "has_anomaly": has_anomaly,
                "scene": "cuhk_avenue_1"
            })
            
        for item in mock_definitions:
            video_id = f"avenue_{os.path.splitext(item['name'])[0]}"
            dest = os.path.join(args.out, f"{video_id}.mp4")
            print(f"Generating synthetic video: {dest}")
            generate_dummy_video(dest, duration=item["duration"], fps=args.fps)
            videos.append({
                "video_id": video_id,
                "source_path": f"mock://{item['name']}",
                "prepared_video_path": dest,
                "duration_sec": float(item["duration"]),
                "fps": args.fps,
                "width": 640,
                "height": 480,
                "num_frames": int(item["duration"] * args.fps),
                "split": item["split"],
                "has_anomaly": item["has_anomaly"],
                "scene_id": item["scene"],
                "label_type": "temporal",
                "notes": "Synthetic mock video for testing"
            })
            
    # Write manifest
    headers = [
        "dataset_id", "video_id", "source_path", "prepared_video_path", "split",
        "scene_id", "label_type", "has_anomaly", "duration_sec", "fps",
        "width", "height", "num_frames", "notes"
    ]
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for v in videos:
            writer.writerow([
                "avenue_full",
                v["video_id"],
                v["source_path"],
                v["prepared_video_path"],
                v["split"],
                v.get("scene_id", "scene_1"),
                v.get("label_type", "none"),
                v["has_anomaly"],
                v["duration_sec"],
                v["fps"],
                v["width"],
                v["height"],
                v["num_frames"],
                v.get("notes", "")
            ])
            
    print(f"CUHK Avenue preparation complete. Manifest saved to {args.manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
