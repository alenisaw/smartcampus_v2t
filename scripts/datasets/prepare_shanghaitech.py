# scripts/datasets/prepare_shanghaitech.py
import argparse
import os
import sys
import csv
import cv2
import numpy as np

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv')

def make_video_id(root, source, prefix="shanghaitech"):
    relative = os.path.relpath(source, root)
    stem = os.path.splitext(relative)[0]
    safe = "_".join(part for part in stem.replace("\\", "/").split("/") if part)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe)
    return f"{prefix}_{safe.lower()}"

def generate_dummy_video(path, duration=5, fps=25, width=640, height=480):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    num_frames = int(duration * fps)
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Create visual variation
        cv2.circle(img, (320 + int(50 * np.sin(i * 0.1)), 240 + int(50 * np.cos(i * 0.1))), 40, (0, 0, 255), -1)
        cv2.putText(img, f"ShanghaiTech Mock Video {os.path.basename(path)} Frame {i}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(img)
    out.release()

def convert_frames_to_video(frame_dir, out_video_path, fps):
    # Sort frames
    files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files:
        return 0, 0, 0, 0
        
    first_frame_path = os.path.join(frame_dir, files[0])
    img = cv2.imread(first_frame_path)
    height, width, layers = img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    for f in files:
        img_path = os.path.join(frame_dir, f)
        frame = cv2.imread(img_path)
        video.write(frame)
        
    video.release()
    return len(files), width, height, len(files)/fps

def main():
    parser = argparse.ArgumentParser(description="Prepare ShanghaiTech dataset")
    parser.add_argument("--root", type=str, default="/datasets/ShanghaiTech")
    parser.add_argument("--out", type=str, default="data/imports/shanghaitech_full")
    parser.add_argument("--manifest", type=str, default="data/manifests/shanghaitech_full.csv")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--mock", action="store_true", help="Explicitly generate synthetic test videos")
    parser.add_argument("--video-mode", choices=("reference", "transcode"), default="reference",
                        help="Reference existing videos in place (default) or explicitly transcode them to --out")
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
        print(f"Scanning raw ShanghaiTech from: {args.root}")
        # Look for video files or frame directories
        # Case 1: Video files
        video_files = []
        frame_dirs = []
        for r, d, files in os.walk(args.root):
            # Check if this directory contains mostly images (frame directory)
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > 10:
                frame_dirs.append(r)
            else:
                for f in files:
                    if f.lower().endswith(VIDEO_EXTENSIONS):
                        video_files.append(os.path.join(r, f))
                        
        if video_files or frame_dirs:
            # Process video files
            seen_ids = set()
            for src in sorted(video_files):
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
                    "scene_id": "shanghaitech_scene",
                    "has_anomaly": "yes" if "abnormal" in src.lower() or "anomaly" in src.lower() else "no"
                })
                
            # Process frame directories
            for fd in sorted(frame_dirs):
                video_id = make_video_id(args.root, fd, prefix="shanghaitech_frame")
                if video_id in seen_ids:
                    raise RuntimeError(f"Duplicate generated video_id: {video_id}")
                seen_ids.add(video_id)
                dest = os.path.join(args.out, f"{video_id}.mp4")
                print(f"Converting frames in {fd} -> {dest}")
                num_frames, w, h, dur = convert_frames_to_video(fd, dest, args.fps)
                if num_frames > 0:
                    videos.append({
                        "video_id": video_id,
                        "source_path": fd,
                        "prepared_video_path": dest,
                        "duration_sec": round(dur, 2),
                        "fps": args.fps,
                        "width": w,
                        "height": h,
                        "num_frames": num_frames,
                        "split": "test" if "test" in fd.lower() else "train",
                        "scene_id": "shanghaitech_scene",
                        "has_anomaly": "yes" if "abnormal" in fd.lower() or "anomaly" in fd.lower() else "no"
                    })
        else:
            print("Error: no raw videos or frame folders found. Use --mock only for explicit synthetic test data.", file=sys.stderr)
            return 2
    else:
        print(f"Error: ShanghaiTech root directory '{args.root}' not found. Use --mock only for synthetic test data.", file=sys.stderr)
        return 2
        
    if is_mock:
        # Generate full scale mock videos (330 train, 107 test)
        mock_definitions = []
        
        # Generate 330 train videos distributed across 13 scenes
        train_count = 0
        scene_idx = 1
        video_idx = 1
        while train_count < 330:
            name = f"train_{scene_idx:02d}_{video_idx:03d}.mp4"
            mock_definitions.append({
                "name": name,
                "duration": 2.0,
                "split": "train",
                "has_anomaly": "no",
                "scene": f"shanghaitech_scene_{scene_idx}"
            })
            train_count += 1
            video_idx += 1
            if video_idx > 26:
                video_idx = 1
                scene_idx = (scene_idx % 13) + 1

        # Generate 107 test videos distributed across 13 scenes
        test_count = 0
        scene_idx = 1
        video_idx = 1
        while test_count < 107:
            name = f"test_{scene_idx:02d}_{video_idx:03d}.mp4"
            # Mark some test videos as having anomaly
            has_anomaly = "yes" if video_idx % 2 == 0 else "no"
            mock_definitions.append({
                "name": name,
                "duration": 2.0,
                "split": "test",
                "has_anomaly": has_anomaly,
                "scene": f"shanghaitech_scene_{scene_idx}"
            })
            test_count += 1
            video_idx += 1
            if video_idx > 9:
                video_idx = 1
                scene_idx = (scene_idx % 13) + 1
                
        for item in mock_definitions:
            video_id = f"shanghaitech_{os.path.splitext(item['name'])[0]}"
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
                "shanghaitech_full",
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
            
    print(f"ShanghaiTech preparation complete. Manifest saved to {args.manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
