# scripts/datasets/frames_to_video.py
import argparse
import os
import sys
import cv2

def convert_frames(input_dir, output_path, fps):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
        
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files:
        print(f"Error: No image files found in {input_dir}")
        return False
        
    print(f"Found {len(files)} frames in {input_dir}")
    first_frame_path = os.path.join(input_dir, files[0])
    img = cv2.imread(first_frame_path)
    height, width, layers = img.shape
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for idx, f in enumerate(files):
        img_path = os.path.join(input_dir, f)
        frame = cv2.imread(img_path)
        video.write(frame)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(files)} frames...")
            
    video.release()
    print(f"Video saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert frame sequence to video file")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing frames")
    parser.add_argument("--out", type=str, required=True, help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=25, help="Video framerate")
    args = parser.parse_args()
    
    success = convert_frames(args.input, args.out, args.fps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
