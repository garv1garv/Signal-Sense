import os
import zipfile
import re
import cv2
import numpy as np
from collections import defaultdict

def main():
    zip_path = "archive.zip"
    normal_dir = os.path.join("data", "eval_videos", "normal")
    anomaly_dir = os.path.join("data", "eval_videos", "anomaly")
    raw_dir = os.path.join("data", "raw")

    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print("Analyzing zip file structure...")
    
    # Dictionary to hold {video_name: [(frame_index, zip_filepath), ...]}
    video_groups = defaultdict(list)
    
    pattern = re.compile(r"^(.*)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

    try:
        z = zipfile.ZipFile(zip_path, 'r')
    except Exception as e:
        print(f"Failed to open zip file: {e}")
        return

    for file_info in z.infolist():
        if file_info.is_dir():
            continue
        
        filename = file_info.filename
        basename = os.path.basename(filename)
        
        match = pattern.match(basename)
        if match:
            vid_name = match.group(1)
            frame_idx = int(match.group(2))
            video_groups[vid_name].append((frame_idx, filename))
        else:
            # Fallback if naming convention is slightly different
            if basename.lower().endswith(('.png', '.jpg')):
                parts = basename.rsplit('_', 1)
                if len(parts) == 2 and parts[1].split('.')[0].isdigit():
                    vid_name = parts[0]
                    frame_idx = int(parts[1].split('.')[0])
                    video_groups[vid_name].append((frame_idx, filename))

    if not video_groups:
        print("Could not group any frames into videos. Please check the zip structure.")
        return

    print(f"Found {len(video_groups)} unique videos to construct. Starting stitching...")

    normal_count = 0
    anomaly_count = 0
    raw_count = 0

    total_videos = len(video_groups)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i, (vid_name, frames) in enumerate(video_groups.items()):
        # Sort frames numerically
        frames.sort(key=lambda x: x[0])
        
        # Determine destination path
        is_normal = "normal" in vid_name.lower()
        if is_normal:
            if normal_count < 50:
                dest_dir = normal_dir
                normal_count += 1
            else:
                dest_dir = raw_dir
                raw_count += 1
        else:
            if anomaly_count < 50:
                dest_dir = anomaly_dir
                anomaly_count += 1
            else:
                dest_dir = raw_dir
                raw_count += 1
                
        out_path = os.path.join(dest_dir, f"{vid_name}.mp4")
        
        # Create VideoWriter
        writer = None
        
        for idx, frame_path in frames:
            try:
                # Read image directly from zip into memory
                img_data = z.read(frame_path)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                    
                if writer is None:
                    h, w = img.shape[:2]
                    # Assume 30 fps, since it's just for model digestion
                    writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
                
                writer.write(img)
            except Exception as e:
                pass
                
        if writer:
            writer.release()
            
        if (i + 1) % 10 == 0 or (i + 1) == total_videos:
            print(f"Progress: {i + 1}/{total_videos} videos constructed and sorted...")

    z.close()
    
    print("\n✅ Video Stitching and Sorting Complete!")
    print(f"Normal Eval Videos: {normal_count}")
    print(f"Anomaly Eval Videos: {anomaly_count}")
    print(f"Raw Training Videos: {raw_count}")

if __name__ == "__main__":
    main()
