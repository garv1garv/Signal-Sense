import os
import zipfile

def main():
    zip_path = "archive.zip"
    normal_dir = os.path.join("data", "eval_videos", "normal")
    anomaly_dir = os.path.join("data", "eval_videos", "anomaly")
    raw_dir = os.path.join("data", "raw")

    print(f"Ensuring directories exist...")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Clean up the placeholder test.mp4 if it exists so it doesn't skew our real dataset
    for d in [normal_dir, anomaly_dir]:
        test_file = os.path.join(d, "test.mp4")
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Removed placeholder {test_file}")

    print("Opening archive.zip (This may take a while since it is 11GB)...")
    
    normal_count = 0
    anomaly_count = 0
    raw_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Filter for video files
            video_files = [f for f in z.infolist() if f.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            total_videos = len(video_files)
            print(f"Found {total_videos} video files in the archive. Starting extraction & sorting...")

            for i, file_info in enumerate(video_files):
                basename = os.path.basename(file_info.filename)
                
                # Determine where this file goes
                if "normal" in basename.lower():
                    if normal_count < 50:
                        dest_path = os.path.join(normal_dir, basename)
                        normal_count += 1
                    else:
                        dest_path = os.path.join(raw_dir, basename)
                        raw_count += 1
                else:
                    if anomaly_count < 50:
                        dest_path = os.path.join(anomaly_dir, basename)
                        anomaly_count += 1
                    else:
                        dest_path = os.path.join(raw_dir, basename)
                        raw_count += 1
                
                # Extract and write directly to flatten the folder structure
                with z.open(file_info) as source, open(dest_path, "wb") as target:
                    while True:
                        chunk = source.read(1024*1024*8) # 8MB chunks
                        if not chunk:
                            break
                        target.write(chunk)
                
                # Print progress every 100 files
                if (i + 1) % 100 == 0 or (i + 1) == total_videos:
                    print(f"Progress: {i + 1}/{total_videos} videos processed...")

    except Exception as e:
        print(f"Error during extraction: {e}")
        return

    print("\n✅ Dataset Sorting Complete!")
    print(f"Normal Eval Videos: {normal_count}")
    print(f"Anomaly Eval Videos: {anomaly_count}")
    print(f"Raw Training Videos: {raw_count}")

if __name__ == "__main__":
    main()
