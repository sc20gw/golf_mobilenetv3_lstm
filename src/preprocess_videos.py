import os
import cv2

# Path where your MP4 videos are stored
video_dir = "../data/golfdb"   # MP4 videos folder
output_dir = "../data/golfdb_frames"  # Output folder for frames

os.makedirs(output_dir, exist_ok=True)

# Loop through all MP4 files
for video_name in os.listdir(video_dir):
    if not video_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(video_dir, video_name)
    video_id = os.path.splitext(video_name)[0]  # e.g., "video_0001"
    video_output_dir = os.path.join(output_dir, video_id)

    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_filename = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)

        cv2.imwrite(frame_path, frame)

    cap.release()
    print(f"Processed {video_name}: {frame_count} frames saved.")

print("All videos processed!")
