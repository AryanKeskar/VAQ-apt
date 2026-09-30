import cv2
import os
import numpy as np

# --- Configuration ---
VIDEO_PATH = "video/hd_epic.mp4"
OUTPUT_DIR = "video/atd_frames"

# Define the query-focused bin in seconds (e.g., focal action happens between min 3 and min 7)
QUERY_START_SEC = 180  
QUERY_END_SEC = 420    

# Define the sampling density (number of frames to extract per bin)
SPARSE_FRAME_COUNT = 2   # Frames for the non-query bins
DENSE_FRAME_COUNT = 10   # Frames for the query-focused bin

def extract_evenly_spaced_frames(cap, start_frame, end_frame, num_frames, bin_name):
    """Calculates exact frame indices and extracts them directly."""
    if num_frames == 0 or start_frame >= end_frame:
        return []

    # Calculate evenly spaced indices using linear interpolation
    indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    saved_paths = []

    for idx in indices:
        # Jump directly to the calculated frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            out_path = os.path.join(OUTPUT_DIR, f"{bin_name}_frame_{idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            print(f"Saved: {out_path} (Time: {idx / cap.get(cv2.CAP_PROP_FPS):.2f}s)")
            
    return saved_paths

def run_atd_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    print(f"Video loaded: {total_sec:.2f} seconds | {fps:.2f} FPS | {total_frames} total frames")

    # Convert seconds to frame indices
    query_start_frame = int(QUERY_START_SEC * fps)
    query_end_frame = int(QUERY_END_SEC * fps)

    # Validate boundaries
    query_start_frame = max(0, min(query_start_frame, total_frames))
    query_end_frame = max(query_start_frame, min(query_end_frame, total_frames))

    print("\n--- Extracting Pre-Query Bin (Sparse) ---")
    extract_evenly_spaced_frames(cap, 0, query_start_frame, SPARSE_FRAME_COUNT, "pre_query")

    print("\n--- Extracting Query Bin (Dense) ---")
    extract_evenly_spaced_frames(cap, query_start_frame, query_end_frame, DENSE_FRAME_COUNT, "query")

    print("\n--- Extracting Post-Query Bin (Sparse) ---")
    extract_evenly_spaced_frames(cap, query_end_frame, total_frames, SPARSE_FRAME_COUNT, "post_query")

    cap.release()
    print("\nExtraction complete.")

if __name__ == "__main__":
    run_atd_extraction()