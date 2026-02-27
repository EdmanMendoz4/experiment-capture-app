import matplotlib
matplotlib.use('Agg') # Force non-interactive backend (Faster)
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
VIDEO_IN = "experiment_video.avi"
VIDEO_OUT = "experiment_overlay_final.avi"
DATA_CSV = "experiment_data.csv"      # Voltage only (Single column)
TIMESTAMPS_CSV = "video_timestamps.csv" # The file from the video recorder

# Signal Parameters
SAMPLE_RATE = 10000.0   # 10 kHz
WINDOW_SIZE_SEC = 0.2  # 20ms window (Shows 20 cycles of 1kHz wave)
VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 5.0

class SignalParameters:
    def __init__(self, sample_rate=10000.0, window_size_sec=0.02, voltage_min=0.0, voltage_max=5.0):
        self.sample_rate = sample_rate
        self.window_size_sec = window_size_sec
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

def create_overlay():
    print(f"Loading voltage data from {DATA_CSV}...")
    
    # 1. Load Voltage Data (No Header or Header? Adjust as needed)
    # If your file has a header like 'Voltage', keep header=0. 
    # If it's just raw numbers, use header=None.
    try:
        df_signal = pd.read_csv(DATA_CSV)
        # Convert first column to numpy array
        voltage_data = df_signal.iloc[:, 0].to_numpy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(voltage_data)} samples.")

    # 2. Load Video Timestamps (To sync strictly to the video's real time)
    try:
        df_video_time = pd.read_csv(TIMESTAMPS_CSV)
        video_timestamps = df_video_time['Timestamp'].to_numpy()
        # Normalize video start time to 0.0
        start_t = video_timestamps[0]
        video_timestamps = video_timestamps - start_t
    except:
        print("Warning: No video timestamps found. Falling back to FPS-based timing.")
        video_timestamps = None

    # 3. Setup Video Input/Output
    cap = cv2.VideoCapture(VIDEO_IN)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))
    
    # 4. Setup Matplotlib Figure (The "Scope")
    dpi = 100
    fig_w = width / dpi
    fig_h = (height / 3.5) / dpi 
    
    # Fully transparent figure
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0) 
    
    print(f"Processing {total_frames} frames...")
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # --- SYNC LOGIC (The Core) ---
        # Determine current time in the experiment
        if video_timestamps is not None and i < len(video_timestamps):
            current_time = video_timestamps[i]
        else:
            # Fallback: Estimate time based on frame count
            current_time = i / fps
            
        # Calculate start/end times for the window
        t_end = current_time
        t_start = t_end - WINDOW_SIZE_SEC
        
        # Convert Time -> Sample Indices
        # Index = Time * Sample_Rate
        idx_end = int(t_end * SAMPLE_RATE)
        idx_start = int(t_start * SAMPLE_RATE)
        
        # Handle "Pre-trigger" (Negative time before recording started)
        if idx_end < 0:
            continue # Skip frames before data starts
            
        # Slice the data array
        # We must handle cases where idx_start is negative (pad with 0 or clip)
        if idx_start < 0:
            # If we are at the very start, slice from 0 to idx_end
            subset_v = voltage_data[0:idx_end]
            # Create a matching time array
            subset_t = np.linspace(0, t_end, len(subset_v))
        else:
            # Normal operation
            # Ensure we don't read past the end of the file
            safe_end = min(idx_end, len(voltage_data))
            subset_v = voltage_data[idx_start:safe_end]
            subset_t = np.linspace(t_start, t_end, len(subset_v))
        
        # --- PLOTTING ---
        ax.clear()
        
        # Styling: Semi-transparent black background
        ax.set_facecolor((0, 0, 0, 0.5)) 
        
        # Plot Signal (Yellow)
        if len(subset_v) > 1:
            ax.plot(subset_t, subset_v, color='#FFFF00', linewidth=2)
        
        # Fixed Scaling (Crucial for Oscilloscope look)
        ax.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
        ax.set_xlim(t_start, t_end) # This makes the graph "scroll"
        
        # Clean up axes (Minimalist HUD)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        
        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.3, color='white') 
        
        # Add Time Label
        ax.set_title(f"T: {current_time:.3f}s", color='white', loc='right', fontsize=10)
        
        plt.tight_layout()
        
        # --- RENDER TO IMAGE ---
        fig.canvas.draw()
        
        # Get RGBA buffer
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        
        # Convert RGBA -> BGR
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        
        # --- OVERLAY ---
        # Resize if needed (handle potential DPI mismatch)
        h_plot, w_plot, _ = img_plot.shape
        if w_plot != width:
             img_plot = cv2.resize(img_plot, (width, int(h_plot * (width/w_plot))))
             h_plot, w_plot, _ = img_plot.shape

        # Paste at bottom
        roi_y = height - h_plot
        
        # Blending (Optional: make it look slightly transparent)
        # alpha = 1.0 (Opaque)
        frame[roi_y:roi_y+h_plot, 0:width] = img_plot
        
        out.write(frame)
        
        if i % 100 == 0:
            print(f"Processed {i}/{total_frames} frames ({i/total_frames*100:.1f}%)")

    cap.release()
    out.release()
    plt.close()
    print(f"Done! Saved to {VIDEO_OUT}")

if __name__ == "__main__":
    create_overlay()