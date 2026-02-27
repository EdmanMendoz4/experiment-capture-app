from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

@dataclass 
class SignalConfig:
    sample_rate: int = 10000            # Samples per Second 
    voltage_min: float = 0.0            # Volts
    voltage_max: float = 5.0            # Volts
    signal_freq: float = 1000           # Freq in Hz 
    window_size_wave_cycles: Optional[int] = None   # Number of wave periods to show
    window_size_sec: Optional[float] = None         # Seconds of data to show
    
    def __post_init__(self):
        if self.window_size_wave_cycles is not None:
            # If wave cycles is provided, calculate window size in seconds
            self.window_size_sec = self.window_size_wave_cycles / self.signal_freq
        if self.window_size_sec is None and self.window_size_wave_cycles is None:
            # Default to 20ms window if neither is provided
            self.window_size_sec = 0.02
            self.window_size_wave_cycles = int(self.signal_freq * self.window_size_sec)
        if self.window_size_sec > 1.0:
            # Assuming window_size_sec was inputted in miliseconds instead of seconds
            self.window_size_sec = self.window_size_sec / 1000.0
        if self.voltage_min >= self.voltage_max:
            raise ValueError("Minimum Voltage must be smaller than Max Voltage")
    
@dataclass 
class DisplayConfig:
    voltage_color: str = '#FFFF00'      # Color of the voltage line
    voltage_linewidth: int = 2            # Thickness of the voltage line
    voltage_transp: float = 0.8           # Transparency of the voltage line
    bg_transp: float = 0.5                # Transparency of the background box
    hud_height_ratio: float = 0.25        # Height of the HUD as a ratio of total frame height
    position: str = 'bottom'              # Position of the HUD: 'top' or 'bottom'
    dpi: int = 100                       # Dots per inch for the Matplotlib figure 

# --- Helper Function ---    
def _render_plot_to_image(fig, ax, t_start, t_end, subset_t, subset_v, current_time, signal_cfg, display_cfg):
    """Handles all Matplotlib drawing and returns a BGR numpy image for OpenCV."""
    ax.clear()
    ax.set_facecolor((0, 0, 0, display_cfg.bg_transp)) 
    
    if len(subset_v) > 1:
        ax.plot(subset_t, subset_v, color=display_cfg.voltage_color, linewidth=display_cfg.voltage_linewidth, alpha=display_cfg.voltage_transp)
    
    ax.set_ylim(signal_cfg.voltage_min, signal_cfg.voltage_max)
    ax.set_xlim(t_start, t_end) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, color='white') 
    ax.set_title(f"T: {current_time:.3f}s", color='white', loc='right', fontsize=10)
    
    fig.tight_layout()
    fig.canvas.draw()
    
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

# --- Main Function ---

def create_overlay(
    video_in: str,
    video_out: str,
    data_csv: str,
    timestamps_csv: Optional[str] = None,
    signal_cfg: SignalConfig = SignalConfig(),
    display_cfg: DisplayConfig = DisplayConfig()
):
    """Overlays a scrolling oscilloscope-style plot of voltage data onto a video.

    Args:
        video_in (str): Path to input video file
        video_out (str): Path to save output video with overlay
        data_csv (str): CSV file with 'Voltage' header
        timestamps_csv (Optional[str]): CSV file with frame timestamps for video.
        signal_cfg (SignalConfig, optional): Signal configuration object. Optional
        display_cfg (DisplayConfig, optional): Display configuration object. Optional
    """
    # Load signal data
    print (f'Loading voltage data from {data_csv}...')
    try:
        df_signal = pd.read_csv(data_csv)
        voltage_data = df_signal['Voltage'].values.to_numpy()
    except Exception as e:
        print(f"Error loading voltage data: {e}")
        return
    
    # Load Timestamps safely
    video_timestamps = None
    if timestamps_csv:
        try:
            df_video_time = pd.read_csv(timestamps_csv)
            video_timestamps = df_video_time['Timestamp'].to_numpy()
            video_timestamps -= video_timestamps[0] # Normalize
        except Exception as e:
            print(f"Warning: Could not load timestamps ({e}). Using FPS-based timing.")

    # Setup Video
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_in}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    # Setup Figure
    fig_w, fig_h = width / display_cfg.dpi, (height * display_cfg.hud_height_ratio) / display_cfg.dpi 
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=display_cfg.dpi)
    fig.patch.set_alpha(0.0) 
    
    print(f"Processing {total_frames} frames...")
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # Determine time
        if video_timestamps is not None and i < len(video_timestamps):
            current_time = video_timestamps[i]
        else:
            current_time = i / fps
            
        t_start = current_time - signal_cfg.window_size_sec
        idx_end = int(current_time * signal_cfg.sample_rate)
        idx_start = int(t_start * signal_cfg.sample_rate)
        
        if idx_end < 0: continue
            
        # Data Slicing
        if idx_start < 0:
            subset_v = voltage_data[0:idx_end]
            subset_t = np.linspace(0, current_time, len(subset_v))
        else:
            safe_end = min(idx_end, len(voltage_data))
            subset_v = voltage_data[idx_start:safe_end]
            subset_t = np.linspace(t_start, current_time, len(subset_v))
        
        # Render Plot Image
        img_plot = _render_plot_to_image(
            fig, ax, t_start, current_time, subset_t, subset_v, current_time, signal_cfg, display_cfg
        )
        
        # Overlay Logic
        h_plot, w_plot, _ = img_plot.shape
        if w_plot != width:
             img_plot = cv2.resize(img_plot, (width, int(h_plot * (width/w_plot))))
             h_plot = img_plot.shape[0]

        if display_cfg.position == 'bottom':
            roi_y = height - h_plot
        else:
            roi_y = 0 # Top

        frame[roi_y:roi_y+h_plot, 0:width] = img_plot
        out.write(frame)
        
        if i > 0 and i % 100 == 0:
            print(f"Processed {i}/{total_frames} frames ({(i/total_frames)*100:.1f}%)")

    cap.release()
    out.release()
    plt.close(fig) # Crucial to prevent memory leaks when calling in a loop
    print(f"Done! Saved to {video_out}")