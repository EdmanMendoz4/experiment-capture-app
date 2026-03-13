import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from dataclasses import dataclass
from scipy.signal import hilbert, butter, filtfilt
import scipy.fftpack

@dataclass
class AppConfig:
    video_in: str = r"C:\Users\Sell\Desktop\Experimentos\NewTests\Test0\experiment_video.avi"
    data_csv: str = r"C:\Users\Sell\Desktop\Experimentos\NewTests\Test0\experiment_data.csv"
    timestamps_csv: str = r"C:\Users\Sell\Desktop\Experimentos\NewTests\Test0\video_timestamps.csv"
    sample_rate: float = 10000.0  
    window_size_sec: float = 0.1 
    expected_frequency: float = 1000.0 
    filter_bandwidth: float = 20.0 
    voltage_min: float = -2.5
    voltage_max: float = 2.5
    phase_window : float = 1

class ToggleAnalyzerGUI:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.is_macro_view = False
        
        self.load_data()
        self.setup_ui()

    def load_data(self):
        print("Loading, filtering, and pre-calculating data...")
        df_signal = pd.read_csv(self.cfg.data_csv)
        raw_v_data = df_signal.iloc[:, 0].to_numpy()
        
        # --- NEW: Zero-Phase Bandpass Filter ---
        # 1. Center the raw signal first
        centered_raw_signal = raw_v_data - np.mean(raw_v_data)
        
        # 2. Design the Butterworth bandpass filter
        nyq = 0.5 * self.cfg.sample_rate
        low = (self.cfg.expected_frequency - self.cfg.filter_bandwidth / 2.0) / nyq
        high = (self.cfg.expected_frequency + self.cfg.filter_bandwidth / 2.0) / nyq
        b, a = butter(4, [low, high], btype='band')
        
        # 3. Apply forward-backward filtering (Zero phase shift)
        self.v_data = filtfilt(b, a, centered_raw_signal)
        
        df_video_time = pd.read_csv(self.cfg.timestamps_csv)
        self.v_times = df_video_time['Timestamp'].to_numpy()
        self.v_times = self.v_times - self.v_times[0]
        
        # Time array for the whole signal
        self.t_array = np.arange(len(self.v_data)) / self.cfg.sample_rate

        # --- Hilbert Transform on the FILTERED data ---
        analytic_signal = hilbert(self.v_data)
        
        # Flattened Phase
        raw_phase = np.unwrap(np.angle(analytic_signal))
        expected_phase = 2.0 * np.pi * self.cfg.expected_frequency * self.t_array
        self.phase_data = raw_phase - expected_phase
        self.phase_baseline = np.median(self.phase_data)

        self.cap = cv2.VideoCapture(self.cfg.video_in)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        max_vid = self.v_times[-1] if len(self.v_times) > 0 else self.total_frames / self.fps
        self.max_time = min(max_vid, self.t_array[-1])

    def setup_ui(self):
        self.fig = plt.figure(figsize=(12, 10))
        
        self.ax_video = self.fig.add_axes([0.1, 0.55, 0.8, 0.4])
        self.ax_signal = self.fig.add_axes([0.1, 0.32, 0.8, 0.18])
        self.ax_phase = self.fig.add_axes([0.1, 0.10, 0.8, 0.18])
        
        self.ax_video.axis('off')
        self.img_display = self.ax_video.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Plot entire dataset once
        self.line_signal, = self.ax_signal.plot(self.t_array, self.v_data, color='blue', linewidth=1)
        self.ax_signal.set_ylabel("Filtered Voltage (V)")
        self.ax_signal.grid(True, linestyle='--', alpha=0.6)
        
        self.line_phase, = self.ax_phase.plot(self.t_array, self.phase_data, color='red', linewidth=1)
        self.ax_phase.set_ylabel("Phase Shift (rad)")
        self.ax_phase.set_xlabel("Time (s)")
        self.ax_phase.grid(True, linestyle='--', alpha=0.6)
        
        # Indicator Lines for Macro View (Hidden by default)
        self.vline_signal = self.ax_signal.axvline(x=0, color='black', linewidth=2, linestyle='--', visible=False)
        self.vline_phase = self.ax_phase.axvline(x=0, color='black', linewidth=2, linestyle='--', visible=False)
        
        # Widgets
        ax_slider = self.fig.add_axes([0.1, 0.02, 0.8, 0.03])
        self.slider = Slider(ax_slider, 'Time (s)', 0.0, self.max_time - self.cfg.window_size_sec, valinit=0.0)
        self.slider.on_changed(self.update_frame)

        ax_button = self.fig.add_axes([0.02, 0.9, 0.08, 0.05])
        self.btn_toggle = Button(ax_button, 'Toggle View')
        self.btn_toggle.on_clicked(self.toggle_view)

        # Keyboard controls
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_frame(0.0)
        plt.show()

    def on_key_press(self, event):
        step = 1.0 / self.fps 
        current_time = self.slider.val
        
        if event.key == 'right':
            new_time = min(current_time + step, self.max_time - self.cfg.window_size_sec)
            self.slider.set_val(new_time)
        elif event.key == 'left':
            new_time = max(current_time - step, 0.0)
            self.slider.set_val(new_time)

    def toggle_view(self, event):
        self.is_macro_view = not self.is_macro_view
        self.update_frame(self.slider.val)

    def update_frame(self, val):
        t_start = self.slider.val
        t_end = t_start + self.cfg.window_size_sec
        
        # Video
        frame_idx = np.searchsorted(self.v_times, t_start)
        frame_idx = min(frame_idx, self.total_frames - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.ax_video.set_title(f"Frame: {frame_idx} | T: {t_start:.3f}s")

        # Graphs
        if self.is_macro_view:
            self.vline_signal.set_visible(True)
            self.vline_phase.set_visible(True)
            self.vline_signal.set_xdata([t_start, t_start])
            self.vline_phase.set_xdata([t_start, t_start])
            
            self.ax_signal.set_xlim(0, self.max_time)
            self.ax_phase.set_xlim(0, self.max_time)
            
            # Autoscale Y-axes for macro view
            self.ax_signal.set_ylim(np.min(self.v_data), np.max(self.v_data))
            self.ax_phase.set_ylim(np.min(self.phase_data), np.max(self.phase_data))
        else:
            self.vline_signal.set_visible(False)
            self.vline_phase.set_visible(False)
            
            self.ax_signal.set_xlim(t_start, t_end)
            self.ax_phase.set_xlim(t_start, t_end)
            
            # Fixed Y-axis limits for the signal graph
            self.ax_signal.set_ylim(self.cfg.voltage_min, self.cfg.voltage_max)
            
            # Autoscale phase for the window
            y_min = self.phase_baseline - self.cfg.phase_window
            y_max = self.phase_baseline + self.cfg.phase_window
            self.ax_phase.set_ylim(y_min, y_max)

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = ToggleAnalyzerGUI(AppConfig())