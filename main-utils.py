import nidaqmx
import nidaqmx.system
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogSingleChannelReader
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from pynput import keyboard
import os
import re
from utils import KeyboardHandler, get_next_cell_id
import cv2
import threading
import time


# --- CONFIGURATION ---
DAQ_CHANNEL = "Dev1/ai1"
DAQ_RATE = 10000
DAQ_CHUNK = 1000

# --- Experiment Configuration Data Class ---
@dataclass
class ExperimentConfig:
    cam_src : int                   # Camera Source Index
    
    daq_rate: int = 10000           # DAQ SPS (Samples Per Second)
    daq_device : str = "Dev1"       # DAQ Device Name (usually "Dev1" in NIDAQmx)
    daq_channel : str = "ai1"       # DAQ Channel Name (e.g., "ai1", Differential)
    daq_chunk : int = 1000          # DAQ Buffer Size for Each Read
    
    cell_id : Optional[int] = None      # Cell ID
    excit_amplitude : int               # Waveform Amplitude in V
    excit_frequency : int               # Waveform Frequency in Hz
    
    project_path : str = "C:\\Users\\Sell\\Desktop\\Experimentos\\"  # Base Path  
    video_filename : str = "experiment_video.avi"   # Output Video Filename
    signal_filename : str = "experiment_data.csv"   # Output Signal CSV Filename
    
    def __post_init__(self):
        # Complete Data Channel for NIDAQmx
        self.daq_full_channel = f"{self.daq_device}/{self.daq_channel}"
        # Create full paths for outputs
        timestamp = datetime.now().strftime("%Y%m%d_")
        today_path = os.path.join(self.project_path, timestamp)
        if self.cell_id is None:
            self.cell_id =  get_next_cell_id(today_path)
        self.experiment_path = os.path.join(today_path, f"Cell_{self.cell_id}")
        self.video_path = os.path.join(self.experiment_path, self.video_filename)
        self.signal_path = os.path.join(self.experiment_path, self.signal_filename)  


# --- Open CV Video Recorder Class ---
class VideoRecorder:
    def __init__(self, config : ExperimentConfig):
        self.cap = cv2.VideoCapture(config.cam_src)
        # Force MJPG to allow higher FPS on some cameras
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual properties
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0 # Standard webcam FPS
        
        self.writer = cv2.VideoWriter(
            config.video_path, 
            cv2.VideoWriter_fourcc(*'XVID'), 
            fps, (w, h)
        )
        self.running = False
        self.start_time = None
        self.frame_timestamps = [] # Store (frame_idx, timestamp)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print(f"[Video] Background recording started...")

    def _record(self):
        self.start_time = time.time()
        frame_idx = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Show the video frame in a window
                cv2.imshow('Recording', frame)
                cv2.waitKey(1) # Needed to update the window
                # Save timestamp immediately after read
                t_now = time.time()
                self.writer.write(frame)
                self.frame_timestamps.append((frame_idx, t_now))
                frame_idx += 1
            else:
                break
        
        self.cap.release()
        self.writer.release()
    
    def stop(self):
        self.running = False
        self.thread.join()
        print(f"[Video] Stopped. Saved {len(self.frame_timestamps)} frames.")
        
        # Save frame timestamps to CSV for synchronization
        with open("video_timestamps.csv", "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["Frame_Index", "Timestamp"])
            w.writerows(self.frame_timestamps)

# --- Run Video and Signal Recording ---
def run_experiment(exp: ExperimentConfig):
    # 1. Reset DAQ Device
    try:
        device = nidaqmx.system.Device(exp.daq_device)
        device.reset_device()
    except:
        pass

    # 2. Setup Video
    cam = VideoRecorder(config=exp)

    # 3. Setup DAQ Task
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            exp.daq_full_channel, terminal_config=TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0
        )
        task.timing.cfg_samp_clk_timing(
            rate=exp.daq_rate, sample_mode=AcquisitionType.CONTINUOUS
        )
        
        reader = AnalogSingleChannelReader(task.in_stream)
        buffer = np.zeros(exp.daq_chunk, dtype=np.float64)
        
        print("Ready. Press ENTER to start recording...")
        
        kb_handler = KeyboardHandler()
        listener = keyboard.Listener(on_press=kb_handler.on_press_start)
        listener.start()
        listener.join()  # Wait for ENTER to be pressed
    
        # --- START RECORDING ---
        cam.start()      # Start Video Thread
        task.start()     # Start DAQ Hardware
        
        print("Recording... Press ESC to stop.")
        
        listener = keyboard.Listener(on_press=kb_handler.on_press_stop)
        listener.start()
        
        with open(DAQ_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Voltage'])
            
            try:
                while not kb_handler.stop_pressed:
                    # Read from DAQ buffer into memory
                    reader.read_many_sample(
                        buffer,
                        number_of_samples_per_channel=exp.daq_chunk
                    )

                    # Write to CSV
                    # reshaping to (-1, 1) creates a column vector for the CSV writer
                    writer.writerows(buffer.reshape(-1, 1))
            finally:
                listener.stop()  #  Stop keyboard listener
                cam.stop() # Stop Video Thread
                print("Experiment Finished.")