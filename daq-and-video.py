import nidaqmx
import nidaqmx.system
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogSingleChannelReader
import numpy as np
import cv2
import threading
import time
import csv
import msvcrt
import os

# Force a hardware reset to clear "Ghost" tasks
print("Resetting Device...")
nidaqmx.system.Device("Dev1").reset_device()
print("Device Reset Complete.")

# --- CONFIGURATION ---
DAQ_CHANNEL = "Dev1/ai1"
DAQ_RATE = 10000
DAQ_CHUNK = 1000
VIDEO_SRC = 0             # 0 is usually the default webcam
VIDEO_FILENAME = "experiment_video.avi"
DAQ_FILENAME = "experiment_data.csv"

# --- HELPER CLASS: BACKGROUND VIDEO RECORDER ---
class VideoRecorder:
    def __init__(self, source=0, filename="video.avi"):
        self.cap = cv2.VideoCapture(source)
        # Force MJPG to allow higher FPS on some cameras
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual properties
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0 # Standard webcam FPS
        
        self.writer = cv2.VideoWriter(
            filename, 
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

# --- MAIN: DAQ & ORCHESTRATION ---
def run_experiment():
    # 1. Reset DAQ Device
    try:
        device = nidaqmx.system.Device(DAQ_CHANNEL.split('/')[0])
        device.reset_device()
    except:
        pass

    # 2. Setup Video
    cam = VideoRecorder(source=VIDEO_SRC, filename=VIDEO_FILENAME)

    # 3. Setup DAQ Task
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            DAQ_CHANNEL, terminal_config=TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0
        )
        task.timing.cfg_samp_clk_timing(
            rate=DAQ_RATE, sample_mode=AcquisitionType.CONTINUOUS
        )
        
        reader = AnalogSingleChannelReader(task.in_stream)
        buffer = np.zeros(DAQ_CHUNK, dtype=np.float64)
        
        print("Ready. Press ENTER to start recording...")
        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b'\r': break

        # --- START RECORDING ---
        cam.start()      # Start Video Thread
        task.start()     # Start DAQ Hardware
        
        print("Recording... Press ESC to stop.")
        
        with open(DAQ_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Voltage'])
            
            try:
                while True:
                    # Read from DAQ buffer into memory
                    reader.read_many_sample(
                        buffer,
                        number_of_samples_per_channel=DAQ_CHUNK
                    )

                    # Write to CSV
                    # reshaping to (-1, 1) creates a column vector for the CSV writer
                    writer.writerows(buffer.reshape(-1, 1))

                    if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                        break
            finally:
                cam.stop() # Stop Video Thread
                print("Experiment Finished.")

if __name__ == "__main__":
    run_experiment()