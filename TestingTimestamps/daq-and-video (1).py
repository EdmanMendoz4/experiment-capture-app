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
import pyautogui

# Force a hardware reset to clear "Ghost" tasks
print("Resetting Device...")
nidaqmx.system.Device("Dev1").reset_device()
print("Device Reset Complete.")

# --- CONFIGURATION ---
DAQ_CHANNEL     = "Dev1/ai1"
DAQ_RATE        = 10000           # samples/sec
DAQ_CHUNK       = 1000            # samples per read (~0.1s per chunk)
VIDEO_SRC       = 0
VIDEO_FILENAME  = "experiment_video.avi"
DAQ_FILENAME    = "experiment_data.csv"
LASER_TOGGLE_S  = 1.0             # seconds between each laser toggle
LASER_X         = 10488           # pyautogui click coords for laser GUI
LASER_Y         = 1308
TERMINATION_TIME = 30.0           # seconds after start to auto-stop

# --- HELPER CLASS: BACKGROUND VIDEO RECORDER ---
class VideoRecorder:
    def __init__(self, source=0, filename="video.avi"):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0

        self.writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps, (w, h)
        )
        self.running         = False
        self.start_time      = None
        self.frame_timestamps = []

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._record, daemon=True)
        self.thread.start()
        print("[Video] Background recording started...")

    def _record(self):
        self.start_time = time.time()
        frame_idx = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Recording', frame)
                cv2.waitKey(1)
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
        with open("video_timestamps.csv", "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["Frame_Index", "Timestamp"])
            w.writerows(self.frame_timestamps)


# --- LASER TOGGLER: fires on a background thread ---
class LaserToggler:
    """
    Clicks the laser GUI button every LASER_TOGGLE_S seconds starting at
    t_start + LASER_TOGGLE_S, and records each toggle event with its
    exact timestamp and state (ON / OFF).
    
    Also exposes `laser_on` so the DAQ loop can decide whether to write
    real data or zeros.
    """
    def __init__(self, t_start, interval=LASER_TOGGLE_S):
        self.t_start    = t_start
        self.interval   = interval
        self.laser_on   = False          # starts OFF
        self.running    = False
        self.events     = []             # (toggle_index, timestamp, state)
        self._lock      = threading.Lock()

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        toggle_index = 0
        while self.running:
            # Sleep until the next scheduled toggle moment
            next_toggle = self.t_start + self.interval * (toggle_index + 1)
            sleep_for   = next_toggle - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

            if not self.running:
                break

            # Stop toggling if TERMINATION_TIME has been reached
            if time.time() - self.t_start >= TERMINATION_TIME:
                break

            # Click laser GUI
            pyautogui.click(LASER_X, LASER_Y)
            t_click = time.time()

            with self._lock:
                self.laser_on = not self.laser_on
                state = "ON" if self.laser_on else "OFF"
                self.events.append((toggle_index, t_click, state))

            elapsed = t_click - self.t_start
            print(f"[Laser] Toggle #{toggle_index + 1}  →  {state}  "
                  f"(t = {elapsed:.4f} s)")
            toggle_index += 1

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        print(f"[Laser] Stopped. {len(self.events)} toggles recorded.")
        with open("laser_events.csv", "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["Toggle_Index", "Timestamp", "State"])
            w.writerows(self.events)

    @property
    def is_on(self):
        with self._lock:
            return self.laser_on


# --- MAIN ---
def run_experiment():
    # Reset DAQ device
    try:
        nidaqmx.system.Device(DAQ_CHANNEL.split('/')[0]).reset_device()
    except Exception:
        pass

    cam = VideoRecorder(source=VIDEO_SRC, filename=VIDEO_FILENAME)

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            DAQ_CHANNEL,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=-10.0,
            max_val=10.0
        )
        task.timing.cfg_samp_clk_timing(
            rate=DAQ_RATE,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        reader = AnalogSingleChannelReader(task.in_stream)
        buffer = np.zeros(DAQ_CHUNK, dtype=np.float64)
        zeros  = np.zeros((DAQ_CHUNK, 1))          # pre-built zero column

        print("Ready. Press ENTER to start recording...")
        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                break

        # ---- SIMULTANEOUS START ----
        t_start = time.time()
        cam.start()
        task.start()

        laser = LaserToggler(t_start=t_start, interval=LASER_TOGGLE_S)
        laser.start()

        print("Recording...  Laser toggles every 1 s.  Press ESC to stop.")

        with open(DAQ_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Voltage'])

            try:
                while True:
                    reader.read_many_sample(
                        buffer,
                        number_of_samples_per_channel=DAQ_CHUNK
                    )

                    if laser.is_on:
                        # Laser ON  → write zeros (suppress real signal)
                        writer.writerows(zeros)
                    else:
                        # Laser OFF → write real DAQ data
                        writer.writerows(buffer.reshape(-1, 1))

                    if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                        break

                    if time.time() - t_start >= TERMINATION_TIME:
                        print(f"[Main] TERMINATION_TIME ({TERMINATION_TIME} s) reached. Stopping.")
                        break

            finally:
                laser.stop()
                cam.stop()
                cv2.destroyAllWindows()
                print("Experiment finished.")
                print("Output files:")
                print(f"  {DAQ_FILENAME}          – voltage trace (zeros during laser ON)")
                print("  laser_events.csv       – exact toggle timestamps & states")
                print("  video_timestamps.csv   – per-frame timestamps for sync")


if __name__ == "__main__":
    run_experiment()
