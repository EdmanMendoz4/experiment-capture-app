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
import jds6600

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
LASER_TOGGLE_S   = 2.0             # seconds between each laser toggle
LASER_X          = 10488           # pyautogui click coords for laser GUI
LASER_Y          = 1308
TERMINATION_TIME = 30.0            # seconds after start to auto-stop
SG_PORT          = "COM7"          # serial port for JDS6600 signal generator

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


# --- LASER & SIGNAL GENERATOR TOGGLER: fires on a background thread ---
class LaserToggler:
    """
    Every LASER_TOGGLE_S seconds:
      - Clicks the laser GUI (pyautogui)
      - Toggles the JDS6600 signal generator channel 1 (ON when laser ON, OFF when laser OFF)
    Records exact timestamps for both events so DAQ latency can be measured.
    """
    def __init__(self, t_start, fg, interval=LASER_TOGGLE_S):
        self.t_start  = t_start
        self.fg       = fg               # jds6600 instance
        self.interval = interval
        self.laser_on = False
        self.running  = False
        self.events   = []               # (toggle_index, t_laser_click, t_sg_toggle, state)
        self._lock    = threading.Lock()

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

            # 1. Click laser GUI and record timestamp
            pyautogui.click(LASER_X, LASER_Y)
            t_laser = time.time()

            # 2. Toggle signal generator channel 1 and record timestamp
            with self._lock:
                self.laser_on = not self.laser_on
                new_state = self.laser_on

            if new_state:
                self.fg.set_channels(channel1=True, channel2=False)
            else:
                self.fg.set_channels(channel1=False, channel2=False)
            t_sg = time.time()

            state = "ON" if new_state else "OFF"
            self.events.append((toggle_index, t_laser, t_sg, state))

            elapsed = t_laser - self.t_start
            print(f"[Toggler] #{toggle_index + 1}  →  {state}  "
                  f"(laser t={elapsed:.4f} s  |  sg delay={1000*(t_sg - t_laser):.2f} ms)")
            toggle_index += 1

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        # Make sure signal generator is left OFF
        try:
            self.fg.set_channels(channel1=False, channel2=False)
        except Exception:
            pass
        print(f"[Toggler] Stopped. {len(self.events)} toggles recorded.")
        with open("toggle_events.csv", "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["Toggle_Index", "T_Laser_Click", "T_SG_Toggle", "State"])
            w.writerows(self.events)


# --- MAIN ---
def run_experiment():
    # Reset DAQ device
    try:
        nidaqmx.system.Device(DAQ_CHANNEL.split('/')[0]).reset_device()
    except Exception:
        pass

    cam = VideoRecorder(source=VIDEO_SRC, filename=VIDEO_FILENAME)

    # Connect to JDS6600 signal generator
    fg = jds6600.JDS6600(port=SG_PORT)
    fg.connect()
    fg.set_channels(channel1=False, channel2=False)  # start with signal OFF
    print(f"[SG] JDS6600 connected on {SG_PORT}, channel 1 set to OFF.")

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

        print("Ready. Press ENTER to start recording...")
        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                break

        # ---- SIMULTANEOUS START ----
        t_start = time.time()
        cam.start()
        task.start()

        laser = LaserToggler(t_start=t_start, fg=fg, interval=LASER_TOGGLE_S)
        laser.start()

        print("Recording...  Signal toggles every 1 s.  Press ESC to stop.")

        with open(DAQ_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Voltage'])

            try:
                while True:
                    reader.read_many_sample(
                        buffer,
                        number_of_samples_per_channel=DAQ_CHUNK
                    )

                    # Always write real DAQ data — signal generator handles the toggle
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
                print(f"  {DAQ_FILENAME}          – full voltage trace")
                print("  toggle_events.csv      – laser click & SG toggle timestamps per event")
                print("  video_timestamps.csv   – per-frame timestamps for sync")


if __name__ == "__main__":
    run_experiment()