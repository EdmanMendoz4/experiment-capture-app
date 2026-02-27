import nidaqmx
import nidaqmx.system
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogSingleChannelReader
import numpy as np

# --- CONFIGURATION ---
DAQ_CHANNEL = "Dev1/ai1"
DAQ_RATE = 10000
DAQ_CHUNK = 1000
VIDEO_SRC = 0             # 0 is usually the default webcam
VIDEO_FILENAME = "experiment_video.avi"
DAQ_FILENAME = "experiment_data.csv"

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