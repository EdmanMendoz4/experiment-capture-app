import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import rfft, rfftfreq

# --- CONFIGURATION ---
FILE_PATH = "experiment_data.csv" # Your new voltage-only file
SAMPLE_RATE = 10000.0  # Must match your recording rate exactly
VOLTAGE_RANGE = 10.0  # USB-6008 Max Range (+/- 10V)
WINDOW_SIZE = 0.05  # Initial view width (seconds)

def analyze_signal():
    print(f"--- LOADING {FILE_PATH} ---")
    
    # 1. Load Data (Assumes single column of voltage)
    # If you have a header, keep 'header=0'. If no header, use 'header=None'
    try:
        df = pd.read_csv(FILE_PATH)
        # Auto-detect column name
        col_name = df.columns[0]
        voltage = df[col_name].to_numpy()
        print(f"Loaded {len(voltage)} samples.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Reconstruct Time (Implicit Timing)
    # This creates a perfect, jitter-free time axis
    duration = len(voltage) / SAMPLE_RATE
    time_axis = np.linspace(0, duration, len(voltage), endpoint=False)
    print(f"Calculated Duration: {duration:.2f} seconds")

    # --- HEALTH CHECKS ---
    print("\n--- DIAGNOSTIC REPORT ---")
    
    # A. Clipping Check (Saturation)
    # Checks if signal hits the hardware limit (usually +/- 10V or +/- 5V)
    max_val = np.max(voltage)
    min_val = np.min(voltage)
    saturation_threshold = VOLTAGE_RANGE * 0.99 # 99% of range
    
    is_clipping = (max_val >= saturation_threshold) or (min_val <= -saturation_threshold)
    print(f"1. Range Check:   [{min_val:.2f}V to {max_val:.2f}V]")
    if is_clipping:
        print("   WARNING: SIGNAL CLIPPING DETECTED! (Input too high)")
    else:
        print("   Status: OK (Within Range)")

    # B. "Dead Sensor" Check (Flatline)
    # Checks if the standard deviation is suspiciously low
    std_dev = np.std(voltage)
    print(f"2. Activity Check: (Std Dev: {std_dev:.4f})")
    if std_dev < 0.001:
        print("   WARNING: FLATLINE DETECTED. Is the sensor plugged in?")
    else:
        print("   Status: OK (Signal is active)")

    # C. Frequency Analysis (FFT) to find Noise
    # This converts Time Domain -> Frequency Domain
    N = len(voltage)
    yf = rfft(voltage)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    
    # Get magnitude of frequencies
    magnitude = np.abs(yf)
    
    # Find the dominant frequency (excluding DC component at index 0)
    peak_idx = np.argmax(magnitude[1:]) + 1 
    peak_freq = xf[peak_idx]
    
    print(f"3. Dominant Freq:  {peak_freq:.1f} Hz")
    
    # Check for 60Hz Hum (Common in US/Mexico)
    # We look for a spike near 60Hz
    idx_60hz = np.argmin(np.abs(xf - 60.0))
    amp_60hz = magnitude[idx_60hz]
    avg_amp = np.mean(magnitude)
    
    if amp_60hz > (avg_amp * 5): # If 60Hz is 5x stronger than average
        print("   WARNING: STRONG 60Hz HUM DETECTED. Check grounding/shielding.")
    else:
        print("   Status: OK (No significant mains hum)")

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # Plot 1: Time Domain (First 0.1s)
    zoom_samples = int(SAMPLE_RATE * 0.1) # Show 100ms
    ax1.plot(time_axis[:zoom_samples], voltage[:zoom_samples], color='blue')
    ax1.set_title("Waveform Inspection (First 100ms)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True)
    
    # Plot 2: Frequency Domain (Spectrum)
    # Normalize magnitude for display
    ax2.plot(xf, magnitude, color='red')
    ax2.set_title("Frequency Spectrum (FFT)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Strength")
    ax2.set_xlim(0, 1000) # Zoom to relevant frequencies
    ax2.grid(True)
    
    print("\nDisplaying diagnostic plots...")
    plt.show()
    
    # --- 2. SLIDER PLOT ---
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) # Make room for slider
    
    # Initial Plot (First window)
    t_min, t_max = time_axis[0], time_axis[0] + WINDOW_SIZE
    line, = ax.plot(time_axis, voltage, lw=1.5, color='blue')
    
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-10, 10) # Fixed vertical scale
    ax.set_title(f"Waveform Inspector (Sorted Data) - Window: {WINDOW_SIZE*1000:.0f}ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.5)

    # --- 3. ADD SLIDER ---
    # Position: [left, bottom, width, height]
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    
    # Slider ranges from Start Time to End Time
    slider = Slider(
        ax=ax_slider,
        label='Scroll Time',
        valmin=time_axis[0],
        valmax=time_axis[-1] - WINDOW_SIZE,
        valinit=time_axis[0],
        valstep=WINDOW_SIZE/10 # Smooth scrolling step
    )

    # --- 4. UPDATE FUNCTION ---
    def update(val):
        start = slider.val
        end = start + WINDOW_SIZE
        ax.set_xlim(start, end)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    print("Graph ready. Use the slider at the bottom to scroll.")
    plt.show()

if __name__ == "__main__":
    analyze_signal()