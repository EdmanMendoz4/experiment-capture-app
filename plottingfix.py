import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- CONFIGURATION ---
DATA_FILE = "experiment_data.csv"
WINDOW_SIZE = 0.05  # Initial view width (seconds)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # --- 1. DATA CLEANING (Crucial Fix) ---
    print("Cleaning data artifacts...")
    
    # Sort by timestamp to fix "Backward Time" errors
    df = df.sort_values(by='Timestamp')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Timestamp'])
    
    # Optional: Re-calculate timestamps to fix "Gaps"
    # If we assume the sampling rate was constant (5000 Hz), we can force perfect timing:
    # This makes the graph look perfect even if the recording lagged.
    # df['Timestamp'] = np.arange(len(df)) / 5000.0 

    t = df['Timestamp'].values
    v = df['Voltage'].values
    
    # --- 2. SETUP PLOT ---
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) # Make room for slider
    
    # Initial Plot (First window)
    t_min, t_max = t[0], t[0] + WINDOW_SIZE
    line, = ax.plot(t, v, lw=1.5, color='blue')
    
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
        valmin=t[0],
        valmax=t[-1] - WINDOW_SIZE,
        valinit=t[0],
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
    import numpy as np # Needed for the optional re-calc
    main()