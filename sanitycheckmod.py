import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import rfft, rfftfreq
from dataclasses import dataclass
from typing import Tuple, Optional
from videoprocessmod import SignalConfig

# --- 1. CONFIGURATION & DATA STRUCTURES ---

@dataclass
class DiagnosticConfig:
    sample_rate: float = 10000.0
    voltage_range: float = 10.0
    window_size_sec: float = 0.05
    mains_freq_hz: float = 60.0  # 60Hz for Mexico/US, 50Hz for Europe

@dataclass
class HealthReport:
    """Stores the results of the analysis so other scripts can read them."""
    is_clipping: bool
    is_flatline: bool
    has_mains_hum: bool
    peak_freq_hz: float
    min_voltage: float
    max_voltage: float
    std_dev: float
    # We store the FFT arrays here so we don't have to recalculate them for plotting
    fft_freqs: np.ndarray 
    fft_magnitudes: np.ndarray

# --- 2. DATA LOADING ---

def load_signal_data(file_path: str, sample_rate: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads CSV and generates the implicit time axis."""
    try:
        df = pd.read_csv(file_path)
        voltage = df.iloc[:, 0].to_numpy()
        duration = len(voltage) / sample_rate
        time_axis = np.linspace(0, duration, len(voltage), endpoint=False)
        return time_axis, voltage
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# --- 3. ANALYSIS LOGIC ---

def run_health_checks(voltage: np.ndarray, config: SignalConfig) -> HealthReport:
    """Runs all math and diagnostics, returning a structured report."""
    
    # A. Clipping Check
    max_val, min_val = np.max(voltage), np.min(voltage)
    saturation_threshold = (config.voltage_max - config.voltage_min) * 0.99
    is_clipping = bool((max_val >= saturation_threshold) or (min_val <= -saturation_threshold))
    
    # B. Flatline Check
    std_dev = float(np.std(voltage))
    is_flatline = bool(std_dev < 0.001)
    
    # C. FFT Analysis
    N = len(voltage)
    yf = rfft(voltage)
    xf = rfftfreq(N, 1 / config.sample_rate)
    magnitude = np.abs(yf)
    
    # Dominant Freq (skip DC offset at index 0)
    peak_idx = np.argmax(magnitude[1:]) + 1 
    peak_freq = float(xf[peak_idx])
    
    # Mains Hum Check
    idx_mains = np.argmin(np.abs(xf - config.mains_freq_hz))
    amp_mains = magnitude[idx_mains]
    avg_amp = np.mean(magnitude)
    has_mains_hum = bool(amp_mains > (avg_amp * 5))
    
    return HealthReport(
        is_clipping=is_clipping,
        is_flatline=is_flatline,
        has_mains_hum=has_mains_hum,
        peak_freq_hz=peak_freq,
        min_voltage=min_val,
        max_voltage=max_val,
        std_dev=std_dev,
        fft_freqs=xf,
        fft_magnitudes=magnitude
    )

def print_report(report: HealthReport):
    """Formats the HealthReport into a readable terminal output."""
    print("\n--- DIAGNOSTIC REPORT ---")
    print(f"1. Range Check:   [{report.min_voltage:.2f}V to {report.max_voltage:.2f}V]")
    if report.is_clipping: print("   WARNING: SIGNAL CLIPPING DETECTED!")
    else: print("   Status: OK")

    print(f"2. Activity Check: (Std Dev: {report.std_dev:.4f})")
    if report.is_flatline: print("   WARNING: FLATLINE DETECTED. Check sensor.")
    else: print("   Status: OK")

    print(f"3. Dominant Freq:  {report.peak_freq_hz:.1f} Hz")
    if report.has_mains_hum: print("   WARNING: STRONG MAINS HUM DETECTED. Check shielding.")
    else: print("   Status: OK (No significant hum)")
    print("-------------------------\n")

# --- 4. VISUALIZATION FUNCTIONS ---



def plot_static_diagnostics(time_axis: np.ndarray, voltage: np.ndarray, report: HealthReport, config: SignalConfig):
    """Shows the 100ms time domain and the FFT frequency spectrum."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    zoom_samples = int(config.sample_rate * 0.1)
    ax1.plot(time_axis[:zoom_samples], voltage[:zoom_samples], color='blue')
    ax1.set_title("Waveform Inspection (First 100ms)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True)
    
    ax2.plot(report.fft_freqs, report.fft_magnitudes, color='red')
    ax2.set_title("Frequency Spectrum (FFT)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Strength")
    ax2.set_xlim(0, max(1000, report.peak_freq_hz * 2)) # Dynamic zoom based on peak
    ax2.grid(True)
    
    plt.show(block=False) # Non-blocking so the slider can open immediately after

def plot_interactive_viewer(time_axis: np.ndarray, voltage: np.ndarray, config: SignalConfig):
    """Opens the scrollable waveform inspector."""
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) 
    
    t_min, t_max = time_axis[0], time_axis[0] + config.window_size_sec
    ax.plot(time_axis, voltage, lw=1.5, color='blue')
    
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-config.voltage_range, config.voltage_range) 
    ax.set_title(f"Waveform Inspector - Window: {config.window_size_sec*1000:.0f}ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.5)

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    
    slider = Slider(
        ax=ax_slider, label='Scroll Time',
        valmin=time_axis[0], valmax=time_axis[-1] - config.window_size_sec,
        valinit=time_axis[0], valstep=config.window_size_sec/10
    )

    def update(val):
        ax.set_xlim(slider.val, slider.val + config.window_size_sec)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    # Attach slider to figure so it isn't garbage collected
    fig._slider = slider 
    plt.show()

# --- 5. HIGH-LEVEL ORCHESTRATOR ---

def analyze_and_display(file_path: str, config: SignalConfig = SignalConfig()):
    """Runs the full pipeline: Load -> Analyze -> Print -> Plot -> Interactive."""
    print(f"Loading {file_path}...")
    time_axis, voltage = load_signal_data(file_path, config.sample_rate)
    
    if voltage is None: return

    report = run_health_checks(voltage, config)
    print_report(report)
    
    print("Opening diagnostic plots...")
    plot_static_diagnostics(time_axis, voltage, report, config)
    plot_interactive_viewer(time_axis, voltage, config)

if __name__ == "__main__":
    analyze_and_display("experiment_data.csv")