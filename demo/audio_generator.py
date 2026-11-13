"""
Synthetic Bio-Acoustic Signal Generator for Plant Stem Piezo Sensor.
Generates realistic audio/vibration signals matching the 5 plant conditions
described in 'AI based Bio-Acoustic sensor in plants.docx':
  1. Healthy Baseline (Calm fluid flow / resting)
  2. Drought Stress (Cavitation clicks: discrete 'tik tik')
  3. Heat Stress (Fast burst clicks then silent)
  4. Pest Attack (Continuous mechanical chewing friction: 'krrr')
  5. Physical Attack (Sudden high-amplitude impact shock: 'DHAM spike')
"""

import os
import numpy as np
import scipy.io.wavfile as wav

SAMPLE_RATE = 22050  # 22.05 kHz standard
DURATION = 5.0       # 5 seconds audio clips


def generate_click(sr=SAMPLE_RATE, center_freq=3500, decay_ms=4.0):
    """Generates a single xylem cavitation click (sharp damped sinusoid)."""
    t = np.linspace(0, decay_ms / 1000.0, int(sr * (decay_ms / 1000.0)), endpoint=False)
    decay = np.exp(-t * 1200)
    click = np.sin(2 * np.pi * center_freq * t) * decay
    return click


def generate_healthy(duration=DURATION, sr=SAMPLE_RATE, seed=None):
    """Healthy Plant: Gentle baseline fluid micro-oscillation and low background noise."""
    if seed is not None:
        np.random.seed(seed)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Low amplitude fluid rhythmic motion
    fluid_flow = 0.015 * np.sin(2 * np.pi * 1.2 * t) + 0.008 * np.sin(2 * np.pi * 3.5 * t)
    # Gentle pink/white noise floor
    noise = np.random.normal(0, 0.006, n_samples)
    signal = fluid_flow + noise
    return np.clip(signal, -1.0, 1.0)


def generate_drought(duration=DURATION, sr=SAMPLE_RATE, seed=None, n_clicks=12):
    """Drought Stress: Discrete xylem cavitation clicks ('tik... tik') spaced throughout."""
    if seed is not None:
        np.random.seed(seed)
    signal = generate_healthy(duration, sr, seed=seed) * 0.5
    n_samples = len(signal)
    
    # Place discrete clicks uniformly spread over time
    click_positions = np.sort(np.random.choice(range(int(0.3 * sr), n_samples - int(0.3 * sr)), size=n_clicks, replace=False))
    
    for pos in click_positions:
        freq = np.random.uniform(2800, 5200)  # Acoustic cavitation frequency range
        amp = np.random.uniform(0.35, 0.65)
        click = generate_click(sr, center_freq=freq, decay_ms=np.random.uniform(3.0, 6.0)) * amp
        end = min(pos + len(click), n_samples)
        signal[pos:end] += click[:end - pos]
        
    return np.clip(signal, -1.0, 1.0)


def generate_heat(duration=DURATION, sr=SAMPLE_RATE, seed=None, burst_clicks=18):
    """Heat Stress: Fast burst of cavitation clicks in first 1.8s, then sudden heat silence."""
    if seed is not None:
        np.random.seed(seed)
    signal = generate_healthy(duration, sr, seed=seed) * 0.3
    burst_end_sample = int(1.8 * sr)
    
    # Rapid burst clicks in first 1.8s
    click_positions = np.sort(np.random.choice(range(int(0.1 * sr), burst_end_sample), size=burst_clicks, replace=False))
    for pos in click_positions:
        freq = np.random.uniform(3200, 5800)
        amp = np.random.uniform(0.30, 0.60)
        click = generate_click(sr, center_freq=freq, decay_ms=np.random.uniform(2.5, 5.0)) * amp
        end = min(pos + len(click), len(signal))
        signal[pos:end] += click[:end - pos]
        
    # Second half is silent / quenched transpiration
    return np.clip(signal, -1.0, 1.0)


def generate_pest(duration=DURATION, sr=SAMPLE_RATE, seed=None):
    """Pest Attack: Continuous mechanical chewing friction vibration ('krrr')."""
    if seed is not None:
        np.random.seed(seed)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Chewing modulation: cyclic rhythmic chewing (approx 1.5 - 2.5 chews per second)
    chew_envelope = np.maximum(0, np.sin(2 * np.pi * 2.2 * t)) ** 2
    
    # Friction noise textured at mid-frequencies (600 - 1800 Hz)
    carrier = np.sin(2 * np.pi * 850 * t) + 0.6 * np.sin(2 * np.pi * 1350 * t) + 0.4 * np.sin(2 * np.pi * 420 * t)
    scrape = np.random.normal(0, 0.25, n_samples)
    
    signal = (carrier * 0.2 + scrape * 0.4) * chew_envelope * 0.8
    # Add small baseline
    signal += generate_healthy(duration, sr, seed=seed) * 0.3
    return np.clip(signal, -1.0, 1.0)


def generate_attack(duration=DURATION, sr=SAMPLE_RATE, seed=None):
    """Physical Attack: Sudden high-amplitude shockwave ('DHAM spike') and resonant ring-down."""
    if seed is not None:
        np.random.seed(seed)
    signal = generate_healthy(duration, sr, seed=seed) * 0.4
    n_samples = len(signal)
    
    # Hit occurs at around 1.2 seconds
    hit_start = int(1.2 * sr)
    hit_samples = n_samples - hit_start
    t_hit = np.linspace(0, hit_samples / sr, hit_samples, endpoint=False)
    
    # Resonant stem mechanical thump (120 Hz) with exponential decay
    thump = np.sin(2 * np.pi * 125 * t_hit) * np.exp(-t_hit * 3.5)
    harmonics = 0.5 * np.sin(2 * np.pi * 310 * t_hit) * np.exp(-t_hit * 7.0)
    impact_crackle = np.random.normal(0, 0.4, hit_samples) * np.exp(-t_hit * 15.0)
    
    spike = (thump + harmonics + impact_crackle) * 0.95
    signal[hit_start:] += spike
    return np.clip(signal, -1.0, 1.0)


def save_wav(filepath, signal, sr=SAMPLE_RATE):
    """Saves float signal [-1.0, 1.0] to 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    signal_16bit = np.int16(signal * 32767)
    wav.write(filepath, sr, signal_16bit)


def generate_all_samples(output_dir="demo/static/samples"):
    """Generates the 5 canonical demo samples and a dataset for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    generators = {
        "healthy_baseline.wav": (generate_healthy, "Healthy Plant (Quiet baseline)"),
        "drought_cavitation.wav": (generate_drought, "Drought Stress (Cavitation tik-tik)"),
        "heat_stress.wav": (generate_heat, "Heat Stress (Rapid burst then silent)"),
        "pest_chewing.wav": (generate_pest, "Pest Attack (Continuous chewing krrr)"),
        "physical_damage.wav": (generate_attack, "Physical Attack (DHAM shock spike)"),
    }
    
    print("Generating canonical demo audio files...")
    for filename, (gen_func, desc) in generators.items():
        path = os.path.join(output_dir, filename)
        sig = gen_func(duration=5.0, seed=42)
        save_wav(path, sig)
        print(f" -> Created {path} ({desc})")
        
    print("All demo samples generated successfully!")


if __name__ == "__main__":
    generate_all_samples()
