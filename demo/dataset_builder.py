"""
Dataset Builder & Bio-Acoustic Synthesizer with Real Plant Sounds Integration.
Integrates:
  1. Real Plant Stress Recordings (1,000 samples from plant_sounds dataset with ground-truth CSV):
     - Drought Stress
     - Heat Stress
     - Pest Infestation
     - Mechanical Damage
     - Baseline / Nutrient Deficiency
  2. Ultrasonic & Micro-Acoustic Plant Sounds (Cell 2023 PlantSounds dataset):
     - Tomato Dry & Tobacco Dry (Drought Cavitation)
     - Tomato Cut & Tobacco Cut (Mechanical Damage / Shock)
     - Empty Pot / Greenhouse Controls (Baseline Fluid Transport)
  3. Real Pest & Cricket Acoustics (ESC-50 pest pool)
  4. Real Structural Mechanical Impact Shockwaves
Produces a balanced 6,000-sample dataset (1,200 samples per class).
"""

import os
import glob
import csv
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 22050
DURATION = 5.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw_audio")
PLANT_SOUNDS_DIR = os.path.join(DATASET_DIR, "plant_sounds")
PLANT_SOUNDS_CSV = os.path.join(PLANT_SOUNDS_DIR, "plant_stress_sound_dataset.csv")
PLANT_SOUNDS_CELL_DIR = os.path.join(DATASET_DIR, "PlantSounds")


def load_and_resample(filepath, target_sr=SAMPLE_RATE, target_samples=TARGET_SAMPLES):
    """Loads an audio file, converts to mono float32, and adjusts to target length."""
    try:
        y, sr = sf.read(filepath)
    except Exception:
        try:
            sr, y = wav.read(filepath)
        except Exception:
            return np.zeros(target_samples, dtype=np.float32)
        
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
        
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        y = resample_poly(y, target_sr // gcd, sr // gcd)
        
    # Trim or loop/pad to target_samples
    if len(y) >= target_samples:
        return y[:target_samples]
    else:
        # Loop or repeat to fill duration naturally
        repeats = int(np.ceil(target_samples / max(1, len(y))))
        tiled = np.tile(y, repeats)[:target_samples]
        return tiled


def generate_click(sr=SAMPLE_RATE, center_freq=3800, decay_ms=4.0):
    """Xylem cavitation acoustic click impulse."""
    t = np.linspace(0, decay_ms / 1000.0, int(sr * (decay_ms / 1000.0)), endpoint=False)
    decay = np.exp(-t * 1300)
    return np.sin(2 * np.pi * center_freq * t) * decay


def stream_samples(target_per_class=1200):
    """
    Streams audio samples one-by-one to avoid holding 6,000 raw 5-second audio arrays in memory (which requires >4.9GB RAM).
    Classes:
      0: Healthy Baseline
      1: Drought Stress (Cavitation)
      2: Heat Stress (Burst & Quenched)
      3: Pest Attack (Chewing Vibration)
      4: Physical Damage (Shock Spike)
    """
    # 1. Load real plant sounds dataset mappings
    real_plant_by_class = {0: [], 1: [], 2: [], 3: [], 4: []}
    if os.path.exists(PLANT_SOUNDS_CSV):
        with open(PLANT_SOUNDS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = r.get('Audio_File')
                stype = r.get('Stress_Type', '').strip()
                fpath = os.path.join(PLANT_SOUNDS_DIR, fname)
                if not os.path.exists(fpath):
                    continue
                if stype == 'Drought':
                    real_plant_by_class[1].append(fpath)
                elif stype == 'Heat Stress':
                    real_plant_by_class[2].append(fpath)
                elif stype == 'Pest Infestation':
                    real_plant_by_class[3].append(fpath)
                elif stype == 'Mechanical Damage':
                    real_plant_by_class[4].append(fpath)
                else:
                    # Nutrient Deficiency / Steady state baseline
                    real_plant_by_class[0].append(fpath)

    # 2. Load PlantSounds Cell 2023 dataset pools
    ps_dry = glob.glob(os.path.join(PLANT_SOUNDS_CELL_DIR, "*Dry", "*.wav"))
    ps_cut = glob.glob(os.path.join(PLANT_SOUNDS_CELL_DIR, "*Cut", "*.wav"))
    ps_pot = glob.glob(os.path.join(PLANT_SOUNDS_CELL_DIR, "Empty Pot", "*.wav"))
    
    # 3. Locate raw ESC pools
    pest_files = glob.glob(os.path.join(RAW_DIR, "pest", "*.wav"))
    attack_files = glob.glob(os.path.join(RAW_DIR, "attack", "*.wav"))
    click_files = glob.glob(os.path.join(RAW_DIR, "drought_clicks", "*.wav"))
    ambient_files = glob.glob(os.path.join(RAW_DIR, "ambient", "*.wav"))

    np.random.seed(42)

    # Class 0: Healthy Baseline (Sap flow, fluid resonance, quiet ambient)
    c0_reals = real_plant_by_class[0]
    for i in range(target_per_class):
        if c0_reals and np.random.rand() < 0.65:
            base = load_and_resample(np.random.choice(c0_reals))
            gain = np.random.uniform(0.7, 1.1)
            sig = base * gain
            t = np.linspace(0, DURATION, TARGET_SAMPLES, endpoint=False)
            sap = 0.008 * np.sin(2 * np.pi * np.random.uniform(1.0, 2.5) * t)
            sig += sap
        else:
            t = np.linspace(0, DURATION, TARGET_SAMPLES, endpoint=False)
            sap_flow = 0.015 * np.sin(2 * np.pi * np.random.uniform(0.8, 1.8) * t)
            fluid_micro = 0.008 * np.sin(2 * np.pi * np.random.uniform(2.5, 4.2) * t)
            ambient_noise = np.random.normal(0, np.random.uniform(0.003, 0.008), TARGET_SAMPLES)
            sig = sap_flow + fluid_micro + ambient_noise
            if ps_pot and np.random.rand() < 0.3:
                pot_sig = load_and_resample(np.random.choice(ps_pot)) * 0.1
                sig += pot_sig
        
        yield np.clip(sig, -1.0, 1.0).astype(np.float32), 0

    # Class 1: Drought Stress (Cavitation Clicks: discrete xylem tik-tik)
    c1_reals = real_plant_by_class[1]
    for i in range(target_per_class):
        if c1_reals and np.random.rand() < 0.65:
            base = load_and_resample(np.random.choice(c1_reals))
            gain = np.random.uniform(0.75, 1.2)
            sig = base * gain
            if np.random.rand() < 0.5:
                n_extra = np.random.randint(3, 8)
                for _ in range(n_extra):
                    loc = np.random.randint(int(0.1 * SAMPLE_RATE), TARGET_SAMPLES - int(0.1 * SAMPLE_RATE))
                    c = generate_click(SAMPLE_RATE, center_freq=np.random.uniform(3200, 5400), decay_ms=np.random.uniform(3.0, 6.0)) * np.random.uniform(0.2, 0.5)
                    end = min(loc + len(c), TARGET_SAMPLES)
                    sig[loc:end] += c[:end - loc]
        else:
            sig = np.random.normal(0, 0.005, TARGET_SAMPLES)
            n_clicks = np.random.randint(10, 24)
            click_locs = np.random.choice(range(int(0.2 * SAMPLE_RATE), TARGET_SAMPLES - int(0.2 * SAMPLE_RATE)), size=n_clicks, replace=False)
            for loc in click_locs:
                freq = np.random.uniform(2600, 5600)
                amp = np.random.uniform(0.35, 0.75)
                c = generate_click(SAMPLE_RATE, center_freq=freq, decay_ms=np.random.uniform(3.0, 6.5)) * amp
                end = min(loc + len(c), TARGET_SAMPLES)
                sig[loc:end] += c[:end - loc]
            if ps_dry and np.random.rand() < 0.3:
                ps_sig = load_and_resample(np.random.choice(ps_dry)) * 0.15
                sig += ps_sig

        yield np.clip(sig, -1.0, 1.0).astype(np.float32), 1

    # Class 2: Heat Stress (Rapid Burst Cavitation then Thermal Shutdown)
    c2_reals = real_plant_by_class[2]
    for i in range(target_per_class):
        if c2_reals and np.random.rand() < 0.65:
            base = load_and_resample(np.random.choice(c2_reals))
            gain = np.random.uniform(0.75, 1.15)
            sig = base * gain
            burst_end = int(np.random.uniform(1.2, 2.0) * SAMPLE_RATE)
            sig[burst_end:] *= np.random.uniform(0.25, 0.55)
        else:
            sig = np.random.normal(0, 0.004, TARGET_SAMPLES)
            burst_clicks = np.random.randint(14, 28)
            burst_end = int(np.random.uniform(1.4, 2.0) * SAMPLE_RATE)
            burst_locs = np.random.choice(range(int(0.1 * SAMPLE_RATE), burst_end), size=burst_clicks, replace=False)
            for loc in burst_locs:
                freq = np.random.uniform(3000, 5800)
                amp = np.random.uniform(0.30, 0.65)
                c = generate_click(SAMPLE_RATE, center_freq=freq, decay_ms=np.random.uniform(2.5, 5.0)) * amp
                end = min(loc + len(c), TARGET_SAMPLES)
                sig[loc:end] += c[:end - loc]

        yield np.clip(sig, -1.0, 1.0).astype(np.float32), 2

    # Class 3: Pest Attack (Chewing vibrations, friction, insect harmonics)
    c3_reals = real_plant_by_class[3]
    for i in range(target_per_class):
        if c3_reals and np.random.rand() < 0.65:
            base = load_and_resample(np.random.choice(c3_reals))
            gain = np.random.uniform(0.75, 1.2)
            sig = base * gain
            t = np.linspace(0, DURATION, TARGET_SAMPLES, endpoint=False)
            chew_env = np.maximum(0, np.sin(2 * np.pi * np.random.uniform(1.5, 3.0) * t)) ** 2
            sig += np.random.normal(0, 0.05, TARGET_SAMPLES) * chew_env
        else:
            if pest_files:
                real_pest = load_and_resample(np.random.choice(pest_files))
                sig = real_pest * np.random.uniform(0.4, 0.85)
                t = np.linspace(0, DURATION, TARGET_SAMPLES, endpoint=False)
                chew_env = np.maximum(0, np.sin(2 * np.pi * np.random.uniform(1.8, 3.2) * t)) ** 2
                scrape = np.random.normal(0, 0.15, TARGET_SAMPLES) * chew_env
                sig += scrape
            else:
                t = np.linspace(0, DURATION, TARGET_SAMPLES, endpoint=False)
                chew_env = np.maximum(0, np.sin(2 * np.pi * 2.2 * t)) ** 2
                carrier = np.sin(2 * np.pi * 900 * t) + 0.5 * np.sin(2 * np.pi * 1400 * t)
                scrape = np.random.normal(0, 0.3, TARGET_SAMPLES)
                sig = (carrier * 0.25 + scrape * 0.45) * chew_env * 0.8

        yield np.clip(sig, -1.0, 1.0).astype(np.float32), 3

    # Class 4: Physical Damage / Attack (Stem cuts, mechanical trauma, fracture shocks)
    c4_reals = real_plant_by_class[4]
    for i in range(target_per_class):
        if c4_reals and np.random.rand() < 0.65:
            base = load_and_resample(np.random.choice(c4_reals))
            gain = np.random.uniform(0.8, 1.25)
            sig = base * gain
        else:
            if ps_cut and np.random.rand() < 0.5:
                cut_sig = load_and_resample(np.random.choice(ps_cut)) * np.random.uniform(0.7, 1.1)
                sig = cut_sig
            elif attack_files:
                real_attack = load_and_resample(np.random.choice(attack_files))
                shift = np.random.randint(-int(0.5 * SAMPLE_RATE), int(0.5 * SAMPLE_RATE))
                sig = np.roll(real_attack, shift) * np.random.uniform(0.7, 1.0)
            else:
                sig = np.zeros(TARGET_SAMPLES, dtype=np.float32)
                hit_start = int(np.random.uniform(0.8, 1.6) * SAMPLE_RATE)
                t_hit = np.linspace(0, (TARGET_SAMPLES - hit_start) / SAMPLE_RATE, TARGET_SAMPLES - hit_start, endpoint=False)
                thump = np.sin(2 * np.pi * 125 * t_hit) * np.exp(-t_hit * 3.5)
                sig[hit_start:] += thump * 0.95

        sig += np.random.normal(0, 0.005, TARGET_SAMPLES)
        yield np.clip(sig, -1.0, 1.0).astype(np.float32), 4


def build_dataset_features(target_per_class=1200, extract_fn=None):
    """
    Builds dataset and extracts features on-the-fly, avoiding >4.9GB RAM allocation.
    Returns (X_feats, y_labels).
    """
    X_feats = []
    y_labels = []
    total = target_per_class * 5
    count = 0
    print(f"Building bio-acoustic features dataset ({total} total samples)...")
    for sig, label in stream_samples(target_per_class=target_per_class):
        if extract_fn:
            feats, _ = extract_fn(sig, SAMPLE_RATE)
        else:
            feats = sig
        X_feats.append(feats)
        y_labels.append(label)
        count += 1
        if count % 1000 == 0:
            print(f"  Processed & extracted features: {count}/{total} samples...")
            
    return np.array(X_feats, dtype=np.float32), np.array(y_labels, dtype=np.int32)


def build_samples(target_per_class=1200):
    """Legacy helper: yields all audio signals as float32 in a list (or small batch)."""
    X_audio = []
    y_labels = []
    for audio, label in stream_samples(target_per_class=target_per_class):
        X_audio.append(audio.astype(np.float32))
        y_labels.append(label)
    return X_audio, np.array(y_labels, dtype=np.int32)


if __name__ == "__main__":
    X, y = build_samples(target_per_class=10)
    print("Test build shapes:", X.shape, y.shape)
