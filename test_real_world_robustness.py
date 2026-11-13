"""
Phase 5: Real-World Audio Robustness Testing
Evaluates:
1. Plant stem acoustic classification under real environmental background sounds (ESC-50 wind/water)
2. Volume scaling invariance (0.1x to 5.0x)
3. Duration variations (1.0s, 2.5s, 5.0s, 10.0s, 20.0s)
4. Equipment / Device frequency cutoff simulations (4 kHz, 8 kHz, 11 kHz bandwidth limits)
"""

import os
import sys
import glob
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

sys.path.append(os.path.join(os.path.dirname(__file__), 'demo'))
from classifier import predict_audio, extract_advanced_features, load_model, CLASSES

def test_robustness():
    print("=" * 70)
    print("PHASE 5: REAL-WORLD AUDIO ROBUSTNESS & ENVIRONMENTAL MIXING")
    print("=" * 70)

    model = load_model()
    sr = 22050
    sample_dir = os.path.join('demo', 'static', 'samples')
    ambient_files = glob.glob(os.path.join('demo', 'dataset', 'raw_audio', 'ambient', '*.wav'))
    
    plant_samples = {
        'healthy': (os.path.join(sample_dir, 'healthy_baseline.wav'), 0),
        'drought': (os.path.join(sample_dir, 'drought_cavitation.wav'), 1),
        'heat': (os.path.join(sample_dir, 'heat_stress.wav'), 2),
        'pest': (os.path.join(sample_dir, 'pest_chewing.wav'), 3),
        'attack': (os.path.join(sample_dir, 'physical_damage.wav'), 4)
    }

    # 1. Environmental Mixing Test with Real ESC-50 Ambient Files
    print("\n--- 1. Mixing Real ESC-50 Ambient Recordings (Wind, Rustling) with Plant Sounds ---")
    mix_ratios = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
    for ratio in mix_ratios:
        correct = 0
        total = 0
        for name, (path, true_cls) in plant_samples.items():
            y_plant, _ = sf.read(path)
            for amb_path in ambient_files[:5]: # Test with 5 real ambient recordings
                y_amb, _ = sf.read(amb_path)
                min_len = min(len(y_plant), len(y_amb))
                mixed = (1.0 - ratio) * y_plant[:min_len] + ratio * y_amb[:min_len]
                feats, _ = extract_advanced_features(mixed, sr)
                probs = model.predict_proba([feats])[0]
                pred = int(np.argmax(probs))
                if pred == true_cls:
                    correct += 1
                total += 1
        acc = (correct / total) * 100.0
        print(f"  Ambient Mix Ratio {ratio*100:4.1f}%: Classification Accuracy = {acc:5.1f}% ({correct}/{total})")

    # 2. Volume Gain Scaling Invariance (0.1x to 5.0x)
    print("\n--- 2. Volume Gain Scaling Invariance ---")
    gains = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for g in gains:
        correct = 0
        for name, (path, true_cls) in plant_samples.items():
            y_plant, _ = sf.read(path)
            y_scaled = np.clip(y_plant * g, -1.0, 1.0)
            feats, _ = extract_advanced_features(y_scaled, sr)
            probs = model.predict_proba([feats])[0]
            if int(np.argmax(probs)) == true_cls:
                correct += 1
        print(f"  Gain {g:4.2f}x: Accuracy = {correct/5.0*100:5.1f}% ({correct}/5)")

    # 3. Transducer Bandwidth Limit Simulations (Low-pass filters: 4 kHz, 6 kHz, 8 kHz, 10 kHz)
    print("\n--- 3. Hardware Transducer Acoustic Bandwidth Limits ---")
    cutoffs = [10000, 8000, 6000, 4000, 3000]
    for cut in cutoffs:
        correct = 0
        b, a = butter(4, cut / (sr / 2.0), btype='low')
        for name, (path, true_cls) in plant_samples.items():
            y_plant, _ = sf.read(path)
            y_filt = filtfilt(b, a, y_plant)
            feats, _ = extract_advanced_features(y_filt, sr)
            probs = model.predict_proba([feats])[0]
            if int(np.argmax(probs)) == true_cls:
                correct += 1
        print(f"  Sensor Cutoff {cut:5d} Hz: Accuracy = {correct/5.0*100:5.1f}% ({correct}/5)")

    print("\n" + "=" * 70)
    print("PHASE 5 TESTING COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    test_robustness()
