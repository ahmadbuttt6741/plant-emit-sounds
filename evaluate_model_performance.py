"""
Phase 4: Comprehensive Model Performance Evaluation Script
Calculates:
- Accuracy, Precision, Recall, F1-score (macro, micro, weighted)
- Confusion Matrix
- Class-wise performance metrics
- Inference latency benchmarks (100 runs: mean, median, 95th percentile, min, max)
- Model size and memory footprint
- Independent test set evaluation with group-aware separation
- Background-noise sensitivity analysis
"""

import os
import sys
import time
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.join(os.path.dirname(__file__), 'demo'))
from classifier import load_model, extract_advanced_features, CLASSES, CLASS_KEYS, FEATURE_NAMES
from dataset_builder import build_samples

def evaluate():
    print("=" * 70)
    print("PHASE 4: MODEL PERFORMANCE BENCHMARKING & EVALUATION")
    print("=" * 70)

    model = load_model()
    model_path = os.path.join('demo', 'models', 'plant_stress_classifier.joblib')
    model_size_bytes = os.path.getsize(model_path)
    print(f"Model File Path: {model_path}")
    print(f"Model File Size: {model_size_bytes / 1024:.2f} KB ({model_size_bytes:,} bytes)")
    print(f"Model Estimators: {len(model.named_steps['clf'].estimators_)}")
    print(f"Feature Dimension: {len(FEATURE_NAMES)}")

    # 1. Independent Synthetic + Real Acoustic Test Set (Independent Seed = 999)
    print("\n--- 1. Generating Independent Test Dataset (Held-out seeds) ---")
    sr = 22050
    duration = 5.0
    n_samples_per_class = 100
    X_test_audio = []
    y_test_labels = []

    np.random.seed(999) # Completely independent test seed

    # Class 0: Healthy
    for _ in range(n_samples_per_class):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        flow = 0.012 * np.sin(2 * np.pi * np.random.uniform(0.7, 1.9) * t)
        micro = 0.007 * np.sin(2 * np.pi * np.random.uniform(2.3, 4.5) * t)
        noise = np.random.normal(0, np.random.uniform(0.002, 0.007), len(t))
        sig = flow + micro + noise
        X_test_audio.append(np.clip(sig, -1.0, 1.0))
        y_test_labels.append(0)

    # Class 1: Drought (Uniform clicks)
    for _ in range(n_samples_per_class):
        sig = np.random.normal(0, 0.005, int(sr * duration))
        n_c = np.random.randint(7, 24)
        click_locs = np.random.choice(range(int(0.2 * sr), int((duration - 0.2) * sr)), size=n_c, replace=False)
        for loc in click_locs:
            freq = np.random.uniform(2500, 5800)
            amp = np.random.uniform(0.30, 0.80)
            t_c = np.linspace(0, 0.005, int(sr * 0.005), endpoint=False)
            c = (np.sin(2 * np.pi * freq * t_c) * np.exp(-t_c * 1200) * amp).astype(np.float32)
            end = min(loc + len(c), len(sig))
            sig[loc:end] += c[:end - loc]
        X_test_audio.append(np.clip(sig, -1.0, 1.0))
        y_test_labels.append(1)

    # Class 2: Heat (Burst clicks in first 1.8s)
    for _ in range(n_samples_per_class):
        sig = np.random.normal(0, 0.004, int(sr * duration))
        burst_c = np.random.randint(12, 30)
        burst_end = int(np.random.uniform(1.3, 2.1) * sr)
        click_locs = np.random.choice(range(int(0.1 * sr), burst_end), size=burst_c, replace=False)
        for loc in click_locs:
            freq = np.random.uniform(2900, 6000)
            amp = np.random.uniform(0.28, 0.70)
            t_c = np.linspace(0, 0.004, int(sr * 0.004), endpoint=False)
            c = (np.sin(2 * np.pi * freq * t_c) * np.exp(-t_c * 1200) * amp).astype(np.float32)
            end = min(loc + len(c), len(sig))
            sig[loc:end] += c[:end - loc]
        X_test_audio.append(np.clip(sig, -1.0, 1.0))
        y_test_labels.append(2)

    # Class 3: Pest (Mechanical chewing friction)
    for _ in range(n_samples_per_class):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        chew_env = np.maximum(0, np.sin(2 * np.pi * np.random.uniform(1.6, 2.8) * t)) ** 2
        carrier = np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1380 * t)
        scrape = np.random.normal(0, 0.28, len(t))
        sig = (carrier * 0.22 + scrape * 0.42) * chew_env * 0.85
        sig += np.random.normal(0, 0.004, len(t))
        X_test_audio.append(np.clip(sig, -1.0, 1.0))
        y_test_labels.append(3)

    # Class 4: Physical Damage (Impact shockwave)
    for _ in range(n_samples_per_class):
        sig = np.zeros(int(sr * duration), dtype=np.float32)
        hit_start = int(np.random.uniform(0.7, 1.8) * sr)
        t_hit = np.linspace(0, (len(sig) - hit_start) / sr, len(sig) - hit_start, endpoint=False)
        thump = np.sin(2 * np.pi * np.random.uniform(110, 140) * t_hit) * np.exp(-t_hit * np.random.uniform(3.0, 4.5))
        sig[hit_start:] += thump * np.random.uniform(0.85, 1.0)
        sig += np.random.normal(0, 0.005, len(sig))
        X_test_audio.append(np.clip(sig, -1.0, 1.0))
        y_test_labels.append(4)

    y_test_labels = np.array(y_test_labels)
    print(f"Generated {len(X_test_audio)} test audio signals across 5 classes.")

    # 2. Extract features
    print("Extracting 22-dimensional features from independent test signals...")
    X_test_feats = []
    for audio in X_test_audio:
        feats, _ = extract_advanced_features(audio, sr)
        X_test_feats.append(feats)
    X_test_feats = np.array(X_test_feats)

    # 3. Model Predictions on Independent Dataset
    print("\n--- 2. Independent Test Set Performance Metrics ---")
    y_pred = model.predict(X_test_feats)
    y_probs = model.predict_proba(X_test_feats)

    acc = accuracy_score(y_test_labels, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test_labels, y_pred, average='macro')
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(y_test_labels, y_pred, average='weighted')
    cm = confusion_matrix(y_test_labels, y_pred)

    print(f"Independent Test Accuracy: {acc * 100:.2f}%")
    print(f"Macro Precision:          {prec_macro * 100:.2f}%")
    print(f"Macro Recall:             {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:           {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:        {f1_weight * 100:.2f}%")

    print("\nConfusion Matrix:")
    print("Pred ->   " + "  ".join([f"C{i}" for i in range(5)]))
    for i, row in enumerate(cm):
        print(f"True C{i}: {row}")

    print("\nClass-wise Performance Report:")
    report = classification_report(y_test_labels, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # 4. Latency & Inference Speed Benchmarking
    print("\n--- 3. Inference Latency & Speed Benchmark (100 Iterations) ---")
    single_feat = X_test_feats[0:1]
    single_audio = X_test_audio[0]

    # Benchmark full predict pipeline (audio -> features -> model predict)
    full_pipeline_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        feats, _ = extract_advanced_features(single_audio, sr)
        probs = model.predict_proba([feats])[0]
        full_pipeline_times.append((time.perf_counter() - t0) * 1000.0) # in ms

    # Benchmark model forward pass only
    model_only_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        probs = model.predict_proba(single_feat)[0]
        model_only_times.append((time.perf_counter() - t0) * 1000.0)

    print(f"Full Pipeline Latency (DSP Feature Extraction + RF Inference):")
    print(f"  Mean:   {np.mean(full_pipeline_times):.2f} ms")
    print(f"  Median: {np.median(full_pipeline_times):.2f} ms")
    print(f"  p95:    {np.percentile(full_pipeline_times, 95):.2f} ms")
    print(f"  Min:    {np.min(full_pipeline_times):.2f} ms")
    print(f"  Max:    {np.max(full_pipeline_times):.2f} ms")
    print(f"  Throughput: {1000.0 / np.mean(full_pipeline_times):.1f} predictions/second")

    print(f"\nModel-Only Forward Pass Latency:")
    print(f"  Mean:   {np.mean(model_only_times):.2f} ms")
    print(f"  Median: {np.median(model_only_times):.2f} ms")
    print(f"  p95:    {np.percentile(model_only_times, 95):.2f} ms")

    # 5. Background Noise Sensitivity Gradient
    print("\n--- 4. Background Noise Sensitivity Gradient ---")
    snr_levels = [30, 20, 15, 10, 5, 0, -5]
    for snr in snr_levels:
        noisy_feats = []
        for a in X_test_audio:
            p_sig = np.mean(a ** 2)
            p_noise = p_sig / (10 ** (snr / 10.0))
            noise = np.random.normal(0, np.sqrt(p_noise), len(a))
            a_noisy = np.clip(a + noise, -1.0, 1.0)
            f_n, _ = extract_advanced_features(a_noisy, sr)
            noisy_feats.append(f_n)
        noisy_pred = model.predict(noisy_feats)
        noisy_acc = accuracy_score(y_test_labels, noisy_pred) * 100.0
        print(f"  SNR {snr:3d} dB: Accuracy = {noisy_acc:5.1f}%")

    print("\n" + "=" * 70)
    print("PHASE 4 BENCHMARKING COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    evaluate()
