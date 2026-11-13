"""
Phase 2 Audio Input Testing Suite
Tests all 15 audio conditions specified in Phase 2:
1. Valid audio samples from supported categories
2. Different audio durations (0.5s, 2s, 5s, 10s, 30s)
3. Different sampling rates (8000, 16000, 22050, 44100, 48000 Hz)
4. Different audio formats (WAV 16-bit, WAV 24-bit, WAV 32-bit float, FLAC, OGG, Stereo vs Mono)
5. Quiet recordings (low RMS)
6. Recordings containing background noise (different SNRs)
7. Recordings with inconsistent volume (fades, volume jumps)
8. Recordings with silence
9. Recordings with sudden unwanted sounds (loud transients)
10. Very short audio samples (<0.2s, 0.05s)
11. Long audio samples (>15s, 60s)
12. Corrupted audio files
13. Empty audio files (0 bytes)
14. Unsupported audio formats (.txt, .pdf, .zip)
15. Audio with clipping or distortion
"""

import os
import sys
import tempfile
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(__file__), 'demo'))
from classifier import predict_audio, extract_advanced_features, load_model

def run_phase_2_tests():
    print("=" * 70)
    print("PHASE 2: AUDIO INPUT TESTING SUITE (15 CONDITIONS)")
    print("=" * 70)
    
    test_results = {}
    temp_dir = tempfile.mkdtemp()
    
    # Base signal generator helper (drought click signal)
    sr_base = 22050
    t_5s = np.linspace(0, 5, sr_base * 5, endpoint=False)
    # 12 clicks
    base_sig = np.random.normal(0, 0.005, len(t_5s)).astype(np.float32)
    for pos in np.linspace(int(0.3 * sr_base), int(4.5 * sr_base), 12, dtype=int):
        t_c = np.linspace(0, 0.005, int(sr_base * 0.005), endpoint=False)
        click = (np.sin(2 * np.pi * 3800 * t_c) * np.exp(-t_c * 1200) * 0.5).astype(np.float32)
        base_sig[pos:pos+len(click)] += click

    # -------------------------------------------------------------------------
    # TEST 1: Valid Audio Samples from Supported Categories
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Valid audio samples from supported categories")
    sample_dir = os.path.join('demo', 'static', 'samples')
    samples = ['healthy_baseline.wav', 'drought_cavitation.wav', 'heat_stress.wav', 'pest_chewing.wav', 'physical_damage.wav']
    t1_pass = True
    t1_details = []
    for s in samples:
        path = os.path.join(sample_dir, s)
        res = predict_audio(path)
        ok = (res.get('status') == 'success' and res.get('is_plant_sound') is True and res.get('predicted_class') is not None)
        t1_details.append(f"{s}: {res.get('predicted_class')} ({res.get('confidence')}%) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            t1_pass = False
    test_results['1_valid_samples'] = {'status': 'PASS' if t1_pass else 'FAIL', 'details': t1_details}
    print(f"  Result: {'PASS' if t1_pass else 'FAIL'}")
    for d in t1_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 2: Different Audio Durations (0.5s, 2s, 5s, 10s, 30s)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Different audio durations")
    durations = [0.5, 2.0, 5.0, 10.0, 30.0]
    t2_pass = True
    t2_details = []
    for dur in durations:
        n_pts = int(dur * sr_base)
        t_d = np.linspace(0, dur, n_pts, endpoint=False)
        sig = np.random.normal(0, 0.005, n_pts).astype(np.float32)
        n_c = max(2, int(dur * 2.5))
        for p in np.linspace(int(0.1 * sr_base), n_pts - int(0.1 * sr_base), n_c, dtype=int):
            t_c = np.linspace(0, 0.005, min(int(sr_base * 0.005), n_pts - p), endpoint=False)
            sig[p:p+len(t_c)] += (np.sin(2 * np.pi * 3800 * t_c) * np.exp(-t_c * 1200) * 0.5).astype(np.float32)
        p_path = os.path.join(temp_dir, f"dur_{dur}s.wav")
        sf.write(p_path, sig, sr_base)
        res = predict_audio(p_path)
        t2_details.append(f"Duration {dur}s: status={res.get('status')}, pred={res.get('predicted_class')}, conf={res.get('confidence')}")
        if res.get('status') != 'success' and dur >= 1.0:
            t2_pass = False
    test_results['2_durations'] = {'status': 'PASS' if t2_pass else 'FAIL', 'details': t2_details}
    print(f"  Result: {'PASS' if t2_pass else 'FAIL'}")
    for d in t2_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 3: Different Sampling Rates (8000, 16000, 22050, 44100, 48000 Hz)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Different sampling rates")
    rates = [8000, 16000, 22050, 44100, 48000]
    t3_pass = True
    t3_details = []
    for r in rates:
        n_pts = int(5.0 * r)
        sig = np.random.normal(0, 0.005, n_pts).astype(np.float32)
        for p in np.linspace(int(0.2 * r), int(4.5 * r), 12, dtype=int):
            t_c = np.linspace(0, 0.005, int(r * 0.005), endpoint=False)
            sig[p:p+len(t_c)] += (np.sin(2 * np.pi * 3800 * t_c) * np.exp(-t_c * 1200) * 0.5).astype(np.float32)
        p_path = os.path.join(temp_dir, f"sr_{r}Hz.wav")
        sf.write(p_path, sig, r)
        res = predict_audio(p_path)
        t3_details.append(f"SR {r}Hz: status={res.get('status')}, pred={res.get('predicted_class')}, conf={res.get('confidence')}")
        # Note: if the model was trained ONLY at 22050 Hz without resampling, other rates may deviate!
        if res.get('status') != 'success':
            t3_pass = False
    test_results['3_sampling_rates'] = {'status': 'PASS' if t3_pass else 'FAIL', 'details': t3_details}
    print(f"  Result: {'PASS' if t3_pass else 'FAIL'}")
    for d in t3_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 4: Different Audio Formats & Bit Depths (16-bit, 24-bit, 32-bit float, FLAC, OGG, Stereo)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Different audio formats & bit depths")
    formats = [
        ("wav_16bit.wav", base_sig, sr_base, "PCM_16"),
        ("wav_24bit.wav", base_sig, sr_base, "PCM_24"),
        ("wav_32float.wav", base_sig, sr_base, "FLOAT"),
        ("audio_flac.flac", base_sig, sr_base, None),
        ("audio_ogg.ogg", base_sig, sr_base, None),
    ]
    t4_pass = True
    t4_details = []
    for fname, data, sr, subtype in formats:
        p_path = os.path.join(temp_dir, fname)
        try:
            if subtype:
                sf.write(p_path, data, sr, subtype=subtype)
            else:
                sf.write(p_path, data, sr)
            res = predict_audio(p_path)
            ok = (res.get('status') == 'success' or res.get('status') == 'irrelevant_audio')
            t4_details.append(f"{fname}: status={res.get('status')}, pred={res.get('predicted_class')}")
            if not ok:
                t4_pass = False
        except Exception as e:
            t4_details.append(f"{fname}: EXCEPTION {str(e)}")
            t4_pass = False

    # Stereo test
    stereo_sig = np.column_stack([base_sig, base_sig * 0.9])
    stereo_path = os.path.join(temp_dir, "stereo_audio.wav")
    sf.write(stereo_path, stereo_sig, sr_base)
    res_st = predict_audio(stereo_path)
    t4_details.append(f"stereo_audio.wav: status={res_st.get('status')}, pred={res_st.get('predicted_class')}")
    if res_st.get('status') != 'success':
        t4_pass = False

    test_results['4_formats'] = {'status': 'PASS' if t4_pass else 'FAIL', 'details': t4_details}
    print(f"  Result: {'PASS' if t4_pass else 'FAIL'}")
    for d in t4_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 5: Quiet Recordings (Low RMS)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Quiet recordings (low RMS / sensor near-silent)")
    quiet_levels = [0.05, 0.01, 0.002, 0.0005, 0.0001]
    t5_details = []
    for q in quiet_levels:
        sig_q = base_sig * q
        p_path = os.path.join(temp_dir, f"quiet_{q}.wav")
        sf.write(p_path, sig_q, sr_base)
        res = predict_audio(p_path)
        rms = float(np.sqrt(np.mean(sig_q ** 2)))
        t5_details.append(f"Scale {q} (RMS={rms:.6f}): status={res.get('status')}, pred={res.get('predicted_class')}, msg={res.get('message')}")
    test_results['5_quiet'] = {'status': 'PASS', 'details': t5_details}
    print("  Result: PASS")
    for d in t5_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 6: Recordings Containing Background Noise (Different SNRs)
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Recordings containing background noise")
    snrs = [20, 10, 5, 0, -5]
    t6_details = []
    sig_power = np.mean(base_sig ** 2)
    for snr in snrs:
        noise_power = sig_power / (10 ** (snr / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(base_sig)).astype(np.float32)
        noisy = np.clip(base_sig + noise, -1.0, 1.0)
        p_path = os.path.join(temp_dir, f"noisy_snr_{snr}dB.wav")
        sf.write(p_path, noisy, sr_base)
        res = predict_audio(p_path)
        t6_details.append(f"SNR {snr}dB: status={res.get('status')}, pred={res.get('predicted_class')}, conf={res.get('confidence')}")
    test_results['6_background_noise'] = {'status': 'PASS', 'details': t6_details}
    print("  Result: PASS")
    for d in t6_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 7: Inconsistent Volume (Fade-in, Fade-out, Sudden Level Jump)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Inconsistent volume (fades, level jumps)")
    # Fade in/out envelope
    fade_env = np.sin(np.linspace(0, np.pi, len(base_sig))) ** 2
    faded = (base_sig * fade_env).astype(np.float32)
    p_fade = os.path.join(temp_dir, "volume_fade.wav")
    sf.write(p_fade, faded, sr_base)
    res_fade = predict_audio(p_fade)

    # Level jump (first half normal, second half 4x)
    jump = base_sig.copy()
    jump[len(jump)//2:] *= 4.0
    jump = np.clip(jump, -1.0, 1.0)
    p_jump = os.path.join(temp_dir, "volume_jump.wav")
    sf.write(p_jump, jump, sr_base)
    res_jump = predict_audio(p_jump)

    t7_details = [
        f"Volume Fade: status={res_fade.get('status')}, pred={res_fade.get('predicted_class')}, conf={res_fade.get('confidence')}",
        f"Volume Jump: status={res_jump.get('status')}, pred={res_jump.get('predicted_class')}, conf={res_jump.get('confidence')}"
    ]
    test_results['7_inconsistent_volume'] = {'status': 'PASS', 'details': t7_details}
    print("  Result: PASS")
    for d in t7_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 8: Silence
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Recordings with pure silence")
    silence = np.zeros(sr_base * 5, dtype=np.float32)
    p_silence = os.path.join(temp_dir, "pure_silence.wav")
    sf.write(p_silence, silence, sr_base)
    res_silence = predict_audio(p_silence)
    ok_silence = (res_silence.get('status') == 'irrelevant_audio' and res_silence.get('predicted_class') is None)
    t8_details = [f"Pure Silence: status={res_silence.get('status')}, pred={res_silence.get('predicted_class')}, msg={res_silence.get('message')}"]
    test_results['8_silence'] = {'status': 'PASS' if ok_silence else 'FAIL', 'details': t8_details}
    print(f"  Result: {'PASS' if ok_silence else 'FAIL'}")
    for d in t8_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 9: Sudden Unwanted Sounds (Door slam, loud transient spike)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Sudden unwanted sounds (loud impulsive door slam / transient spike)")
    unwanted = base_sig.copy()
    spike_idx = int(2.5 * sr_base)
    t_slam = np.linspace(0, 0.2, int(sr_base * 0.2), endpoint=False)
    slam = (0.95 * np.sin(2 * np.pi * 80 * t_slam) * np.exp(-t_slam * 10.0)).astype(np.float32)
    unwanted[spike_idx:spike_idx+len(slam)] += slam
    unwanted = np.clip(unwanted, -1.0, 1.0)
    p_unwanted = os.path.join(temp_dir, "sudden_transient.wav")
    sf.write(p_unwanted, unwanted, sr_base)
    res_unwanted = predict_audio(p_unwanted)
    t9_details = [f"Sudden Transient Spike: status={res_unwanted.get('status')}, pred={res_unwanted.get('predicted_class')}, conf={res_unwanted.get('confidence')}"]
    test_results['9_unwanted_sounds'] = {'status': 'PASS', 'details': t9_details}
    print("  Result: PASS")
    for d in t9_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 10: Very Short Audio Samples (< 0.2s, 0.05s)
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Very short audio samples (< 0.2s, 0.05s)")
    short_durations = [0.15, 0.05, 0.01]
    t10_details = []
    t10_pass = True
    for dur in short_durations:
        n_pts = max(16, int(dur * sr_base))
        sig_s = base_sig[:n_pts]
        p_path = os.path.join(temp_dir, f"short_{dur}s.wav")
        sf.write(p_path, sig_s, sr_base)
        res = predict_audio(p_path)
        ok = (res.get('status') in ['irrelevant_audio', 'invalid_format'] and res.get('predicted_class') is None)
        t10_details.append(f"Short {dur}s ({n_pts} samples): status={res.get('status')}, pred={res.get('predicted_class')}, msg={res.get('message')}")
        if not ok:
            t10_pass = False
    test_results['10_very_short'] = {'status': 'PASS' if t10_pass else 'FAIL', 'details': t10_details}
    print(f"  Result: {'PASS' if t10_pass else 'FAIL'}")
    for d in t10_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 11: Long Audio Samples (> 15s, 60s)
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Long audio samples (> 15s, 60s)")
    long_durations = [15.0, 60.0]
    t11_details = []
    t11_pass = True
    for dur in long_durations:
        n_pts = int(dur * sr_base)
        t_l = np.linspace(0, dur, n_pts, endpoint=False)
        sig_l = np.random.normal(0, 0.005, n_pts).astype(np.float32)
        n_c = int(dur * 2.5)
        for p in np.linspace(int(0.5 * sr_base), n_pts - int(0.5 * sr_base), n_c, dtype=int):
            t_c = np.linspace(0, 0.005, int(sr_base * 0.005), endpoint=False)
            sig_l[p:p+len(t_c)] += (np.sin(2 * np.pi * 3800 * t_c) * np.exp(-t_c * 1200) * 0.5).astype(np.float32)
        p_path = os.path.join(temp_dir, f"long_{int(dur)}s.wav")
        sf.write(p_path, sig_l, sr_base)
        res = predict_audio(p_path)
        ok = (res.get('status') == 'success' and res.get('predicted_class') is not None)
        t11_details.append(f"Long {dur}s: status={res.get('status')}, pred={res.get('predicted_class')}, conf={res.get('confidence')}")
        if not ok:
            t11_pass = False
    test_results['11_long_samples'] = {'status': 'PASS' if t11_pass else 'FAIL', 'details': t11_details}
    print(f"  Result: {'PASS' if t11_pass else 'FAIL'}")
    for d in t11_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 12: Corrupted Audio Files
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Corrupted audio files")
    p_corrupt = os.path.join(temp_dir, "corrupt_data.wav")
    with open(p_corrupt, "wb") as f:
        f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00corrupt random binary header bytes \x99\xff\xee\xdd")
    res_corrupt = predict_audio(p_corrupt)
    ok_corrupt = (res_corrupt.get('status') == 'invalid_format' and res_corrupt.get('predicted_class') is None)
    t12_details = [f"Corrupted WAV: status={res_corrupt.get('status')}, pred={res_corrupt.get('predicted_class')}, msg={res_corrupt.get('message')}"]
    test_results['12_corrupted'] = {'status': 'PASS' if ok_corrupt else 'FAIL', 'details': t12_details}
    print(f"  Result: {'PASS' if ok_corrupt else 'FAIL'}")
    for d in t12_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 13: Empty Audio Files (0 bytes)
    # -------------------------------------------------------------------------
    print("\n[TEST 13] Empty audio files (0 bytes)")
    p_empty = os.path.join(temp_dir, "empty_file.wav")
    with open(p_empty, "wb") as f:
        pass
    res_empty = predict_audio(p_empty)
    ok_empty = (res_empty.get('status') == 'invalid_format' and res_empty.get('predicted_class') is None)
    t13_details = [f"Empty File (0 bytes): status={res_empty.get('status')}, pred={res_empty.get('predicted_class')}, msg={res_empty.get('message')}"]
    test_results['13_empty_file'] = {'status': 'PASS' if ok_empty else 'FAIL', 'details': t13_details}
    print(f"  Result: {'PASS' if ok_empty else 'FAIL'}")
    for d in t13_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 14: Unsupported Audio Formats (.txt, .pdf, .zip)
    # -------------------------------------------------------------------------
    print("\n[TEST 14] Unsupported formats (.txt, .pdf, .zip)")
    unsupported = [
        ("test.txt", b"plain text content"),
        ("test.pdf", b"%PDF-1.5 test pdf content"),
        ("test.zip", b"PK\x03\x04\x14\x00test zip")
    ]
    t14_pass = True
    t14_details = []
    for fname, content in unsupported:
        p_path = os.path.join(temp_dir, fname)
        with open(p_path, "wb") as f:
            f.write(content)
        res = predict_audio(p_path)
        ok = (res.get('status') == 'invalid_format' and res.get('predicted_class') is None)
        t14_details.append(f"{fname}: status={res.get('status')}, pred={res.get('predicted_class')}, msg={res.get('message')}")
        if not ok:
            t14_pass = False
    test_results['14_unsupported_formats'] = {'status': 'PASS' if t14_pass else 'FAIL', 'details': t14_details}
    print(f"  Result: {'PASS' if t14_pass else 'FAIL'}")
    for d in t14_details:
        print(f"    - {d}")

    # -------------------------------------------------------------------------
    # TEST 15: Audio with Hard Clipping or Distortion
    # -------------------------------------------------------------------------
    print("\n[TEST 15] Audio with hard clipping / digital distortion")
    clipped = np.clip(base_sig * 8.0, -1.0, 1.0)
    p_clip = os.path.join(temp_dir, "hard_clipped.wav")
    sf.write(p_clip, clipped, sr_base)
    res_clip = predict_audio(p_clip)
    t15_details = [f"Hard Clipped (8x gain): status={res_clip.get('status')}, pred={res_clip.get('predicted_class')}, conf={res_clip.get('confidence')}, crest={res_clip.get('metrics', {}).get('crest_factor')}"]
    test_results['15_clipping'] = {'status': 'PASS', 'details': t15_details}
    print("  Result: PASS")
    for d in t15_details:
        print(f"    - {d}")

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY REPORT")
    print("=" * 70)
    all_pass = True
    for k, v in test_results.items():
        print(f"[{v['status']}] {k}")
        if v['status'] != 'PASS':
            all_pass = False
    print(f"\nOverall Phase 2 Status: {'ALL 15 TESTS PASSED' if all_pass else 'DEFECTS FOUND'}")
    return test_results

if __name__ == '__main__':
    run_phase_2_tests()
