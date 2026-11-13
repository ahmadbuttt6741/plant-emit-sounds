"""
Download Real Bio-Acoustic and Environmental Sound Datasets from ESC-50 open archive.
Categories collected:
  - insects & crickets (for real Pest Attack chewing & stridulation)
  - door_wood_knock (for real stem/wood Mechanical Shockwaves & Physical Impacts)
  - clock_tick & water_drops (for real micro-impulsive clicks & Cavitation dynamics)
  - wind (for real botanical ambient baseline flow)
"""

import os
import urllib.request
import csv
from collections import defaultdict

BASE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
META_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "raw_audio")

TARGET_CATEGORIES = {
    "insects": "pest",
    "crickets": "pest",
    "door_wood_knock": "attack",
    "water_drops": "drought_clicks",
    "clock_tick": "drought_clicks",
    "wind": "ambient"
}

SAMPLES_PER_CATEGORY = 10  # 10 * 6 = 60 real audio WAV files


def download_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("Fetching ESC-50 metadata...")
    req = urllib.request.urlopen(META_URL)
    lines = req.read().decode('utf-8').splitlines()
    reader = csv.DictReader(lines)
    
    cat_files = defaultdict(list)
    for r in reader:
        if r['category'] in TARGET_CATEGORIES:
            cat_files[r['category']].append(r['filename'])
            
    total_downloaded = 0
    for cat, tag in TARGET_CATEGORIES.items():
        cat_dir = os.path.join(DATASET_DIR, tag)
        os.makedirs(cat_dir, exist_ok=True)
        files = cat_files[cat][:SAMPLES_PER_CATEGORY]
        print(f"Downloading {len(files)} samples for '{cat}' -> {tag}/...")
        for fname in files:
            dest = os.path.join(cat_dir, fname)
            if not os.path.exists(dest):
                url = BASE_URL + fname
                try:
                    urllib.request.urlretrieve(url, dest)
                    total_downloaded += 1
                except Exception as e:
                    print(f"  Error downloading {fname}: {e}")
            else:
                total_downloaded += 1
                
    print(f"Dataset successfully ready! Total {total_downloaded} real acoustic recordings in {DATASET_DIR}")


if __name__ == "__main__":
    download_dataset()
