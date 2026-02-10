import librosa
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = 'data/raw'
OUTPUT_PATH = 'data/processed/features_250.csv'
GENRES = ['techno', 'house', 'dubstep']

def extract_features_from_file(file_info):
    """Worker function to process a single file."""
    file_path, genre = file_info
    try:
        # Load 30s clip at 60s offset
        y, sr = librosa.load(file_path, offset=60, duration=30)
        
        # 1. Base Features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # HPSS Ratio
        harmonic, percussive = librosa.effects.hpss(y)
        h_mean = np.mean(harmonic)
        p_mean = np.mean(percussive)
        ratio = p_mean / h_mean if h_mean > 0 else 0
        
        # 2. MFCCs (Interleaved Mean/Std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Combine into interleaved list: [Base 5] + [Mean 0, Std 0, Mean 1, Std 1...]
        row = [float(tempo), centroid, flatness, zcr, ratio]
        for i in range(13):
            row.append(mfccs_mean[i])
            row.append(mfccs_std[i])
            
        row.append(genre)
        return row
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_all_files():
    tasks = []
    for genre in GENRES:
        folder = os.path.join(DATA_PATH, genre)
        for file in os.listdir(folder):
            if file.endswith(('.mp3', '.wav')):
                tasks.append((os.path.join(folder, file), genre))
    
    print(f"🚀 Starting parallel extraction for {len(tasks)} tracks...")
    
    # Using all available CPU cores
    results = []
    with ProcessPoolExecutor() as executor:
        # tqdm shows a progress bar
        results = list(tqdm(executor.map(extract_features_from_file, tasks), total=len(tasks)))
    
    # Filter out failed tracks and save
    valid_results = [r for r in results if r is not None]
    
    columns = ['tempo', 'spectral_centroid', 'spectral_flatness', 'zcr', 'perc_har_ratio']
    for i in range(13):
        columns.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    columns.append('genre')
    
    df = pd.DataFrame(valid_results, columns=columns)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Saved {len(df)} tracks to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_all_files()