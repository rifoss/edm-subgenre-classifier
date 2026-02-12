import librosa
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = 'data/raw'
OUTPUT_PATH = 'data/processed/features_250_v4.csv'
GENRES = ['techno', 'house', 'dubstep']

def extract_features_v4(file_info):
    """
    Version 4: The 'Variance Fix'. 
    Includes Mean AND Standard Deviation for texture-heavy features.
    """
    file_path, genre = file_info
    try:
        y, sr = librosa.load(file_path, offset=60, duration=30)
        
        # 1. Basic Stats
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 2. HPSS
        harmonic, percussive = librosa.effects.hpss(y)
        h_mean = np.mean(harmonic)
        p_mean = np.mean(percussive)
        ratio = p_mean / h_mean if h_mean > 0 else 0
        
        # 3. Spectral Contrast (Mean + Std) - CRITICAL FOR TECHNO
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # 4. MFCCs (Mean + Std) - CRITICAL FOR TEXTURE
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # 5. Chroma (Mean only is usually fine for harmony)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # Build the row
        row = [float(tempo), centroid, rolloff, flatness, zcr, ratio]
        
        # Interleave Mean and Std for Contrast and MFCCs
        for i in range(len(contrast_mean)):
            row.extend([contrast_mean[i], contrast_std[i]])
            
        for i in range(len(mfccs_mean)):
            row.extend([mfccs_mean[i], mfccs_std[i]])
            
        row.extend(chroma.tolist())
        row.append(genre)
        
        return row
    except Exception:
        return None

def main():
    tasks = []
    for genre in GENRES:
        folder = os.path.join(DATA_PATH, genre)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.mp3', '.wav')):
                    tasks.append((os.path.join(folder, file), genre))
    
    print(f"Running v4 (The Variance Fix) on {len(tasks)} tracks...")
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_features_v4, tasks), total=len(tasks)))
    
    valid_results = [r for r in results if r is not None]
    
    # Column Setup
    cols = ['tempo', 'centroid', 'rolloff', 'flatness', 'zcr', 'hpss_ratio']
    for i in range(7):
        cols.extend([f'contrast_{i}_mean', f'contrast_{i}_std'])
    for i in range(13):
        cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    cols.extend([f'chroma_{i}' for i in range(12)])
    cols.append('genre')
    
    df = pd.DataFrame(valid_results, columns=cols)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} tracks to {OUTPUT_PATH}")
    print(f"New Feature Count: {len(cols) - 1}")

if __name__ == "__main__":
    main()