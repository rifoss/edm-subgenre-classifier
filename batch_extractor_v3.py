import librosa
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = 'data/raw'
OUTPUT_PATH = 'data/processed/features_500_augmented.csv'
GENRES = ['techno', 'house', 'dubstep']

def extract_features_v5(file_info):
    """
    Version 5: The Augmented Extractor (Fixed).
    Takes TWO samples from every song (at 45s and 120s) to double the dataset.
    """
    file_path, genre = file_info
    offsets = [45, 120]
    results = []

    for offset in offsets:
        try:
            y, sr = librosa.load(file_path, offset=offset, duration=30)
            
            # 1. Stats (7 features)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            rms = np.mean(librosa.feature.rms(y=y))
            
            harmonic, percussive = librosa.effects.hpss(y)
            h_mean = np.mean(harmonic)
            p_mean = np.mean(percussive)
            ratio = p_mean / h_mean if h_mean > 0 else 0
            
            # 2. Spectral Contrast (Mean + Std = 14 features)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            contrast_std = np.std(contrast, axis=1)
            
            # 3. MFCCs (Mean + Std = 26 features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            # 4. Chroma (Mean only = 12 features)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            
            # BUILD ROW (Total: 7 + 14 + 26 + 12 = 59 features)
            row = [float(tempo), centroid, rolloff, flatness, zcr, rms, ratio]
            
            for i in range(len(contrast_mean)):
                row.extend([contrast_mean[i], contrast_std[i]])
                
            for i in range(len(mfccs_mean)):
                row.extend([mfccs_mean[i], mfccs_std[i]])
            
            # FIX: Added the missing Chroma features here
            row.extend(chroma.tolist())
            
            row.append(genre)
            results.append(row)
        except Exception:
            continue
    return results

def main():
    tasks = []
    for genre in GENRES:
        folder = os.path.join(DATA_PATH, genre)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.mp3', '.wav')):
                    tasks.append((os.path.join(folder, file), genre))
    
    print(f"Correcting dataset: Extracting 59 features for {len(tasks)} tracks...")
    
    with ProcessPoolExecutor() as executor:
        nested_results = list(tqdm(executor.map(extract_features_v5, tasks), total=len(tasks)))
    
    final_results = [item for sublist in nested_results for item in sublist]
    
    cols = ['tempo', 'centroid', 'rolloff', 'flatness', 'zcr', 'rms', 'hpss_ratio']
    for i in range(7):
        cols.extend([f'contrast_{i}_mean', f'contrast_{i}_std'])
    for i in range(13):
        cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    cols.extend([f'chroma_{i}' for i in range(12)])
    cols.append('genre')
    
    df = pd.DataFrame(final_results, columns=cols)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Generated {len(df)} samples with 59 features each.")

if __name__ == "__main__":
    main()