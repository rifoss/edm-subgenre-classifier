import os
import librosa
import numpy as np
import pandas as pd

# Define paths relative to the root of your project
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
GENRES = ['techno', 'house', 'dubstep']

def extract_features():
    data_list = []
    
    # Ensure the destination folder exists so we don't get a 'Folder Not Found' error
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    print("Starting Mass Feature Extraction...")

    for genre in GENRES:
        # Construct the path to the specific genre folder (e.g., data/raw/techno)
        genre_folder = os.path.join(RAW_DATA_PATH, genre)
        
        if not os.path.exists(genre_folder):
            print(f"Warning: Folder {genre_folder} not found. Skipping...")
            continue

        print(f"\nProcessing Genre: {genre.upper()}")
        
        # Loop through every file in the genre folder
        for filename in os.listdir(genre_folder):
            # Only process audio files
            if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                file_path = os.path.join(genre_folder, filename)
                
                try:
                    # 1. Load 30s snippet starting at 60s (the 'drop')
                    y, sr = librosa.load(file_path, offset=60, duration=30)
                    
                    # 2. Extract Tempo (BPM)
                    # We use .item() to ensure we get a single number, not an array
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    bpm = tempo.item() if isinstance(tempo, np.ndarray) else tempo

                    # Force 'half-time' detections to full-time (70 -> 140)
                    if bpm < 100:
                        bpm = bpm * 2
                    
                    # 3. Extract spectral features
                    # We take the mean to get one average value for the 30s
                    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

                    # 4. HPSS (The Week 10 Pivot)
                    # Separates percussive (drums) from harmonic (melody/chords)
                    harmonic, percussive = librosa.effects.hpss(y)
                    perc_har_ratio = np.mean(percussive) / np.mean(harmonic) if np.mean(harmonic) > 0 else 0
                    
                    # 5. Extract 13 MFCCs
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    
                    # Calculate Mean (Average Texture) and Std (The 'Swing'/Variation)
                    mfccs_mean = np.mean(mfccs, axis=1)
                    mfccs_std = np.std(mfccs, axis=1)
                    
                    # 6. Create a dictionary representing one 'row' in our future spreadsheet
                    feature_row = {
                        'filename': filename,
                        'genre': genre,
                        'tempo': bpm,
                        'spectral_centroid': centroid,
                        'spectral_flatness': flatness,
                        'zero_crossing_rate': zcr,
                        'perc_har_ratio': perc_har_ratio,
                    }
                    
                    # Add all 13 MFCC Means and 13 MFCC Stds to the dictionary
                    for i in range(13):
                        feature_row[f'mfcc_{i}_mean'] = float(mfccs_mean[i])
                        feature_row[f'mfcc_{i}_std'] = float(mfccs_std[i])
                        
                    data_list.append(feature_row)
                    print(f"Processed: {filename}")

                except Exception as e:
                    # If a file is corrupted, we skip it and keep going
                    print(f"Error processing {filename}: {e}")

    # 6. Convert the list of rows into a Pandas DataFrame
    df = pd.DataFrame(data_list)
    
    # Save to CSV in the processed data folder
    output_file = os.path.join(PROCESSED_DATA_PATH, 'features.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Dataset saved to {output_file}")
    print(f"Total tracks in your training set: {len(df)}")

if __name__ == "__main__":
    extract_features()