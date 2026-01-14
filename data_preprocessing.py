import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. SETUP PATHS ---
# Make sure these match your project structure
INPUT_FILE = 'data/processed/features.csv'
OUTPUT_FILE = 'data/processed/features_scaled.csv'
SCALER_FILE = 'data/processed/scaler.joblib'
ENCODER_FILE = 'data/processed/label_encoder.joblib'

def run_preprocessing():
    print("Starting Day 28 Preprocessing...")

    # Load the data extracted from Day 26
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run your feature extractor first!")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded dataset with {len(df)} rows.")

    # --- 2. TRANSFORM GENRE NAMES (Label Encoding) ---
    # ML models need numbers, not words (e.g., Techno -> 0, House -> 1, etc.)
    label_encoder = LabelEncoder()
    # We fit the encoder to the 'genre' column and transform it
    df['genre_encoded'] = label_encoder.fit_transform(df['genre'])
    
    print(f"Label Encoding complete. Classes: {label_encoder.classes_}")

    # --- 3. FEATURE SCALING (Standardization) ---
    # Separate the metadata (strings) from the actual features (numbers)
    # We drop 'filename' and the original 'genre' word labels
    features_to_scale = df.drop(columns=['filename', 'genre', 'genre_encoded'])
    
    scaler = StandardScaler()
    # Apply (x - mean) / std_dev to every numeric column
    X_scaled = scaler.fit_transform(features_to_scale)
    
    # Put the scaled numbers back into a nice DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=features_to_scale.columns)
    
    # Add our numeric labels back to the scaled data
    df_scaled['genre'] = df['genre_encoded']
    
    print("Feature Scaling (StandardScaler) complete.")

    # --- 4. SAVE SCALED CSV AND JOBLIB FILES ---
    # Save the finalized training data
    df_scaled.to_csv(OUTPUT_FILE, index=False)
    
    # CRITICAL: Save the scaler and encoder. 
    # You will need these to process new songs in your app later!
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)

    print("-" * 30)
    print(f"Day 28 Checklist Complete!")
    print(f"Scaled CSV: {OUTPUT_FILE}")
    print(f"Scaler Object: {SCALER_FILE}")
    print(f"Label Encoder: {ENCODER_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    run_preprocessing()