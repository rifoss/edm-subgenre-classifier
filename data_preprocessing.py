import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. SETUP PATHS ---
# FIX: Point this to the v5 Augmented file (59 features / 500 rows)
INPUT_FILE = 'data/processed/features_500_augmented.csv'
OUTPUT_FILE = 'data/processed/features_scaled.csv'
SCALER_FILE = 'data/processed/scaler.joblib'
ENCODER_FILE = 'data/processed/label_encoder.joblib'

def run_preprocessing():
    print("Starting Preprocessing for v5 Augmented Data...")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run batch_feature_extractor_v5.py first!")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # --- 2. TRANSFORM GENRE NAMES ---
    label_encoder = LabelEncoder()
    df['genre_encoded'] = label_encoder.fit_transform(df['genre'])
    print(f"Label Encoding complete. Classes: {label_encoder.classes_}")

    # --- 3. FEATURE SCALING ---
    # We drop 'genre' and 'genre_encoded'. 
    # If 'filename' exists, we drop that too.
    cols_to_drop = [c for c in ['filename', 'genre', 'genre_encoded'] if c in df.columns]
    features_to_scale = df.drop(columns=cols_to_drop)
    
    print(f"Scaling {len(features_to_scale.columns)} numeric features...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_to_scale)
    
    # Put the scaled numbers back into a DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=features_to_scale.columns)
    
    # Add numeric labels back (Naming it 'label' for the trainer)
    df_scaled['label'] = df['genre_encoded'].values
    
    # --- 4. SAVE ---
    df_scaled.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)

    print("-" * 30)
    print(f"Success! Scaler now supports {len(features_to_scale.columns)} features.")
    print("-" * 30)

if __name__ == "__main__":
    run_preprocessing()