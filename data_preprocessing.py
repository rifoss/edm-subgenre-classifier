import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. SETUP PATHS ---
INPUT_FILE = 'data/processed/features_500_augmented.csv'
OUTPUT_FILE = 'data/processed/features_scaled.csv'
SCALER_FILE = 'data/processed/scaler.joblib'
ENCODER_FILE = 'data/processed/label_encoder.joblib'

def run_preprocessing():
    print("Starting Preprocessing...")

    # Load the data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run your feature extractor first!")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # --- 2. TRANSFORM GENRE NAMES (Label Encoding) ---
    if 'genre' not in df.columns:
        print("Error: 'genre' column missing from CSV. Check extractor output.")
        return

    label_encoder = LabelEncoder()
    df['genre_encoded'] = label_encoder.fit_transform(df['genre'])
    
    print(f"Label Encoding complete. Classes: {label_encoder.classes_}")

    # --- 3. FEATURE SCALING (Standardization) ---
    # We create a list of columns to drop, but only if they actually exist
    # this prevents the 'KeyError' you were likely seeing
    potential_metadata = ['filename', 'genre', 'genre_encoded']
    actual_cols_to_drop = [col for col in potential_metadata if col in df.columns]
    
    features_to_scale = df.drop(columns=actual_cols_to_drop)
    
    print(f"Scaling {len(features_to_scale.columns)} numeric features...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_to_scale)
    
    # Put the scaled numbers back into a nice DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=features_to_scale.columns)
    
    # Add our numeric labels back to the scaled data
    # We'll call the column 'label' to stay consistent with your training scripts
    df_scaled['label'] = df['genre_encoded']
    
    print("Feature Scaling (StandardScaler) complete.")

    # --- 4. SAVE SCALED CSV AND JOBLIB FILES ---
    df_scaled.to_csv(OUTPUT_FILE, index=False)
    
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)

    print("-" * 30)
    print(f"Success: Data scaled and transformers saved.")
    print(f"Final Feature Count: {len(features_to_scale.columns)}")
    print(f"Target Column: 'label'")
    print("-" * 30)

if __name__ == "__main__":
    run_preprocessing()