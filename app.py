import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="EDM Subgenre Classifier", page_icon="🎧")

# --- LOAD ASSETS ---
@st.cache_resource
def load_ml_assets():
    """Loads the model, scaler, and encoder once and caches them."""
    try:
        # Adjust paths if your files are in different folders
        model = joblib.load('models/baseline_model.joblib')
        scaler = joblib.load('data/processed/scaler.joblib')
        encoder = joblib.load('data/processed/label_encoder.joblib')
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Could not load model assets. Ensure you have run your training scripts! Error: {e}")
        return None, None, None

model, scaler, encoder = load_ml_assets()

# --- FEATURE EXTRACTION LOGIC ---
def extract_features_from_upload(uploaded_file):
    """Processes the uploaded file and returns a scaled feature row."""
    # Create a temporary file to save the uploaded buffer so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Load 30s snippet starting at 60s (matches our training data)
        y, sr = librosa.load(tmp_path, offset=60, duration=30)
        
        # 2. Extract Tempo (BPM) with Day 30 Fix
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        if bpm < 100:
            bpm = bpm * 2 # Normalize Dubstep half-time detections
        
        # 3. Spectral Centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 4. MFCCs (13 means and 13 stds)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # 5. Build the dictionary (MUST match the column order used in training)
        feature_dict = {
            'tempo': float(bpm),
            'spectral_centroid': float(centroid)
        }
        for i in range(13):
            feature_dict[f'mfcc_{i}_mean'] = float(mfccs_mean[i])
            feature_dict[f'mfcc_{i}_std'] = float(mfccs_std[i])
            
        # Convert to DataFrame
        feature_df = pd.DataFrame([feature_dict])
        
        # 6. Scale the features using the loaded scaler
        scaled_features = scaler.transform(feature_df)
        
        return scaled_features, bpm
    
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- UI INTERFACE ---
st.title("EDM Subgenre Classifier")
st.write("Upload a track to classify it as **Techno**, **House**, or **Dubstep**.")

uploaded_file = st.file_uploader("Drop your .mp3 or .wav here", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze & Predict"):
        if model is None:
            st.error("Model assets not found. Please check your 'models/' folder.")
        else:
            with st.spinner("Decoding audio and calculating frequencies..."):
                try:
                    # Extract and Scale
                    features, detected_bpm = extract_features_from_upload(uploaded_file)
                    
                    # Predict
                    prediction_code = model.predict(features)[0]
                    prediction_genre = encoder.inverse_transform([prediction_code])[0]
                    
                    # Probability (Optional confidence check)
                    probs = model.predict_proba(features)[0]
                    confidence = np.max(probs) * 100
                    
                    # Result Display
                    st.divider()
                    st.header(f"Prediction: {prediction_genre.upper()}")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.write(f"**Detected Tempo:** {detected_bpm:.1f} BPM")
                    
                    # Show bar chart for probabilities
                    prob_df = pd.DataFrame({
                        'Genre': encoder.classes_,
                        'Confidence': probs
                    })
                    st.bar_chart(prob_df.set_index('Genre'))
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

# --- SIDEBAR ---
st.sidebar.markdown("### Project Specs")
st.sidebar.write("Model: **Random Forest**")
st.sidebar.write("Features: **MFCCs + BPM + Spectral Centroid**")
st.sidebar.info("Tip: Upload tracks with a clear 4/4 beat or heavy bass for better results!")