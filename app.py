import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/music_classifier.joblib'
SCALER_PATH = 'data/processed/scaler.joblib'
ENCODER_PATH = 'data/processed/label_encoder.joblib'

st.set_page_config(page_title="EDM Subgenre Classifier v5", page_icon="🎧")

def extract_features_v5_inference(file_path):
    """Mirroring the v5 Batch Extractor logic for 59 features."""
    try:
        # Load 30s sample from the middle (60s)
        y, sr = librosa.load(file_path, offset=60, duration=30)
        
        # 1. Stats
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 2. HPSS
        harmonic, percussive = librosa.effects.hpss(y)
        h_mean = np.mean(harmonic)
        p_mean = np.mean(percussive)
        ratio = p_mean / h_mean if h_mean > 0 else 0
        
        # 3. Spectral Contrast (Mean + Std)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # 4. MFCCs (Mean + Std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # 5. Chroma (Mean only)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # BUILD FEATURE VECTOR (Must match training column order exactly!)
        row = [float(tempo), centroid, rolloff, flatness, zcr, rms, ratio]
        for i in range(len(contrast_mean)):
            row.extend([contrast_mean[i], contrast_std[i]])
        for i in range(len(mfccs_mean)):
            row.extend([mfccs_mean[i], mfccs_std[i]])
        row.extend(chroma.tolist())
        
        return np.array(row).reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- UI LAYOUT ---
st.title("🎧 EDM Subgenre Classifier")
st.markdown(f"**Model Version:** 5.0 (Augmented) | **Current Accuracy:** 83.8%")

uploaded_file = st.file_uploader("Upload a Techno, House, or Dubstep track (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save temp file
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    with st.spinner("Analyzing audio textures..."):
        # 1. Extract
        features = extract_features_v5_inference("temp_audio.mp3")
        
        if features is not None:
            # 2. Load Assets
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            encoder = joblib.load(ENCODER_PATH)
            
            # 3. Scale & Predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            probs = model.predict_proba(features_scaled)[0]
            
            genre = encoder.inverse_transform(prediction)[0]
            
            # 4. Results Display
            st.success(f"### Predicted Genre: **{genre.upper()}**")
            
            # Confidence Chart
            st.write("#### Model Confidence:")
            prob_df = pd.DataFrame({
                'Genre': encoder.classes_,
                'Confidence': probs
            })
            st.bar_chart(prob_df.set_index('Genre'))
            
            # Advice based on confidence
            top_prob = np.max(probs)
            if top_prob < 0.60:
                st.warning("The model is uncertain. This track might be a 'Genre-Bender' (e.g., Tech-House).")
            else:
                st.info(f"The model is {top_prob:.1%} confident in this classification.")

    # Cleanup
    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")