import streamlit as st
import librosa
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime

# --- CONFIGURATION & LOADING ---
st.set_page_config(page_title="EDM Subgenre Classifier v2.0", page_icon="🎧")

@st.cache_resource
def load_assets():
    # We are loading 3 files. If you added a 4th one, 
    # make sure this return line matches!
    model = joblib.load('models/baseline_model.joblib')
    scaler = joblib.load('data/processed/scaler.joblib')
    encoder = joblib.load('data/processed/label_encoder.joblib')
    return model, scaler, encoder

# --- CRITICAL FIX: Unpack carefully ---
try:
    # This expects EXACTLY 3 items. 
    # If the error was here, it would say "expected 3".
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

# --- FEATURE EXTRACTION ---
def extract_live_features(y, sr):
    # 1. Base Features (5)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    harmonic, percussive = librosa.effects.hpss(y)
    perc_har_ratio = float(np.mean(percussive) / np.mean(harmonic)) if np.mean(harmonic) > 0 else 0.0
    
    # 2. MFCCs (13 Means + 13 Stds = 26)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = [float(m) for m in np.mean(mfccs, axis=1)]
    mfccs_std = [float(s) for s in np.std(mfccs, axis=1)]
    
    # 3. Combine in the EXACT order: [Base 5] + [13 Means] + [13 Stds]
    # Total: 5 + 13 + 13 = 31
    features_list = [tempo_val, centroid, flatness, zcr, perc_har_ratio]
    for i in range(13):
        features_list.append(float(mfccs_mean[i]))
        features_list.append(float(mfccs_std[i]))

    # ADD THIS LINE TO DEBUG:
    print(f"DEBUG: Feature Vector (First 5): {features_list[:5]}")
    
    return np.array(features_list).reshape(1, -1)

# --- UI LAYOUT ---
st.title("🎧 EDM Subgenre Classifier")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a track (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Analyze Subgenre"):
        with st.spinner("Analyzing rhythmic density and audio textures..."):
            try:
                # 1. Load 30 seconds from the middle
                y, sr = librosa.load(uploaded_file, offset=60, duration=30)
                
                # 2. Extract features (31 columns)
                features_vector = extract_live_features(y, sr)
                
                # 3. Scale to 0-mean/1-variance
                features_scaled = scaler.transform(features_vector)
                
                # 4. Predict
                prediction_idx = model.predict(features_scaled)[0]
                prediction_label = encoder.inverse_transform([prediction_idx])[0]
                probs = model.predict_proba(features_scaled)[0]
                
                # --- RESULTS DISPLAY ---
                st.markdown(f"### Prediction: **{prediction_label.upper()}**")
                
                # Calculate percentages
                confidences = {genre: prob * 100 for genre, prob in zip(encoder.classes_, probs)}
                max_conf = confidences[prediction_label]
                
                # Top confidence highlight
                st.metric(label="Confidence Score", value=f"{max_conf:.1f}%")

                # Detailed breakdown
                st.write("#### Confidence Breakdown:")
                cols = st.columns(len(confidences))
                for i, (genre, conf) in enumerate(confidences.items()):
                    with cols[i]:
                        st.write(f"**{genre.capitalize()}**")
                        st.write(f"{conf:.1f}%")
                
                # Confidence Chart
                conf_df = pd.DataFrame({
                    'Genre': encoder.classes_,
                    'Confidence (%)': [c for c in confidences.values()]
                })
                st.bar_chart(conf_df.set_index('Genre'))
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # --- FEEDBACK LOOP ---
    st.divider()
    st.write("### 🤖 Help the model learn")
    st.write("If the prediction was wrong, please let us know to improve the model.")
    
    col1, col2 = st.columns(2)
    with col1:
        correct_genre = st.selectbox("Select the correct genre:", ["techno", "house", "dubstep"])
    with col2:
        if st.button("🚀 Submit Correction"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            feedback_data = f"{timestamp},{uploaded_file.name},{correct_genre}\n"
            
            os.makedirs('data/processed', exist_ok=True)
            with open('data/processed/feedback_log.csv', 'a') as f:
                f.write(feedback_data)
            st.success("Correction logged! This will be used in our next retraining session.")