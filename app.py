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

st.set_page_config(page_title="EDM Subgenre Classifier v5", page_icon="🎧", layout="centered")

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
        
        # BUILD FEATURE VECTOR
        row = [float(tempo), centroid, rolloff, flatness, zcr, rms, ratio]
        for i in range(len(contrast_mean)):
            row.extend([contrast_mean[i], contrast_std[i]])
        for i in range(len(mfccs_mean)):
            row.extend([mfccs_mean[i], mfccs_std[i]])
        row.extend(chroma.tolist())
        
        return np.array(row).reshape(1, -1)

    except Exception as e:
        # We don't use st.error here to avoid double-posting errors in the UI
        print(f"Extraction Error: {e}")
        return None

# --- SIDEBAR & CREDITS ---
st.sidebar.title("🎧 Project Details")
st.sidebar.info("""
**Model Version:** 5.0 (Augmented)
**Accuracy:** 84.5%
**Input:** 59 Signal Descriptors
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developer")
st.sidebar.markdown("**Jody Suryatna**")
# Replace 'your-profile-url' with your actual LinkedIn slug
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jody-suryatna/)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?style=for-the-badge&logo=github)](https://github.com/rifoss/edm-subgenre-classifier)")

# --- MAIN UI LAYOUT ---
st.title("🎧 EDM Subgenre Classifier")
st.markdown("Identify the subgenre of your track using high-dimensional texture analysis.")

# Unified temp file name
TEMP_FILE = "temp_audio_upload.mp3"

uploaded_file = st.file_uploader("Upload a Techno, House, or Dubstep track (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save temp file
    with open(TEMP_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    if st.button("Analyze Track"):
        with st.spinner("Analyzing audio textures..."):
            try:
                # 1. Duration Check - Wrapped in a specific try/except for corrupted files
                try:
                    duration = librosa.get_duration(path=TEMP_FILE)
                except Exception:
                    st.warning("*Invalid or Corrupted File:** Could not analyze file. Please ensure it is a valid, uncorrupted audio file.")
                    st.stop() # Prevents further execution for this button click

                if duration < 90:
                    st.error(f"**File Too Short:** The track is only {duration:.1f}s. Please upload a full track (at least 90s) so the model can sample the 'drop' at the 60s mark.")
                else:
                    # 2. Feature Extraction
                    features = extract_features_v5_inference(TEMP_FILE)
                    
                    if features is not None:
                        # 3. Load Assets
                        model = joblib.load(MODEL_PATH)
                        scaler = joblib.load(SCALER_PATH)
                        encoder = joblib.load(ENCODER_PATH)
                        
                        # 4. Prediction
                        features_scaled = scaler.transform(features)
                        probs = model.predict_proba(features_scaled)[0]
                        prediction = np.argmax(probs)
                        genre = encoder.classes_[prediction]
                        confidence = np.max(probs)
                        
                        # 5. Results Display
                        st.success(f"### Predicted Genre: **{genre.upper()}**")
                        
                        if confidence < 0.65:
                            st.warning(f"**Low Confidence ({confidence:.1%}):** This track may be a hybrid (e.g. Tech-House) or an edge case.")
                        else:
                            st.info(f"Model Confidence: **{confidence:.1%}**")
                        
                        chart_data = pd.DataFrame({'Genre': encoder.classes_, 'Confidence': probs}).set_index('Genre')
                        st.bar_chart(chart_data)
                    else:
                        st.warning("**Analysis Failed:** Could not extract features. The audio file might be corrupted or in an unsupported format.")

            except FileNotFoundError:
                st.error("**Critical Error:** Model files (`.joblib`) not found. Please run training first.")
            except ValueError:
                st.error("**Dimension Mismatch:** The model expects a different number of features. Update your Scaler!")
            except Exception as e:
                st.error(f"**An unexpected error occurred:** {e}")

    # --- CLEANUP ---
    if os.path.exists(TEMP_FILE):
        try:
            os.remove(TEMP_FILE)
        except Exception:
            pass