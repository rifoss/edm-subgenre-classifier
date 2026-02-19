import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
import tempfile

import warnings
import logging
logging.getLogger("xgboost").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from mutagen import File as MutaFile

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to your assets
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'music_classifier.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib')
ENCODER_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'label_encoder.joblib')
FEEDBACK_PATH = os.path.join(BASE_DIR, 'data', 'feedback.csv')

st.set_page_config(page_title="EDM Subgenre Classifier v5", page_icon="🎧", layout="centered")

# Initialize Session State for persistence
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def save_feedback(filename, predicted, corrected):
    """Appends user feedback to a local CSV file. Note: will be replaced with Supabase in next version."""
    new_data = pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'predicted_genre': predicted,
        'corrected_genre': corrected
    }])
    
    # Check if log exists to handle header correctly
    if not os.path.isfile(FEEDBACK_PATH):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        new_data.to_csv(FEEDBACK_PATH, index=False)
    else:
        new_data.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)

def extract_features_v5_inference(file_path):
    """Extracts 59 audio features from a 30s window at the 60s mark, matching the v5 training pipeline."""
    try:
        y, sr = librosa.load(file_path, sr=22050, offset=60, duration=30, res_type='soxr_qq')

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.squeeze(tempo))

        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        harmonic, percussive = librosa.effects.hpss(y)
        h_mean = np.mean(harmonic)
        p_mean = np.mean(percussive)
        ratio = p_mean / h_mean if h_mean > 0 else 0

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

        row = [float(tempo), centroid, rolloff, flatness, zcr, rms, ratio]
        for i in range(len(contrast_mean)):
            row.extend([contrast_mean[i], contrast_std[i]])
        for i in range(len(mfccs_mean)):
            row.extend([mfccs_mean[i], mfccs_std[i]])
        row.extend(chroma.tolist())

        return np.array(row).reshape(1, -1)

    except Exception as e:
        st.error(f"EXTRACTION FAILED AT: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder
    
def get_audio_duration(path):
    try:
        audio = MutaFile(path)
        return audio.info.length if audio else 0
    except Exception:
        return 0

# --- SIDEBAR & CREDITS ---
st.sidebar.title("🎧 Project Details")
st.sidebar.info("""
**Model Version:** 5.0 (Augmented)
**Accuracy:** 83.8%
**Input:** 59 Signal Descriptors
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developer")
st.sidebar.markdown("**Jody Suryatna**")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jody-suryatna/)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?style=for-the-badge&logo=github)](https://github.com/rifoss/edm-subgenre-classifier)")

# --- MAIN UI LAYOUT ---
st.title("🎧 EDM Subgenre Classifier")
st.markdown("Identify the subgenre of your track using high-dimensional texture analysis.")

uploaded_file = st.file_uploader("Upload a Techno, House, or Dubstep track (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Derive extension from upload to support both MP3 and WAV
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    TEMP_FILE = os.path.join(tempfile.gettempdir(), f"temp_audio_upload{ext}")

    # Check if a new file was uploaded to reset previous results
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.prediction_results = None
        st.session_state.submitted = False
        st.session_state.last_file = uploaded_file.name

    with open(TEMP_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    if st.button("Analyze Track"):
        with st.spinner("Analyzing audio textures..."):
            try:
                # 1. Validate duration - rejects short or corrupted files before feature extraction
                try:
                    duration = get_audio_duration(TEMP_FILE)
                except Exception:
                    st.warning("**Invalid or Corrupted File:** Could not analyze file. Please ensure it is a valid, uncorrupted audio file.")
                    st.stop() # Prevents further execution for this button click

                if duration < 90:
                    st.error(f"**File Too Short:** The track is only {duration:.1f}s. Please upload a full track (at least 90s) so the model can sample the 'drop' at the 60s mark.")
                else:
                    # 2. Feature Extraction
                    features = extract_features_v5_inference(TEMP_FILE)
                    
                    if features is not None:
                        # 3. Load cached model assets
                        model, scaler, encoder = load_models()
                        
                        # 4. Scale features and run prediction
                        features_scaled = scaler.transform(features)
                        probs = model.predict_proba(features_scaled)[0]

                        # STORE IN SESSION STATE
                        st.session_state.prediction_results = {
                            'genre': encoder.classes_[np.argmax(probs)],
                            'confidence': np.max(probs),
                            'probs': probs,
                            'classes': encoder.classes_
                        }
            except FileNotFoundError:
                st.error("**Critical Error:** Model files (`.joblib`) not found. Please run training first.")
            except ValueError:
                st.error("**Dimension Mismatch:** The model expects a different number of features. Update your Scaler!")
            except Exception as e:
                st.error(f"**An unexpected error occurred:** {e}")
                        
    if st.session_state.prediction_results:
        res = st.session_state.prediction_results
        st.success(f"### Predicted Genre: **{res['genre'].upper()}**")
        
        if res['confidence'] < 0.65:
            st.warning(f"**Low Confidence ({res['confidence']:.1%}):** This track may be a hybrid (e.g. Tech-House) or an edge case.")
        else:
            st.info(f"Model Confidence: **{res['confidence']:.1%}**")
        
        chart_data = pd.DataFrame({'Genre': res['classes'], 'Confidence': res['probs']}).set_index('Genre')
        st.bar_chart(chart_data)

        # --- DOCUMENTATION: FEATURE LEGEND ---
        with st.expander("📖 Understanding the Analysis (Feature Legend)"):
            st.write("""
            Your track is analyzed across **59 distinct signal descriptors**. Here is what the model looks for:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Spectral Features (Brightness & Texture)**")
                st.write("- **Spectral Centroid:** The 'center of mass' of the sound. Higher values indicate 'brighter' sounds like EDM leads.")
                st.write("- **Spectral Rolloff:** The frequency below which 85% of the spectral energy lies. It distinguishes 'airy' synth textures from bass-heavy ones.")
                st.write("- **Spectral Flatness:** Measures how 'noise-like' a sound is vs. how 'tonal' it is. High flatness often signals aggressive percussion or white-noise risers.")
                
            with col2:
                st.markdown("**Timbral & Rhythmic Features**")
                st.write("- **MFCCs (13 coefficients):** These capture the 'shape' of the sound, similar to how human ears perceive timbre. Crucial for identifying specific synth types.")
                st.write("- **Chroma STFT:** Analyzes the harmonic content. This helps the model distinguish the melodic 'soul' of House from the mechanical loops of Techno.")
                st.write("- **RMS Energy:** The average volume (loudness). Used to detect high-energy 'drops' versus quiet ambient breakdowns.")

            st.info("💡 **Pro Tip:** The model samples a 30-second window starting at the 60-second mark to capture the track's core identity.")

        # --- FEEDBACK LOOP ---
        st.markdown("---")
        
        # Action Bar: Reset and Flagging
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🗑️ Clear Results"):
                st.session_state.prediction_results = None
                st.session_state.submitted = False
                st.rerun()

        with col2:
            # Checkbox with a key to prevent it from resetting randomly
            is_wrong = st.checkbox("🔍 Flag incorrect prediction", key="feedback_toggle")

        if is_wrong:
            correct_genre = st.selectbox("Select the correct genre:", res['classes'], key="correction_select")
            if st.button("Submit Correction", key="submit_btn"):
                save_feedback(uploaded_file.name, res['genre'], correct_genre)
                st.session_state.submitted = True
                st.rerun()

        # Persistent Success Message
        if st.session_state.get("submitted"):
            st.success("Thank you! Feedback saved for future model training.")

    # --- CLEANUP: Remove temp file after processing to free disk space ---
    if os.path.exists(TEMP_FILE):
        try:
            os.remove(TEMP_FILE)
        except Exception:
            pass