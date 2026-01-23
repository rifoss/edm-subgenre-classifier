import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="EDM Subgenre Classifier", page_icon="🎧", layout="wide")

# --- CSS Layout ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

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

# --- GENERATE WAVEFORM PLOT ---
def get_waveform_plot(y, sr):
    """Generates a matplotlib figure of the waveform."""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#1DB954')
    ax.set_title("Audio Waveform (30s Snippet)")
    ax.set_axis_off()
    return fig

# --- FEATURE EXTRACTION LOGIC ---
def extract_features_live(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, offset=60, duration=30)
        
        # BPM Detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        if bpm < 100: bpm = bpm * 2
        
        # Features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        features = {'tempo': float(bpm), 'spectral_centroid': float(centroid)}
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(mfccs_mean[i])
            features[f'mfcc_{i}_std'] = float(mfccs_std[i])
            
        feature_df = pd.DataFrame([features])
        scaled_features = scaler.transform(feature_df)
        
        return scaled_features, bpm, y, sr

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# -- SAVE FEEDBACK LOGIC
def save_feedback(filename, predicted, corrected):
    """Appends feedback to a local CSV file."""
    log_file = 'data/processed/feedback_log.csv'
    
    # Create the data row
    new_data = pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'predicted': predicted,
        'corrected': corrected
    }])
    
    # Check if file exists to decide if we need to write the header
    header_needed = not os.path.exists(log_file)
    
    # Append to CSV (mode='a' means append)
    new_data.to_csv(log_file, mode='a', index=False, header=header_needed)

# --- UI INTERFACE ---
st.title("EDM Subgenre Classifier")
st.markdown("Analyze your tracks with Artificial Intelligence.")

# Sidebar
with st.sidebar:
    st.header("Settings & Info")
    st.write("This model analyzes **Timbre** and **Tempo** to differentiate subgenres.")
    st.divider()
    st.markdown("### How to use:")
    st.write("1. Upload a track.")
    st.write("2. Listen to the preview.")
    st.write("3. Click **Classify** to see the prediction.")

# File Uploader
file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

if file:
    # Use columns for the upload preview and waveform
    col_up1, col_up2 = st.columns([1, 2])
    
    with col_up1:
        st.write("✅ **Track Loaded**")
        st.audio(file, format='audio/wav')
    
    if st.button("🚀 Run Classification", use_container_width=True):
        if model is not None:
            with st.spinner("Deep-scanning audio frequencies..."):
                try:
                    X_input, detected_bpm, y_audio, sr_audio = extract_features_live(file)
                    
                    # Prediction
                    pred_code = model.predict(X_input)[0]
                    prediction = encoder.inverse_transform([pred_code])[0]
                    probs = model.predict_proba(X_input)[0]
                    confidence = np.max(probs) * 100

                    # --- SAVE STATE FOR FEEDBACK ---
                    st.session_state['last_prediction'] = prediction
                    st.session_state['last_bpm'] = detected_bpm
                    st.session_state['last_filename'] = file.name
                    
                    # --- Results Dashboard ---
                    st.divider()
                    
                    # Top metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Genre", prediction.upper())
                    m2.metric("Confidence", f"{confidence:.1f}%")
                    m3.metric("Detected BPM", f"{detected_bpm:.1f}")
                    
                    # Lower Dashboard
                    res_col1, res_col2 = st.columns([2, 1])
                    
                    with res_col1:
                        st.pyplot(get_waveform_plot(y_audio, sr_audio))
                    
                    with res_col2:
                        st.write("**Genre Probability**")
                        chart_data = pd.DataFrame({
                            'Genre': encoder.classes_,
                            'Match': probs * 100
                        }).set_index('Genre')
                        st.bar_chart(chart_data)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# --- FEEDBACK LOGIC ---
if 'last_prediction' in st.session_state:
    st.divider()
    st.subheader("🛠️ Model Improvement Program")
    
    with st.expander("Is this prediction incorrect? Help us retrain."):
        st.write(f"The AI labeled this as **{st.session_state['last_prediction']}**.")
        
        correct_genre = st.selectbox(
            "What is the actual subgenre?", 
            options=["Techno", "House", "Dubstep", "Other / Not EDM"]
        )
        
        if st.button("Submit Correction"):
            # Call the save function
            save_feedback(
                st.session_state['last_filename'], 
                st.session_state['last_prediction'], 
                correct_genre
            )
            
            st.success("Correction saved to feedback_log.csv! We will use this for retraining.")
            
            # Optional: Clear the state so the feedback box closes/resets
            del st.session_state['last_prediction']