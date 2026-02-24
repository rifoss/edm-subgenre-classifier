# 🎧 EDM Subgenre Classifier

A machine learning web application that classifies EDM tracks into subgenres (Techno, House, Dubstep) using high-dimensional audio feature extraction. Built with Streamlit and deployed on Streamlit Cloud with a persistent feedback loop powered by Supabase.

**🔗 Live Demo:** [edm-subgenre-classifier-4dbeqtnwnu6vvugjqpfcud.streamlit.app](https://edm-subgenre-classifier-4dbeqtnwnu6vvugjqpfcud.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Tech Stack](#tech-stack)
- [Supabase Feedback Integration](#supabase-feedback-integration)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Known Limitations](#known-limitations)
- [Future Roadmap](#future-roadmap)

---

## Overview

This app accepts an MP3 or WAV upload and returns a predicted EDM subgenre along with a confidence score and per-class probability breakdown. It was built to explore the application of signal processing and supervised learning to music classification — a domain where subtle timbral and rhythmic differences between subgenres make classification non-trivial.

The classifier achieves **83.8% accuracy** on held-out test data using a model trained on augmented audio features extracted from a curated dataset of Techno, House, and Dubstep tracks.

---

## How It Works

1. The user uploads an MP3 or WAV file (minimum 90 seconds)
2. The app uses [Mutagen](https://mutagen.readthedocs.io/) to validate file duration before processing
3. [Librosa](https://librosa.org/) loads a **30-second window starting at the 60-second mark** — targeting the track's drop or core identity section
4. 59 audio features are extracted from this window (see [Feature Engineering](#feature-engineering))
5. Features are scaled using a pre-fitted `StandardScaler` and passed to the trained classifier
6. The app returns the predicted subgenre, confidence score, and a probability bar chart

---

## Feature Engineering

The model uses **59 signal descriptors** extracted per track, grouped into five categories:

| Category | Features | Count |
|---|---|---|
| Rhythmic | Tempo (BPM) | 1 |
| Spectral | Centroid, Rolloff, Flatness, ZCR, RMS | 5 |
| Harmonic/Percussive | HPSS ratio (percussive/harmonic mean) | 1 |
| Spectral Contrast | Mean + Std across 7 bands | 14 |
| MFCCs | Mean + Std of 13 coefficients | 26 |
| Chroma STFT | Mean across 12 pitch classes | 12 |

**Total: 59 features**

All features are extracted using `librosa` with a sample rate of 22,050 Hz and the `soxr_qq` resampler for memory efficiency on cloud infrastructure.

---

## Model Architecture

- **Algorithm:** XGBoost Classifier
- **Version:** 5.0 (Augmented Dataset)
- **Accuracy:** 83.8% on held-out test set
- **Classes:** Techno, House, Dubstep
- **Preprocessing:** StandardScaler (fitted on training data, serialized as `scaler.joblib`)
- **Label Encoding:** LabelEncoder (serialized as `label_encoder.joblib`)
- **Serialization:** All model assets saved via `joblib`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Audio Processing | librosa, soxr, mutagen |
| Machine Learning | XGBoost, scikit-learn, numpy |
| Web Application | Streamlit |
| Feedback Persistence | Supabase (PostgreSQL) |
| Deployment | Streamlit Cloud |
| System Dependencies | ffmpeg, libsndfile1 |

---

## Supabase Feedback Integration

The app includes a user feedback loop designed to support continuous model improvement. When a user flags an incorrect prediction, the correction is inserted into a Supabase PostgreSQL table with the following schema:

```sql
CREATE TABLE feedback (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    filename TEXT,
    predicted_genre TEXT,
    corrected_genre TEXT
);
```

This data persists across deployments and redeployments — unlike local CSV logging which is wiped on every Streamlit Cloud redeploy. Collected feedback is intended to be used as a correction signal for future model retraining cycles.

Supabase credentials are stored securely using Streamlit Cloud's Secrets Manager and are never exposed in source code.

---

## Project Structure

```
edm-subgenre-classifier/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (ffmpeg, libsndfile1)
│
├── models/
│   └── music_classifier.joblib     # Trained XGBoost model
│
└── data/
    └── processed/
        ├── scaler.joblib           # Fitted StandardScaler
        └── label_encoder.joblib    # Fitted LabelEncoder
```

---

## Local Setup

### Prerequisites
- Python 3.9+
- ffmpeg installed on your system (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux)

### Installation

```bash
# Clone the repository
git clone https://github.com/rifoss/edm-subgenre-classifier.git
cd edm-subgenre-classifier

# Install dependencies
pip install -r requirements.txt
```

### Supabase Configuration (optional for local feedback logging)

Create a `.streamlit/secrets.toml` file in the project root:

```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"
```

### Run the app

```bash
streamlit run app.py
```

---

## Known Limitations

- **Fixed sampling window:** The model always samples from the 60-second mark. Tracks where the drop occurs significantly earlier or later may yield lower confidence scores or misclassifications.
- **Three-class scope:** The current model only classifies Techno, House, and Dubstep. Hybrid subgenres (e.g. Tech-House, Bass House) or adjacent genres (e.g. Drum & Bass, Trance) will be forced into one of the three classes.
- **Minimum duration requirement:** Tracks under 90 seconds cannot be processed, which excludes edits, intros, and short-form content.
- **MP3/WAV only:** Other audio formats (FLAC, AAC, OGG) are not currently supported.
- **Feedback loop is passive:** Collected corrections improve future models but do not affect the current deployed model in real time.

---

## Future Roadmap

- **Expand subgenre coverage** — add Drum & Bass, Trance, and Ambient Techno classes with additional training data
- **Dynamic sampling window** — detect the track's drop automatically using onset strength analysis rather than a fixed 60s offset
- **WAV support improvements** — dynamic file extension handling already partially implemented; full format support planned
- **Automated retraining pipeline** — use accumulated Supabase feedback data to trigger periodic model retraining via a scheduled script
- **Waveform/spectrogram visualization** — display the analyzed audio segment visually to give users insight into what the model is processing
- **Confidence calibration** — investigate Platt scaling or isotonic regression to improve the reliability of probability outputs

---

## Developer

**Jody Suryatna**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jody-suryatna/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?style=for-the-badge&logo=github)](https://github.com/rifoss/edm-subgenre-classifier)
