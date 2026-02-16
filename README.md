🎧 EDM Subgenre Classifier (v5.0)

A production-grade machine learning application that classifies EDM tracks into Techno, House, and Dubstep with 84.5% accuracy. This project demonstrates the full MLOps lifecycle: from raw audio signal processing to cloud deployment via Streamlit.

🚀 Live Demo
https://edm-subgenre-classifier-4dbeqtnwnu6vvugjqpfcud.streamlit.app/

📊 Performance Summary

Overall Accuracy: 83.8%

Features: 59-dimensional vector including Spectral Contrast, MFCCs (Mean/Std), and Chroma STFT.

Dataset: 250 tracks augmented to 500 samples via multi-offset extraction.

🧠 Technical Highlight: The "Techno Variance Fix"

The core challenge was distinguishing "Slow Techno" (124-126 BPM) from House. I broke the 80% accuracy barrier by moving from simple metadata to high-dimensional signal processing:

Texture Variance: Added the Standard Deviation of MFCCs to capture the industrial, shifting textures unique to Techno.

Harmonic-Percussive Separation: Used HPSS to calculate a percussive-to-harmonic ratio, identifying drum-heavy profiles regardless of BPM.

Data Augmentation: Doubled the training set to 500 samples to resolve the "Curse of Dimensionality" caused by the expanded feature set.

📂 Project Architecture

├── app.py                  # Streamlit UI & Inference Engine
├── batch_extractor_v3.py   # 59-feature signal processing (Librosa)
├── data_preprocessing.py   # Data scaling and Label Encoding
├── train_model.py          # Tuned XGBoost training script
├── data/
│   └── processed/          # Scaled CSVs and .joblib transformers
└── models/                 # Saved music_classifier.joblib


💻 Installation & Usage

Clone the repo: git clone https://github.com/your-username/edm-classifier.git
Install dependencies: pip install -r requirements.txt
Run the App: streamlit run app.py

🛠️ Tech Stack

Audio Processing: Librosa, NumPy
Machine Learning: XGBoost, Scikit-learn
Data Handling: Pandas, Joblib
Frontend: Streamlit

👨‍💻 Connect with the Developer

Developed by Jody Suryatna - Machine Learning and Data Hobbyist

If you have questions about the Techno Variance Fix or the HPSS implementation used in this project, feel free to reach out!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jody-suryatna/)
[![Portfolio](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/rifoss)