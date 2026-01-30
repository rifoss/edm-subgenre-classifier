import librosa
import numpy as np
import pandas as pd

# --- FILES TO COMPARE ---
# Pick one House track that the model gets RIGHT
house_path = 'data/raw/house/Disco Lines & Tinashe - No Broke Boys (Official Audio).mp3'
# Pick one Slow Techno track that the model gets WRONG (The "Tempo Trap" track)
techno_path = 'data/raw/techno/Zeli - Only The Fallen [NCS Release].mp3'

def get_gap_metrics(path, label):
    y, sr = librosa.load(path, offset=60, duration=30)
    
    # 1. Existing features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 2. NEW TEST 1: Spectral Flatness 
    # (High = Noise/Techno-like, Low = Tonal/House-like)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # 3. NEW TEST 2: Pulse Clarity 
    # (How steady is the 4/4 grid? Techno is usually a rigid machine)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse_clarity = np.std(librosa.beat.plp(onset_envelope=onset_env, sr=sr))
    
    # 4. NEW TEST 3: Percussive-to-Harmonic Ratio
    # We split the audio into 'Drums' and 'Melody'
    harmonic, percussive = librosa.effects.hpss(y)
    per_har_ratio = np.mean(percussive) / np.mean(harmonic)
    
    return {
        'Genre': label,
        'BPM': tempo,
        'Flatness (Noise)': flatness,
        'Pulse Clarity': pulse_clarity,
        'Perc/Har Ratio': per_har_ratio
    }

print("Pivot Analysis: Testing Rhythmic & Harmonic Features...")

try:
    house_stats = get_gap_metrics(house_path, "House (Correct)")
    techno_stats = get_gap_metrics(techno_path, "Techno (Failed)")

    df = pd.DataFrame([house_stats, techno_stats])
    print("\n--- Comparison Table ---")
    print(df.to_string(index=False))

    print("\nNEW ANALYSIS:")
    # If the previous spectral features failed, we look for 'Robotic' features:
    if techno_stats['Pulse Clarity'] > house_stats['Pulse Clarity']:
        print("-> SUCCESS: Pulse Clarity is higher for Techno. The 'Robotic Grid' is the differentiator.")
    
    if techno_stats['Perc/Har Ratio'] > house_stats['Perc/Har Ratio']:
        print("-> SUCCESS: Techno has a higher Drum-to-Melody ratio. This fixes the overlap.")

except Exception as e:
    print(f"Error: {e}. Check your file paths!")