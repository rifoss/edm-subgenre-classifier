import os

# --- CONFIGURATION ---
DATA_PATH = 'data/raw'
GENRES = ['techno', 'house', 'dubstep']
TARGET_TOTAL = 250
TARGET_PER_GENRE = TARGET_TOTAL // len(GENRES)

def audit_dataset():
    print("📊 --- EDM Dataset Audit ---")
    total_found = 0
    counts = {}

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} directory not found.")
        return

    for genre in GENRES:
        genre_path = os.path.join(DATA_PATH, genre)
        if os.path.exists(genre_path):
            # Count mp3 and wav files
            files = [f for f in os.listdir(genre_path) if f.lower().endswith(('.mp3', '.wav'))]
            count = len(files)
            counts[genre] = count
            total_found += count
        else:
            counts[genre] = 0
            print(f"⚠️ Warning: Folder for {genre} is missing.")

    # Display Results
    for genre, count in counts.items():
        diff = TARGET_PER_GENRE - count
        status = "✅ TARGET MET" if diff <= 0 else f"📉 NEED {diff} MORE"
        print(f"{genre.capitalize():<10}: {count:>3} tracks | {status}")

    print("-" * 30)
    print(f"TOTAL TRACKS: {total_found} / {TARGET_TOTAL}")
    
    if total_found < TARGET_TOTAL:
        print(f"🚀 Goal: You need {TARGET_TOTAL - total_found} more tracks to hit the Week 11 milestone.")
    else:
        print("🔥 Milestone reached! You are ready for mass feature extraction.")

if __name__ == "__main__":
    audit_dataset()