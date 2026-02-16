Case Study: Breaking the 80% Accuracy Barrier in Audio Classification

1. Executive Summary

This project aimed to build a production-grade machine learning classifier to distinguish between three major EDM subgenres: Techno, House, and Dubstep. While initial baselines reached 72% accuracy, a significant performance plateau occurred due to the "Tempo Trap"—where slow Techno tracks were mathematically indistinguishable from House. Through advanced feature engineering (HPSS) and strategic data augmentation, the final model achieved 84.5% overall accuracy and a 79% F1-score for Techno.

2. The Challenge: The "Tempo Trap"

Early versions of the model relied heavily on BPM and basic spectral centroids. This led to a critical failure:

The Overlap: Professional House music typically sits at 124–128 BPM. "Slow Techno" (Melodic or Deep Techno) often occupies the exact same tempo range.

The Result: The model became "lazy," defaulting to House for any track in that BPM range. Techno precision plateaued at 60%, rendering the model unreliable for industrial use.

3. Engineering Phase 1: The Variance Fix

The first major breakthrough involved moving from "Average" features to "Texture" features.

The Insight: Techno is defined by its industrial textures and rhythmic "swing," which are more volatile than the steady loops of House.

The Implementation: I expanded the feature vector to include the Standard Deviation of MFCCs and Spectral Contrast across 30-second clips.

Impact: This allowed the model to "hear" the gritty, shifting textures of Techno, providing the first clear differentiator from the tonal consistency of House.

4. Engineering Phase 2: HPSS Integration

To further decouple the model from its reliance on BPM, I implemented Harmonic-Percussive Source Separation (HPSS).

Method: By separating audio into percussive (drums) and harmonic (melodic) layers, I calculated the Percussive-to-Harmonic Ratio.

Result: Even if a Techno track and a House track share a BPM of 126, the Techno track mathematically displays a higher percussive energy ratio. This pushed Techno precision from 0.44 to 0.60.

5. Overcoming the "Curse of Dimensionality"

As the feature set grew to 59 dimensions, accuracy paradoxically dropped to 76%. With only 250 tracks, the model began overfitting—memorizing specific songs rather than learning general rules.

The Solution: I implemented a custom Data Augmentation pipeline. Instead of one 30-second sample per track, the extractor pulled two distinct samples (at the 45s and 120s marks).

Impact: Doubling the dataset to 500 samples provided the statistical density needed for the high-dimensional model to generalize. Accuracy surged to 84.5%.

6. Final Technical Stack

Engine: XGBoost (Gradient Boosted Trees)

Audio Processing: Librosa (HPSS, MFCCs, Spectral Contrast, Chroma STFT)

Deployment: Streamlit Cloud with real-time confidence visualizers.

Optimization: GridSearchCV for hyperparameter tuning.

7. Conclusion

The project highlights that in audio machine learning, "Domain Knowledge" (understanding rhythmic structure) is superior to "Algorithm Brute Force." By engineering features that specifically targeted the model's confusion, I successfully delivered a tool capable of professional-grade classification in a highly nuanced musical space.