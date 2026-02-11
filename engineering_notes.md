Engineering Note - Best Parameters for 250 Samples

Colsample_bytree: 1.0
Learning_rate: 0.2
Max_depth: 3 - This means simpler model performed better
n_estimators: 300 - We hit the ceiling, more estimators could help, but slightly
subsample: 0.8

Accuracy: 80%






Engineering Note: Solving the Techno/House Overlap

The Problem: The "Tempo Trap"

During Phase 2 of the EDM Subgenre Classifier project, the model reached a performance plateau of 72.2% accuracy. A deep dive into the confusion matrix revealed a significant "feature gap": the model was consistently mislabeling Slow Techno (124–126 BPM) as House.

Because the baseline model relied heavily on tempo and basic spectral centroids, it could not distinguish between a melodic House track and a driving, percussive Techno track when their BPMs were identical.

The Pivot: Moving Beyond Surface Features

To solve this, I implemented an advanced feature extraction pivot (Day 41–44) focused on Timbre and Rhythmic Density rather than just speed. The key addition was Harmonic-Percussive Source Separation (HPSS).

Percussive-to-Harmonic Ratio: By using librosa.effects.hpss to separate the audio into percussive (drums) and harmonic (melody) components, I calculated the ratio of drum energy to melodic energy. This served as the "Magic Bullet," as Techno is mathematically more percussive-heavy than its House counterparts.

Spectral Flatness: Added to measure the "noise-like" quality of a sound, effectively capturing the industrial textures of Techno versus the tonal clarity of House.

Zero Crossing Rate (ZCR): Integrated to identify the aggressive, distorted transients common in Techno kicks.

The Result: Quantitative Improvement

By expanding the feature vector from 14 to 31 dimensions (incorporating both Means and Standard Deviations for MFCCs) and upgrading the engine to XGBoost, the model achieved:

Total Accuracy Increase: 72.2% → 78.0%

Techno Precision/F1-Score: 0.44 → 0.60

System Robustness: The model successfully identified "Slow Techno" tracks that had previously failed, proving that rhythmic texture is a more reliable subgenre indicator than BPM alone.

Key Takeaway

This pivot demonstrated that in audio machine learning, "domain knowledge" (understanding the rhythmic difference between genres) is just as important as the algorithm itself. Sophisticated signal processing can reveal patterns that raw metadata cannot.