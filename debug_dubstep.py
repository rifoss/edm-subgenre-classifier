import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Load the latest scaled data
df = pd.read_csv('data/processed/features_scaled.csv')
X = df.drop(columns=['genre'])
y = df['genre']

# Load the model and encoders
model = joblib.load('models/baseline_model.joblib')
encoder = joblib.load('data/processed/label_encoder.joblib')

# Let's look at the WHOLE dataset predictions, not just the test set
# This helps us see if the model learned the training data at least
y_all_pred = model.predict(X)

print("--- FULL DATASET CHECK (Training + Test) ---")
print(classification_report(y, y_all_pred, target_names=encoder.classes_))

# Let's find exactly which files are being misclassified
raw_df = pd.read_csv('data/processed/features.csv')
raw_df['predicted_code'] = y_all_pred
raw_df['predicted_genre'] = encoder.inverse_transform(y_all_pred)

# Show me the Dubstep tracks that failed
failures = raw_df[(raw_df['genre'] == 'dubstep') & (raw_df['genre'] != raw_df['predicted_genre'])]

print(f"\n--- MISCLASSIFIED DUBSTEP TRACKS ({len(failures)} total) ---")
if len(failures) > 0:
    print(failures[['filename', 'tempo', 'predicted_genre']])
else:
    print("None! The model has memorized all Dubstep tracks perfectly.")

print("\n--- ANALYSIS ---")
print("1. If 'Full Dataset' score is high but 'Test' score is 0: You have an Overfitting problem.")
print("2. If 'Full Dataset' score is 0 for Dubstep: Your features (MFCCs) are identical to Techno.")