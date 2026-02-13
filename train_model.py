import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load the scaled data from Day 28
DATA_PATH = 'data/processed/features_scaled.csv'
MODEL_PATH = 'models/music_classifier.joblib'

if not os.path.exists('models'):
    os.makedirs('models')

def train_baseline():
    print("Starting Model Training (Day 29)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Did you run the preprocessing?")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 2. Split Features (X) and Labels (y)
    X = df.drop(columns=['label'])
    y = df['label']

    # 3. Train/Test Split (80% Train, 20% Test)
    # random_state=42 ensures the split is the same every time we run it
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Data Split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 4. Initialize and Train the Random Forest
    # n_estimators=100 means we are using 100 small decision trees working together
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 4. Initialize and Train the XGBoost Algorithm
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluate the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    
    # We load the encoder just to get the names back for the report
    encoder = joblib.load('data/processed/label_encoder.joblib')
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # 6. Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_baseline()