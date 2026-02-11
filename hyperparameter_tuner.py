import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
FEATURES_PATH = 'data/processed/features_250.csv'

def tune_model():
    # 1. Load Dataset
    print("Loading 250-track dataset...")
    df = pd.read_csv(FEATURES_PATH)
    
    # 2. Preprocess
    X = df.drop('genre', axis=1)
    y = df['genre']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 3. Define Parameter Grid
    # We focus on depth and learning rate to fix the House/Techno overlap
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print("Starting Grid Search (this may take a minute)...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    # cv=5 means it will test each combination 5 different ways
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy', 
        verbose=1, 
        n_jobs=-1 # Uses all CPU cores
    )
    
    grid_search.fit(X_train, y_train)
    
    # 4. Results
    print("\nBEST PARAMETERS FOUND:")
    print(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nFINAL PERFORMANCE WITH TUNED SETTINGS:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Suggestion for next step
    if accuracy_score(y_test, y_pred) > 0.82:
        print("Excellent! You've broken the 80% barrier.")
    else:
        print("Accuracy is steady. We may need to look at feature engineering on Day 49.")

if __name__ == "__main__":
    tune_model()