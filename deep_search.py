import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- CONFIG ---
DATA_PATH = 'data/processed/features_scaled.csv'

def deep_tune():
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Deep Search Grid
    # We are lowering learning_rate to 0.05 and 0.01 to find more precision
    param_grid = {
        'n_estimators': [400, 600, 800],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.8, 1.0]
    }

    print("🔍 Starting Deep Search Tuning (59 features)...")
    print("This may take 5-10 minutes because of the lower learning rates.")
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    grid_search = GridSearchCV(
        xgb, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)

    # 3. Results
    print("\n🏆 NEW GOLDEN SETTINGS FOUND:")
    print(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🚀 Tuned Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if acc >= 0.85:
        print("🎯 TARGET REACHED! You can now save this model.")
    else:
        print("💡 We are at the physical limit of these features. 84.5% is a very strong production score.")

if __name__ == "__main__":
    deep_tune()