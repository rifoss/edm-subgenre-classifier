import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIG ---
# Use the SCALED data we just created
DATA_PATH = 'data/processed/features_scaled.csv'

def tune_v2():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Expanded Grid: Testing deeper trees for the new complex features
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [3, 6, 9], 
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    print("Tuning model for 38 features... (This might take 2-3 minutes)")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\nNEW BEST PARAMETERS:")
    print(grid_search.best_params_)
    
    y_pred = grid_search.best_estimator_.predict(X_test)
    print(f"\nNew Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nTechno Check (Look at class 0 or 2 depending on encoding):")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    tune_v2()