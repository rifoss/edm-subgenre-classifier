import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix

# 1. Load Data
DATA_PATH = 'data/processed/features_scaled.csv'
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['genre'])
y = df['genre']

# Load Label Encoder for genre names
encoder = joblib.load('data/processed/label_encoder.joblib')

def analyze():
    # 2. Cross-Validation
    # This gives us a better idea of accuracy than a single split
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    print(f"5-Fold Cross-Validation Scores: {cv_scores}")
    print(f"Average Accuracy: {np.mean(cv_scores) * 100:.2f}%")

    # 3. Feature Importance
    model.fit(X, y) # Train on full set to see importance
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
    plt.title('Top 10 Most Important Features')
    plt.show()

    # 4. Confusion Matrix (Why did Dubstep fail?)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Where is the model confused?')
    plt.show()

if __name__ == "__main__":
    analyze()