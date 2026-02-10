import pandas as pd
df = pd.read_csv('data/processed/features_250.csv')
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(df['genre'].value_counts())

print(df.isnull().sum().sum())