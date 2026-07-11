import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def download_heart_disease_data():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    print("Fetching Heart Disease dataset from UCI...")
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)

    # Save raw data
    df.to_csv('data/heart_disease_raw.csv', index=False)
    
    # --- Data Cleaning ---
    target_col = df.columns[-1]
    df['target'] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
    if target_col != 'target':
        df = df.drop(columns=[target_col])
        
    # Fill missing values with median
    df = df.fillna(df.median())

    # Save cleaned data
    output_path = 'data/heart_disease_cleaned.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned dataset successfully saved to {output_path}")

if __name__ == "__main__":
    download_heart_disease_data()