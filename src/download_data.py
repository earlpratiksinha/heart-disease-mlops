import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def download_heart_disease_data():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    print("Fetching Heart Disease dataset from UCI...")
    
    # Fetch dataset (ID 45 is the Heart Disease dataset)
    heart_disease = fetch_ucirepo(id=45)

    # Extract features and targets as pandas dataframes
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine them into a single dataframe for easy saving and EDA
    df = pd.concat([X, y], axis=1)

    # Save to the data folder
    output_path = 'data/heart_disease_raw.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully downloaded and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    download_heart_disease_data()