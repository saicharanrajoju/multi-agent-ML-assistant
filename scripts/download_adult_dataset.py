import pandas as pd
import numpy as np
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    print(f"Downloading dataset from {url}...")
    # The dataset has leading spaces for categorical column values.
    # We use skipinitialspace=False to exactly match the problem description (" ?"),
    # or True if we want to clean it up early. We'll set skipinitialspace=False so that " ?" is explicitly what's replaced.
    df = pd.read_csv(url, header=None, names=columns)
    
    # Replace " ?" with NaN to simulate the "encoded nulls" problem trap
    df.replace(" ?", np.nan, inplace=True)
    
    # If any space variations exist
    df.replace("?", np.nan, inplace=True)
    
    # Ensure datasets folder exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, "datasets", "adult_income.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save the dataframe
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}\n")
    
    print("--- Dataset Information ---")
    print(f"Shape: {df.shape}")
    
    print("\n--- Class Distribution (income) ---")
    print("Counts:")
    print(df['income'].value_counts(dropna=False))
    print("\nPercentage:")
    print(df['income'].value_counts(normalize=True, dropna=False) * 100)
    
    print("\n--- Null Counts Per Column ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
