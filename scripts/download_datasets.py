import urllib.request
import os
import pandas as pd

datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
os.makedirs(datasets_dir, exist_ok=True)

import ssl

# Titanic dataset from Stanford CS
titanic_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_path = os.path.join(datasets_dir, "titanic.csv")

if not os.path.exists(titanic_path):
    print("Downloading Titanic dataset...")
    try:
        # Create unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(titanic_url, titanic_path)
        print(f"✅ Saved to {titanic_path}")
    except Exception as e:
        print(f"❌ Failed to download Titanic dataset via urllib: {e}")
        # Try curl as fallback
        try:
            print("Trying curl fallback...")
            os.system(f"curl -k -o {titanic_path} {titanic_url}")
            if os.path.exists(titanic_path):
                 print(f"✅ Saved to {titanic_path} via curl")
        except Exception as e2:
             print(f"❌ Curl fallback failed: {e2}")
else:
    print(f"✅ Titanic already exists at {titanic_path}")

# Verify downloads
print("\nVerifying datasets:")
for name, path in [("Titanic", titanic_path)]:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"\n{name}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
             print(f"❌ Error reading {name}: {e}")
    else:
        print(f"❌ {name} not found at {path}")
