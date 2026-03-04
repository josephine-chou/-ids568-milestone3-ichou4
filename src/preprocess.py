"""Data preprocessing for Iris dataset"""
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import hashlib

def compute_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def preprocess_data(run_date=None, test_size=0.2, random_state=42):
    if run_date is None:
        run_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print(f"Preprocessing data (run_date: {run_date})")
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_dir / f"train_{run_date}.csv"
    test_path = processed_dir / f"test_{run_date}.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / f"scaler_{run_date}.pkl"
    joblib.dump(scaler, scaler_path)
    
    data_hash = compute_hash(train_path)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Data hash: {data_hash[:16]}")
    
    return {
        'run_date': run_date,
        'train_path': str(train_path),
        'test_path': str(test_path),
        'data_hash': data_hash[:16]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-date', type=str, default=None)
    args = parser.parse_args()
    preprocess_data(run_date=args.run_date)
