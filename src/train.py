"""Model training with MLflow tracking"""
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import mlflow
import mlflow.sklearn
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def compute_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()

def train_model(run_date, n_estimators=100, max_depth=5, criterion='gini'):
    print(f"Training model (n_estimators={n_estimators}, max_depth={max_depth})")
    
    processed_dir = Path("data/processed")
    train_path = processed_dir / f"train_{run_date}.csv"
    test_path = processed_dir / f"test_{run_date}.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    mlflow.set_experiment("iris_classification")
    
    with mlflow.start_run() as run:
        mlflow.log_param('run_date', run_date)
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('criterion', criterion)
        mlflow.log_param('random_state', 42)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
        }
        
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
            mlflow.log_metric(name, value)
        
        models_dir = Path("models")
        model_path = models_dir / f"model_{run_date}.pkl"
        joblib.dump(model, model_path)
        
        model_hash = compute_hash(model_path)
        mlflow.log_param('model_hash', model_hash[:16])
        
        cm_path = models_dir / f"cm_{run_date}.png"
        plot_confusion_matrix(y_test, y_test_pred, cm_path)
        
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(model, "model", registered_model_name="iris_classifier")
        
        run_id = run.info.run_id
        
    print(f"Run ID: {run_id}")
    return {'run_id': run_id, 'metrics': metrics}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-date', required=True)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    args = parser.parse_args()
    
    train_model(
        run_date=args.run_date,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
