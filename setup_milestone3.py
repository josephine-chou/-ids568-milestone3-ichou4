#!/usr/bin/env python3
from pathlib import Path

def create_file(filepath, content):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(content)
    print(f"✓ {filepath}")

# preprocess.py
create_file('src/preprocess.py', '''"""Data preprocessing for Iris dataset"""
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
''')

# train.py
create_file('src/train.py', '''"""Model training with MLflow tracking"""
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
''')

# model_validation.py
create_file('src/model_validation.py', '''"""Model validation - quality gate"""
import sys
import mlflow
from mlflow.tracking import MlflowClient

MIN_ACCURACY = 0.90
MIN_F1_SCORE = 0.85

def validate_latest_model():
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name("iris_classification")
    if not experiment:
        print("Error: No experiment found")
        sys.exit(1)
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("Error: No runs found")
        sys.exit(1)
    
    run = runs[0]
    run_id = run.info.run_id
    metrics = run.data.metrics
    
    accuracy = metrics.get('test_accuracy', 0)
    f1_score = metrics.get('test_f1_weighted', 0)
    
    print(f"\\nValidation Report")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f} (threshold: {MIN_ACCURACY})")
    print(f"F1 Score: {f1_score:.4f} (threshold: {MIN_F1_SCORE})")
    
    passed = True
    
    if accuracy < MIN_ACCURACY:
        print("FAILED: Accuracy below threshold")
        passed = False
    else:
        print("PASSED: Accuracy meets threshold")
    
    if f1_score < MIN_F1_SCORE:
        print("FAILED: F1 Score below threshold")
        passed = False
    else:
        print("PASSED: F1 Score meets threshold")
    
    if passed:
        print("\\nModel PASSED validation")
        client.set_tag(run_id, "validation_status", "passed")
        return True
    else:
        print("\\nModel FAILED validation")
        client.set_tag(run_id, "validation_status", "failed")
        sys.exit(1)

if __name__ == "__main__":
    validate_latest_model()
''')

# Airflow DAG
create_file('dags/ml_pipeline.py', '''"""Airflow DAG for ML pipeline"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocess import preprocess_data
from train import train_model

def preprocess_task(**context):
    run_date = context['ds_nodash']
    result = preprocess_data(run_date=run_date)
    context['task_instance'].xcom_push(key='run_date', value=run_date)
    return result

def train_task(n_estimators, max_depth, **context):
    run_date = context['task_instance'].xcom_pull(
        task_ids='preprocess_data', 
        key='run_date'
    )
    return train_model(
        run_date=run_date,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

default_args = {
    'owner': 'mlops_student',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='ML Pipeline: Preprocess -> Train (5 experiments)',
    schedule_interval=None,
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=['milestone3', 'iris'],
) as dag:
    
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_task,
        provide_context=True,
    )
    
    train_baseline = PythonOperator(
        task_id='train_baseline',
        python_callable=train_task,
        op_kwargs={'n_estimators': 50, 'max_depth': 5},
        provide_context=True,
    )
    
    train_exp2 = PythonOperator(
        task_id='train_exp2',
        python_callable=train_task,
        op_kwargs={'n_estimators': 100, 'max_depth': 5},
        provide_context=True,
    )
    
    train_exp3 = PythonOperator(
        task_id='train_exp3',
        python_callable=train_task,
        op_kwargs={'n_estimators': 100, 'max_depth': 10},
        provide_context=True,
    )
    
    train_exp4 = PythonOperator(
        task_id='train_exp4',
        python_callable=train_task,
        op_kwargs={'n_estimators': 150, 'max_depth': 8},
        provide_context=True,
    )
    
    train_exp5 = PythonOperator(
        task_id='train_exp5',
        python_callable=train_task,
        op_kwargs={'n_estimators': 200, 'max_depth': 8},
        provide_context=True,
    )
    
    preprocess >> [train_baseline, train_exp2, train_exp3, train_exp4, train_exp5]
''')

# GitHub Actions
create_file('.github/workflows/ci.yml', '''name: ML Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  train-and-validate:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python 3.12
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Preprocess data
      run: python src/preprocess.py
    
    - name: Train model
      run: |
        RUN_DATE=$(ls data/processed/train_*.csv | head -1 | sed 's/.*train_\\(.*\\)\\.csv/\\1/')
        python src/train.py --run-date $RUN_DATE
    
    - name: Validate model
      run: python src/model_validation.py
    
    - name: Upload MLflow artifacts
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: mlflow-artifacts
        path: mlruns/
        retention-days: 30
''')

# README
create_file('README.md', '''# Milestone 3: ML Workflow Automation

**Student:** Iris Chou (ichou4)  
**Course:** IDS 568 Spring 2026

## Quick Start
```bash
pip install -r requirements.txt
python src/preprocess.py
```

## Run with Airflow
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# Terminal 1
airflow webserver --port 8080

# Terminal 2
airflow scheduler
```

## View MLflow
```bash
mlflow ui --port 5000
```
''')

# Lineage Report
create_file('lineage_report.md', '''# Lineage Report

**Author:** Iris Chou (ichou4)  
**Date:** March 2026

## Experiments

| Experiment | n_estimators | max_depth | Accuracy | F1 Score |
|------------|--------------|-----------|----------|----------|
| Baseline   | 50           | 5         |          |          |
| Exp 2      | 100          | 5         |          |          |
| Exp 3      | 100          | 10        |          |          |
| Exp 4      | 150          | 8         |          |          |
| Exp 5      | 200          | 8         |          |          |

## Production Candidate

[Fill after experiments]
''')

print("\\n✅ All files created")
