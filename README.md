# Milestone 3: ML Workflow Automation

**Student:** Josephine Chou (ichou4)  
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
