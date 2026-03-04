"""Airflow DAG for ML pipeline"""
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
    schedule=None,
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=['milestone3', 'iris'],
) as dag:
    
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_task,
    )
    
    train_baseline = PythonOperator(
        task_id='train_baseline',
        python_callable=train_task,
        op_kwargs={'n_estimators': 50, 'max_depth': 5},
    )
    
    train_exp2 = PythonOperator(
        task_id='train_exp2',
        python_callable=train_task,
        op_kwargs={'n_estimators': 100, 'max_depth': 5},
    )
    
    train_exp3 = PythonOperator(
        task_id='train_exp3',
        python_callable=train_task,
        op_kwargs={'n_estimators': 100, 'max_depth': 10},
    )
    
    train_exp4 = PythonOperator(
        task_id='train_exp4',
        python_callable=train_task,
        op_kwargs={'n_estimators': 150, 'max_depth': 8},
    )
    
    train_exp5 = PythonOperator(
        task_id='train_exp5',
        python_callable=train_task,
        op_kwargs={'n_estimators': 200, 'max_depth': 8},
    )
    
    preprocess >> [train_baseline, train_exp2, train_exp3, train_exp4, train_exp5]
