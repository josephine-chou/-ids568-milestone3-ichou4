#!/usr/bin/env python3
import os
from pathlib import Path

files = {
    'requirements.txt': '''scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
apache-airflow==2.10.4
mlflow==2.18.0
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
python-dotenv==1.0.1
pytest==8.3.3
''',
    
    '.gitignore': '''__pycache__/
*.pyc
venv/
ENV/
airflow.db
airflow.cfg
logs/
mlruns/
models/*.pkl
data/raw/*.csv
data/processed/*.csv
.DS_Store
.env
''',
}

for filepath, content in files.items():
    Path(filepath).write_text(content)
    print(f"✓ {filepath}")

print("\n✓ Basic files created!")
print("Now downloading remaining files from GitHub...")
