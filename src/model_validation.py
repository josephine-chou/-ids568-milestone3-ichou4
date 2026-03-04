"""Model validation - quality gate"""
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
    
    print(f"\nValidation Report")
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
        print("\nModel PASSED validation")
        client.set_tag(run_id, "validation_status", "passed")
        return True
    else:
        print("\nModel FAILED validation")
        client.set_tag(run_id, "validation_status", "failed")
        sys.exit(1)

if __name__ == "__main__":
    validate_latest_model()
