"""
Random Forest Baseline Model for Bank Loan Evaluation Academic Research.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Loads preprocessed training (2,400 samples) and testing (600 samples) data from data/processed/.
- Trains a reproducible Random Forest Classifier (Baseline Model) on the 3,000-record dataset.
- Evaluates the baseline model against standard academic performance metrics.
- Serializes the trained baseline model artifact to models/baseline/random_forest.joblib.
- Persists detailed evaluation metrics, confusion matrix, and sample-level predictions to evaluation/.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# Baseline Hyperparameters & Configuration
# -----------------------------------------------------------------------------

RANDOM_STATE: int = 42
N_ESTIMATORS: int = 200
CLASS_WEIGHT: str = "balanced"
N_JOBS: int = -1
DATASET_NAME: str = "loan_evaluation_dataset_3000.csv (500 Seed + 2,500 SDV CTGAN Synthetic)"


def resolve_paths() -> Tuple[Path, Path, Path]:
    """Resolve paths to processed data, model artifacts, and evaluation directories."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    data_dir = workspace_root / "data" / "processed"
    models_dir = workspace_root / "models" / "baseline"
    eval_dir = workspace_root / "evaluation"

    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, models_dir, eval_dir


def load_processed_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed feature matrices and target vectors."""
    x_train_path = data_dir / "X_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"

    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required processed dataset not found: {path}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Instantiate and train the reproducible Random Forest baseline model."""
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, train_size: int
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Calculate academic evaluation metrics for classification."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    report_dict = classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"], output_dict=True)

    metrics = {
        "dataset_name": DATASET_NAME,
        "model_name": "Random Forest Baseline",
        "training_size": int(train_size),
        "test_size": int(len(y_test)),
        "num_features": int(X_test.shape[1]),
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "class_weight": CLASS_WEIGHT,
            "n_jobs": N_JOBS,
        },
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": report_dict,
    }

    return metrics, y_pred, y_proba, cm


def save_artifacts(
    model: RandomForestClassifier,
    metrics: Dict[str, Any],
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    cm: np.ndarray,
    models_dir: Path,
    eval_dir: Path,
) -> None:
    """Save trained model, metrics JSON, confusion matrix CSV, and test predictions CSV."""
    # 1. Model artifact
    model_path = models_dir / "random_forest.joblib"
    joblib.dump(model, model_path)

    # 2. Evaluation results JSON
    results_path = eval_dir / "random_forest_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 3. Confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Rejected (0)", "Actual_Approved (1)"],
        columns=["Predicted_Rejected (0)", "Predicted_Approved (1)"],
    )
    cm_path = eval_dir / "random_forest_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8")

    # 4. Predictions CSV
    predictions_df = pd.DataFrame({
        "Actual Label": y_test.values,
        "Predicted Label": y_pred,
        "Predicted Probability": np.round(y_proba, 4),
    })
    predictions_path = eval_dir / "random_forest_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")

    print(f"    [OK] Model Saved: {model_path.name}")
    print(f"    [OK] Metrics Saved: {results_path.name}")
    print(f"    [OK] Confusion Matrix Saved: {cm_path.name}")
    print(f"    [OK] Predictions Saved: {predictions_path.name}")


def run_random_forest_pipeline() -> None:
    """Execute complete Random Forest baseline training and evaluation."""
    print("=" * 80)
    print("REVISED STEP 4: RANDOM FOREST BASELINE MODEL (3,000-RECORD DATASET)")
    print("=" * 80)

    data_dir, models_dir, eval_dir = resolve_paths()
    print(f"Loading data from: {data_dir}")

    # Load Data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)
    print(f"Training Features Shape: {X_train.shape} (Expected: 2400, 55)")
    print(f"Testing Features Shape : {X_test.shape} (Expected: 600, 55)")
    print(f"Training Target Class Distribution:\n{y_train.value_counts().to_string()}")
    print(f"Testing Target Class Distribution:\n{y_test.value_counts().to_string()}")

    if X_train.shape != (2400, 55) or X_test.shape != (600, 55):
        raise ValueError(f"Unexpected matrix shapes: Train={X_train.shape}, Test={X_test.shape}")

    # Train Model
    print(f"\nTraining Random Forest baseline classifier (n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}, class_weight='{CLASS_WEIGHT}')...")
    rf_model = train_random_forest(X_train, y_train)
    print("[OK] Training complete.")

    # Evaluate Model
    print("\nEvaluating model on test dataset (X_test, y_test)...")
    metrics, y_pred, y_proba, cm = evaluate_model(rf_model, X_test, y_test, train_size=len(X_train))

    # Display Metrics
    print("\n" + "-" * 40)
    print("RANDOM FOREST BASELINE EVALUATION RESULTS")
    print("-" * 40)
    print(f"Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall    : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score  : {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Positives  (TP) : {metrics['true_positives']}")
    print(f"  True Negatives  (TN) : {metrics['true_negatives']}")
    print(f"  False Positives (FP) : {metrics['false_positives']}")
    print(f"  False Negatives (FN) : {metrics['false_negatives']}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

    # Save All Artifacts
    print("\nPersisting artifacts...")
    save_artifacts(rf_model, metrics, y_test, y_pred, y_proba, cm, models_dir, eval_dir)

    print("\n" + "=" * 80)
    print("[SUCCESS] REVISED STEP 4 RANDOM FOREST BASELINE EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_random_forest_pipeline()
