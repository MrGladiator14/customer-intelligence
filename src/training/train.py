"""Meridian Customer Intelligence Platform - Model Training."""

import logging
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import joblib

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    PROJECT_ROOT,
    MODEL_DIR
)
from src.data_pipeline.ingest import ingest_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Feature Engineering Mapping ─────────────────────────────────────────────
JOB_MAPPING = {
    "management": 0, "technician": 1, "self-employed": 2, "blue-collar": 3,
    "services": 4, "retired": 5, "admin.": 6, "student": 7, "entrepreneur": 8
}
EDUCATION_MAPPING = {
    "primary": 0, "secondary": 1, "tertiary": 2, "unknown": -1
}

def preprocess_dataframe(df: pd.DataFrame):
    """Encodes categorical columns and returns feature matrix X and target y."""
    processed = df.copy()
    processed["job_code"] = processed["job"].map(JOB_MAPPING).fillna(-1).astype(int)
    processed["edu_code"] = processed["education"].map(EDUCATION_MAPPING).fillna(-1).astype(int)
    
    feature_cols = ["age", "balance", "duration", "job_code", "edu_code"]
    X = processed[feature_cols]
    y = processed["converted"] if "converted" in processed.columns else None
    return X, y

def compute_pr_auc(y_true, y_prob):
    """Computes the Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def plot_and_save_calibration(y_true, y_prob, model_name: str, filepath: str):
    """Generates and saves a calibration curve plot."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"{model_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved calibration curve for {model_name} to {filepath}")

def setup_mlflow():
    local_db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{local_db_path}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def main():
    # Setup directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    setup_mlflow()
    
    # 1. Ingest & Validate
    train_df = ingest_csv(PROJECT_ROOT / "data" / "train.csv")
    test_df = ingest_csv(PROJECT_ROOT / "data" / "test.csv")
    
    # 2. Preprocess
    X_train, y_train = preprocess_dataframe(train_df)
    X_test, y_test = preprocess_dataframe(test_df)
    
    # Ensure targets are integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    
    # ── Train Baseline Model (Logistic Regression) ──────────────────────────
    with mlflow.start_run(run_name="Baseline_Model") as baseline_run:
        logger.info("Training Baseline Model (Logistic Regression)...")
        baseline_model = LogisticRegression(random_state=42, max_iter=1000)
        baseline_model.fit(X_train, y_train)
        
        # Predict
        y_pred = baseline_model.predict(X_test)
        y_prob = baseline_model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = compute_pr_auc(y_test, y_prob)
        
        logger.info(f"Baseline: Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        
        # Log to MLflow
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })
        
        # Plot and log calibration curve
        cal_path = MODEL_DIR / "calibration_baseline.png"
        plot_and_save_calibration(y_test, y_prob, "Baseline (Logistic Regression)", str(cal_path))
        mlflow.log_artifact(str(cal_path))
        
        # Save model
        mlflow.sklearn.log_model(baseline_model, "baseline_model")
        joblib.dump(baseline_model, MODEL_DIR / "baseline_model.pkl")
        logger.info("Saved baseline model locally.")
        
    # ── Train Champion Model (LightGBM) ─────────────────────────────────────
    with mlflow.start_run(run_name="LightGBM_Champion") as champion_run:
        logger.info("Training LightGBM Champion Model...")
        # Small dataset, so we use very simple hyper-parameters to avoid issues
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=2,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # Predict
        y_pred_c = lgb_model.predict(X_test)
        y_prob_c = lgb_model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        acc_c = accuracy_score(y_test, y_pred_c)
        f1_c = f1_score(y_test, y_pred_c)
        roc_auc_c = roc_auc_score(y_test, y_prob_c)
        pr_auc_c = compute_pr_auc(y_test, y_prob_c)
        
        logger.info(f"Champion (LightGBM): Accuracy={acc_c:.4f}, F1={f1_c:.4f}, ROC-AUC={roc_auc_c:.4f}, PR-AUC={pr_auc_c:.4f}")
        
        # Log to MLflow
        mlflow.log_params({
            "model_type": "LightGBM",
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 42
        })
        mlflow.log_metrics({
            "accuracy": acc_c,
            "f1_score": f1_c,
            "roc_auc": roc_auc_c,
            "pr_auc": pr_auc_c
        })
        
        # Plot and log calibration curve
        cal_path_c = MODEL_DIR / "calibration_champion.png"
        plot_and_save_calibration(y_test, y_prob_c, "Champion (LightGBM)", str(cal_path_c))
        mlflow.log_artifact(str(cal_path_c))
        
        # Save model
        mlflow.lightgbm.log_model(lgb_model, "champion_model")
        joblib.dump(lgb_model, MODEL_DIR / "champion_model.pkl")
        logger.info("Saved champion model locally.")

if __name__ == "__main__":
    main()
