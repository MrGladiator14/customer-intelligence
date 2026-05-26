"""Meridian Customer Intelligence Platform - Model Evaluation & Promotion Gate."""

import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, auc

from src.config import (
    MODEL_DIR,
    PROJECT_ROOT,
    PROMOTION_PR_AUC_MIN_IMPROVEMENT,
    PROMOTION_F1_MAX_DROP
)
from src.data_pipeline.ingest import ingest_csv
from src.training.train import preprocess_dataframe, compute_pr_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_and_promote():
    logger.info("Starting Relative Promotion Gate evaluation...")
    
    # 1. Load baseline and champion models
    baseline_path = MODEL_DIR / "baseline_model.pkl"
    champion_path = MODEL_DIR / "champion_model.pkl"
    
    if not baseline_path.exists() or not champion_path.exists():
        logger.error("Required model files (baseline or champion) do not exist. Please run training first.")
        sys.exit(1)
        
    baseline_model = joblib.load(baseline_path)
    champion_model = joblib.load(champion_path)
    
    # 2. Ingest evaluation data
    eval_df = ingest_csv(PROJECT_ROOT / "data" / "test.csv")
    X_eval, y_eval = preprocess_dataframe(eval_df)
    y_eval = y_eval.astype(int)
    
    # 3. Compute Baseline Metrics
    y_pred_b = baseline_model.predict(X_eval)
    y_prob_b = baseline_model.predict_proba(X_eval)[:, 1]
    
    baseline_f1 = f1_score(y_eval, y_pred_b)
    baseline_pr_auc = compute_pr_auc(y_eval, y_prob_b)
    
    # 4. Compute Champion Metrics
    y_pred_c = champion_model.predict(X_eval)
    y_prob_c = champion_model.predict_proba(X_eval)[:, 1]
    
    champion_f1 = f1_score(y_eval, y_pred_c)
    champion_pr_auc = compute_pr_auc(y_eval, y_prob_c)
    
    logger.info("=== Evaluation Metrics Summary ===")
    logger.info(f"Baseline: PR-AUC = {baseline_pr_auc:.4f}, F1-Score = {baseline_f1:.4f}")
    logger.info(f"Champion: PR-AUC = {champion_pr_auc:.4f}, F1-Score = {champion_f1:.4f}")
    
    # 5. Evaluate relative promotion rules
    pr_auc_improvement = champion_pr_auc - baseline_pr_auc
    f1_change = champion_f1 - baseline_f1
    
    logger.info(f"PR-AUC Improvement: {pr_auc_improvement:.4f} (Required >= {PROMOTION_PR_AUC_MIN_IMPROVEMENT:.4f})")
    logger.info(f"F1-Score Change: {f1_change:.4f} (Allowed drop <= {PROMOTION_F1_MAX_DROP:.4f}, i.e., Change >= {-PROMOTION_F1_MAX_DROP:.4f})")
    
    is_pr_auc_passed = pr_auc_improvement >= PROMOTION_PR_AUC_MIN_IMPROVEMENT
    is_f1_passed = f1_change >= -PROMOTION_F1_MAX_DROP
    
    if is_pr_auc_passed and is_f1_passed:
        logger.info(">>> Promotion Gate PASSED! Promoting LightGBM Champion model.")
        # Copy the champion model to the promoted endpoint model path
        promoted_path = MODEL_DIR / "active_champion_model.pkl"
        joblib.dump(champion_model, promoted_path)
        logger.info(f"Active Champion model successfully registered/promoted to {promoted_path}")
        
        # In a real environment, we would also register to Azure ML Model Registry:
        # e.g., az ml model create --name customer-intel-champion --path MODEL_DIR/active_champion_model.pkl
        return True
    else:
        logger.error(">>> Promotion Gate FAILED!")
        if not is_pr_auc_passed:
            logger.error(f"Reason: Candidate did not beat active baseline by >= {PROMOTION_PR_AUC_MIN_IMPROVEMENT * 100}% on PR-AUC.")
        if not is_f1_passed:
            logger.error(f"Reason: Candidate dropped F1-Score by more than {PROMOTION_F1_MAX_DROP * 100}%.")
        
        # Hard exit 1 to crash the DevOps / CD pipeline as requested
        sys.exit(1)

if __name__ == "__main__":
    evaluate_and_promote()
