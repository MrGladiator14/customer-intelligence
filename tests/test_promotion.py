"""Unit tests for Relative Model Promotion Gate."""

import sys
from unittest.mock import patch, MagicMock
import pytest
from src.training.evaluate import evaluate_and_promote

@patch("src.training.evaluate.joblib.load")
@patch("src.training.evaluate.ingest_csv")
@patch("src.training.evaluate.preprocess_dataframe")
@patch("src.training.evaluate.f1_score")
@patch("src.training.evaluate.compute_pr_auc")
def test_promotion_gate_success(
    mock_compute_pr_auc,
    mock_f1_score,
    mock_preprocess,
    mock_ingest,
    mock_joblib_load
):
    # Mocking standard inputs
    mock_joblib_load.return_value = MagicMock()
    mock_ingest.return_value = MagicMock()
    mock_preprocess.return_value = (MagicMock(), MagicMock())
    
    # Mocking metrics such that Champion passes the gate
    # Baseline: PR-AUC = 0.80, F1 = 0.82
    # Champion: PR-AUC = 0.85 (>= 0.83), F1 = 0.81 (>= 0.80)
    mock_compute_pr_auc.side_effect = [0.80, 0.85] # first baseline, second champion
    mock_f1_score.side_effect = [0.82, 0.81]
    
    with patch("src.training.evaluate.joblib.dump") as mock_dump:
        result = evaluate_and_promote()
        assert result is True
        mock_dump.assert_called_once()

@patch("src.training.evaluate.joblib.load")
@patch("src.training.evaluate.ingest_csv")
@patch("src.training.evaluate.preprocess_dataframe")
@patch("src.training.evaluate.f1_score")
@patch("src.training.evaluate.compute_pr_auc")
def test_promotion_gate_failed_prauc(
    mock_compute_pr_auc,
    mock_f1_score,
    mock_preprocess,
    mock_ingest,
    mock_joblib_load
):
    # Mocking standard inputs
    mock_joblib_load.return_value = MagicMock()
    mock_ingest.return_value = MagicMock()
    mock_preprocess.return_value = (MagicMock(), MagicMock())
    
    # Mocking metrics such that Champion PR-AUC fails (improvement < 3%)
    # Baseline: PR-AUC = 0.80, F1 = 0.82
    # Champion: PR-AUC = 0.81 (< 0.83), F1 = 0.81 (passes F1)
    mock_compute_pr_auc.side_effect = [0.80, 0.81]
    mock_f1_score.side_effect = [0.82, 0.81]
    
    with pytest.raises(SystemExit) as excinfo:
        evaluate_and_promote()
    assert excinfo.value.code == 1

@patch("src.training.evaluate.joblib.load")
@patch("src.training.evaluate.ingest_csv")
@patch("src.training.evaluate.preprocess_dataframe")
@patch("src.training.evaluate.f1_score")
@patch("src.training.evaluate.compute_pr_auc")
def test_promotion_gate_failed_f1(
    mock_compute_pr_auc,
    mock_f1_score,
    mock_preprocess,
    mock_ingest,
    mock_joblib_load
):
    # Mocking standard inputs
    mock_joblib_load.return_value = MagicMock()
    mock_ingest.return_value = MagicMock()
    mock_preprocess.return_value = (MagicMock(), MagicMock())
    
    # Mocking metrics such that Champion F1-Score fails (drop > 2%)
    # Baseline: PR-AUC = 0.80, F1 = 0.82
    # Champion: PR-AUC = 0.86 (passes PR-AUC), F1 = 0.79 (drop is 3% which is > 2%)
    mock_compute_pr_auc.side_effect = [0.80, 0.86]
    mock_f1_score.side_effect = [0.82, 0.79]
    
    with pytest.raises(SystemExit) as excinfo:
        evaluate_and_promote()
    assert excinfo.value.code == 1
