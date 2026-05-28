"""API Contract & Integration tests for FastAPI Serving Layer."""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from src.serving.serve import app

client = TestClient(app)

@patch("src.serving.serve.load_active_champion")
def test_endpoint_health(mock_load):
    mock_load.return_value = (MagicMock(), "MockModel.pkl")
    
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["model_version"] == "MockModel.pkl"
    assert "vector_index" in json_data

@patch("src.serving.serve.load_active_champion")
def test_endpoint_predict_success(mock_load):
    # Setup mock model predictions
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.1, 0.9]]
    mock_load.return_value = (mock_model, "MockModel.pkl")
    
    # Valid request body
    payload = {
        "customer_id": "CUST201",
        "age": 42,
        "education": "secondary",
        "job": "technician",
        "balance": 1500.0,
        "duration": 340,
        "complaint": "Dissatisfied with response delay. [Doc-555]"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["customer_id"] == "CUST201"
    assert json_data["conversion_prediction"] == 1
    assert json_data["conversion_probability"] == 0.9
    assert json_data["probability_band"] == "High"

def test_endpoint_predict_validation_error():
    # Invalid request body: missing age, negative duration
    payload = {
        "customer_id": "CUST201",
        "education": "secondary",
        "job": "technician",
        "balance": 1500.0,
        "duration": -10,  # Invalid
        "complaint": "Short"
    }
    
    response = client.post("/predict", json=payload)
    # FastAPI automatically catches Pydantic validation errors and returns 422
    assert response.status_code == 422

@patch("src.serving.serve.load_active_champion")
def test_endpoint_batch_score_success(mock_load):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0]
    mock_model.predict_proba.return_value = [[0.2, 0.8], [0.8, 0.2]]
    mock_load.return_value = (mock_model, "MockModel.pkl")
    
    csv_content = """customer_id,age,education,job,balance,duration,complaint,converted
CUST101,35,tertiary,management,2000.0,220,Home loan rate hike.,1
CUST102,41,secondary,technician,600.0,110,App keeps crashing.,0
"""
    
    # Send CSV as file upload
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/batch-score", files=files)
    
    assert response.status_code == 200
    json_data = response.json()
    assert len(json_data) == 2
    assert json_data[0]["customer_id"] == "CUST101"
    assert json_data[0]["probability_band"] == "High"
    assert json_data[1]["customer_id"] == "CUST102"
    assert json_data[1]["probability_band"] == "Low"

@patch("src.serving.serve.run_rag_agent")
def test_endpoint_ask_complaints(mock_run):
    mock_run.return_value = {
        "question": "Is there a card issue?",
        "response": "Card was blocked at grocery store [Doc-104].",
        "citations": ["Doc-104"],
        "latency_ms": 12.5,
        "relevance_score": 0.85
    }
    
    payload = {"question": "Is there a card issue?"}
    response = client.post("/ask-complaints", json=payload)
    
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["response"] == "Card was blocked at grocery store [Doc-104]."
    assert json_data["citations"] == ["Doc-104"]
    assert json_data["latency_ms"] == 12.5

@patch("src.serving.serve.load_active_champion")
@patch("src.rag.retrieve.retrieve_complaints")
@patch("src.rag.answer.call_nvidia_llama")
def test_endpoint_customer_intel(mock_llm, mock_retrieve, mock_load):
    # Setup mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.55, 0.45]]
    mock_load.return_value = (mock_model, "MockModel.pkl")
    
    # Setup mock RAG
    mock_retrieve.return_value = [
        {"source_id": "Doc-102", "text": "App is slow.", "similarity": 0.65}
    ]
    mock_llm.return_value = "Mocked LLM Response"
    
    payload = {
        "customer": {
            "customer_id": "CUST505",
            "age": 28,
            "education": "tertiary",
            "job": "self-employed",
            "balance": 12000.0,
            "duration": 500,
            "complaint": "App is slow."
        },
        "question": "Why is the app slow?"
    }
    
    response = client.post("/customer-intel", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["customer_id"] == "CUST505"
    assert json_data["conversion_band"] == "Medium"
    assert json_data["cited_ids"] == ["Doc-102"]
