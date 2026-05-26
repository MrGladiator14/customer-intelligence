"""Meridian Customer Intelligence Platform — FastAPI Serving Layer."""

from contextlib import asynccontextmanager
import logging
import io
import time
from pathlib import Path
from typing import List

import joblib
import mlflow
import pandas as pd
from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config import PROJECT_ROOT, MODEL_DIR, DATA_DIR, MLFLOW_TRACKING_URI, MLFLOW_INFERENCE_EXPERIMENT_NAME
from src.data_pipeline.validate import validate_dataframe
from src.rag.langgraph_agent import run_rag_agent
from src.serving.schemas import (
    CustomerFeatures,
    PredictionResponse,
    AskRequest,
    AskResponse,
    CustomerIntelRequest,
    CustomerIntelResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Lifespan (replaces deprecated @app.on_event) ────────────────────────────
@asynccontextmanager
async def lifespan(app_instance):
    global INFERENCE_RUN
    # Startup
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_INFERENCE_EXPERIMENT_NAME)
        INFERENCE_RUN = mlflow.start_run(run_name="Serving")
        logger.info("MLflow inference tracking initialized (experiment=%s, run=%s).",
                     MLFLOW_INFERENCE_EXPERIMENT_NAME, INFERENCE_RUN.info.run_id)
    except Exception as e:
        logger.warning("Failed to initialize MLflow inference tracking: %s", e)
    yield
    # Shutdown
    if INFERENCE_RUN is not None:
        try:
            mlflow.end_run()
        except Exception:
            pass
    logger.info("MLflow inference tracking shut down.")

app = FastAPI(
    title="Meridian Customer Intelligence Platform API",
    description="Unified API combining structured predictive ML modeling with generative RAG complaint analysis.",
    version="2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI index.html at the root
@app.get("/", include_in_schema=False)
async def serve_ui():
    ui_path = PROJECT_ROOT / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)

# ── MLflow Inference Tracking ────────────────────────────────────────────────
INFERENCE_RUN = None
_inference_step = 0


def _log_inference(model_name: str, batch_size: int, n_class_1: int, avg_prob: float, latency_ms: float):
    global _inference_step
    if INFERENCE_RUN is None:
        return
    _inference_step += 1
    try:
        mlflow.log_metrics({
            "batch_size": batch_size,
            "n_class_1": n_class_1,
            "n_class_0": batch_size - n_class_1,
            "avg_probability": round(avg_prob, 4),
            "latency_ms": round(latency_ms, 2),
        }, step=_inference_step)
    except Exception as e:
        logger.debug(f"Failed to log inference to MLflow: {e}")


# ── Safe Model Loader with Mock Fallback ────────────────────────────────────
class SafeMockModel:
    """Fallback model if no serialized LightGBM or Baseline is available."""
    def predict(self, X):
        # Predict 1 if duration is long (> 300) else 0
        return (X["duration"] > 300).astype(int).values
        
    def predict_proba(self, X):
        # Returns [1-p, p] probabilities
        p = (X["duration"] / 1000.0).clip(0.0, 0.99).values
        return pd.DataFrame({0: 1.0 - p, 1: p}).values

def load_active_champion():
    promoted_path = MODEL_DIR / "active_champion_model.pkl"
    champion_path = MODEL_DIR / "champion_model.pkl"
    baseline_path = MODEL_DIR / "baseline_model.pkl"
    
    for path in [promoted_path, champion_path, baseline_path]:
        if path.exists():
            try:
                model = joblib.load(path)
                logger.info(f"Successfully loaded model '{path.name}'.")
                return model, path.name
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
                
    logger.warning("No serialized ML models found. Falling back to SafeMockModel.")
    return SafeMockModel(), "SafeMockModel"

def get_probability_band(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    else:
        return "Low"

@mlflow.trace
def perform_inference(model, df: pd.DataFrame) -> List[PredictionResponse]:
    """Preprocesses and runs predictions on the input DataFrame."""
    from src.training.train import preprocess_dataframe
    # Ensure inputs conform to our features
    X, _ = preprocess_dataframe(df)
    
    # Run predictions
    import numpy as np
    preds = np.asarray(model.predict(X))
    probs = np.asarray(model.predict_proba(X))[:, 1]
    
    responses = []
    for i, (idx, row) in enumerate(df.iterrows()):
        prob = float(probs[i])
        responses.append(PredictionResponse(
            customer_id=str(row["customer_id"]),
            conversion_prediction=int(preds[i]),
            conversion_probability=round(prob, 4),
            probability_band=get_probability_band(prob)
        ))
    return responses

# API router with /api prefix (matching the UI JS)
api_router = APIRouter()

# ── Endpoints ───────────────────────────────────────────────────────────────

@api_router.get("/health", tags=["Operational"])
def get_health():
    """Returns system status, active model version, and vector store integrity."""
    model, model_name = load_active_champion()
    
    # Check vector index status
    chroma_dir = DATA_DIR / "chroma_db"
    index_exists = chroma_dir.exists() and any(chroma_dir.iterdir())
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_version": model_name,
        "vector_index": {
            "status": "active" if index_exists else "not_found",
            "path": str(chroma_dir)
        },
        "system_configuration": {
            "environment": "production"
        }
    }

@api_router.post("/predict", response_model=PredictionResponse, tags=["ML Modeling"])
def predict_single(features: CustomerFeatures):
    """Computes real-time structured predictive conversion probability for a customer."""
    start = time.time()
    model, model_name = load_active_champion()

    # 2. Build DataFrame
    record_dict = features.model_dump()
    # Add dummy converted column for validation (required by schema but not used for prediction)
    record_dict["converted"] = 0
    df = pd.DataFrame([record_dict])

    # 3. Validate
    try:
        validated_df = validate_dataframe(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data validation failed: {str(e)}"
        )

    # 4. Infer
    results = perform_inference(model, validated_df)
    latency = (time.time() - start) * 1000
    _log_inference(model_name, 1, int(results[0].conversion_prediction), results[0].conversion_probability, latency)
    return results[0]

@api_router.post("/batch-score", response_model=List[PredictionResponse], tags=["ML Modeling"])
async def batch_score(file: UploadFile = File(...)):
    """Ingests a CSV file containing multiple customer records, validates them,

    and outputs conversion predictions with mapped probability bands.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV file."
        )
        
    # 1. Read file bytes
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV file: {str(e)}"
        )
        
    # 2. Pandera Schema Validation
    try:
        validated_df = validate_dataframe(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch Pandera data validation failed: {str(e)}"
        )
        
    # 3. Load Model & Score
    start = time.time()
    model, model_name = load_active_champion()
    results = perform_inference(model, validated_df)
    latency = (time.time() - start) * 1000
    n_class_1 = sum(r.conversion_prediction for r in results)
    avg_prob = sum(r.conversion_probability for r in results) / len(results) if results else 0.0
    _log_inference(model_name, len(results), n_class_1, avg_prob, latency)
    return results

@api_router.post("/ask-complaints", response_model=AskResponse, tags=["LLM/RAG Lane"])
def ask_complaints(request: AskRequest):
    """Executes the stateful LangGraph agent over indexed complaints to retrieve answers,

    verifying relevance and generating response citations.
    """
    try:
        agent_out = run_rag_agent(request.question)
        return AskResponse(
            question=request.question,
            response=agent_out["response"],
            citations=agent_out["citations"],
            latency_ms=agent_out["latency_ms"],
            relevance_score=agent_out["relevance_score"]
        )
    except Exception as e:
        logger.error(f"Error running LangGraph RAG Agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Agent failure: {str(e)}"
        )

@api_router.post("/customer-intel", response_model=CustomerIntelResponse, tags=["Unified Plane"])
def get_customer_intel(request: CustomerIntelRequest):
    """Combines structured predictive ML models with LangGraph aggregate complaint insights."""
    # 1. ML Lane
    start = time.time()
    model, model_name = load_active_champion()
    customer_dict = request.customer.model_dump()
    # Add dummy converted column for validation (required by schema but not used for prediction)
    customer_dict["converted"] = 0
    record_df = pd.DataFrame([customer_dict])

    try:
        validated_df = validate_dataframe(record_df)
        ml_results = perform_inference(model, validated_df)
        prediction_info = ml_results[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ML lane validation or execution failed: {str(e)}"
        )
    latency = (time.time() - start) * 1000
    _log_inference(model_name, 1, int(prediction_info.conversion_prediction), prediction_info.conversion_probability, latency)
        
    # 2. RAG Lane
    try:
        agent_out = run_rag_agent(request.question)
        complaint_info = AskResponse(
            question=request.question,
            response=agent_out["response"],
            citations=agent_out["citations"],
            latency_ms=agent_out["latency_ms"],
            relevance_score=agent_out["relevance_score"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG lane execution failed: {str(e)}"
        )
        
    return CustomerIntelResponse(
        customer_id=request.customer.customer_id,
        conversion_info=prediction_info,
        complaint_insights=complaint_info
    )

app.include_router(api_router)
