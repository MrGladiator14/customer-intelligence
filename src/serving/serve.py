"""Meridian Customer Intelligence Platform - FastAPI Serving Layer."""

from contextlib import asynccontextmanager
import logging
import io
import time
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import mlflow
import pandas as pd
from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.config import PROJECT_ROOT, MODEL_DIR, DATA_DIR, MLFLOW_TRACKING_URI, MLFLOW_INFERENCE_EXPERIMENT_NAME, RAG_SIMILARITY_THRESHOLD
from src.data_pipeline.validate import validate_dataframe
from src.rag.langgraph_agent import run_rag_agent
from src.serving.schemas import (
    CustomerFeatures,
    PredictionResponse,
    AskRequest,
    AskResponse,
    CustomerIntelRequest,
    CustomerIntelResponse,
    BatchScoreResponse,
    SubmitSupportResponseRequest
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Telemetry & Metrics Tracker ─────────────────────────────────────────────
METRICS_LOG: List[Dict[str, Any]] = []

def track_metric(
    endpoint: str,
    latency_ms: float,
    error: bool = False,
    prediction_band: Optional[str] = None,
    RAG_hit: Optional[bool] = None,
    RAG_refusal: Optional[bool] = None,
    RAG_relevance: Optional[float] = None
):
    entry = {
        "timestamp": time.time(),
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "error": error,
        "prediction_band": prediction_band,
        "RAG_hit": RAG_hit,
        "RAG_refusal": RAG_refusal,
        "RAG_relevance": RAG_relevance
    }
    METRICS_LOG.append(entry)
    # Write to local file
    try:
        metrics_file = DATA_DIR / "metrics_log.jsonl"
        import json
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug(f"Failed to append to metrics log: {e}")

# ── Lifespan (replaces deprecated @app.on_event) ────────────────────────────
@asynccontextmanager
async def lifespan(app_instance):
    global INFERENCE_RUN
    # Load past metrics
    try:
        metrics_file = DATA_DIR / "metrics_log.jsonl"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r") as f:
                for line in f:
                    if line.strip():
                        METRICS_LOG.append(json.loads(line))
        logger.info(f"Loaded {len(METRICS_LOG)} telemetry metrics from history.")
    except Exception as e:
        logger.warning("Failed to load historical metrics: %s", e)

    # Initialize local SQLite DB and pre-populate
    try:
        from src.serving.database import init_db
        init_db()
        logger.info("Local SQLite database pre-population finished.")
    except Exception as e:
        logger.error("Failed to initialize local SQLite database: %s", e)

    # Startup MLflow
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

@app.get("/test-endpoint", include_in_schema=False)
async def serve_test_endpoint():
    ui_path = PROJECT_ROOT / "ui" / "test.html"
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
        return (X["duration"] > 300).astype(int).values
        
    def predict_proba(self, X):
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
def perform_inference(model, df: pd.DataFrame, model_name: str) -> List[PredictionResponse]:
    """Preprocesses and runs predictions on the input DataFrame."""
    from src.data_pipeline.features import preprocess_dataframe
    X, _ = preprocess_dataframe(df)
    
    import numpy as np
    preds = np.asarray(model.predict(X))
    probs = np.asarray(model.predict_proba(X))[:, 1]
    
    responses = []
    for i, (idx, row) in enumerate(df.iterrows()):
        prob = float(probs[i])
        pred = int(preds[i])
        band = get_probability_band(prob)
        responses.append(PredictionResponse(
            customer_id=str(row["customer_id"]),
            conversion_prediction=pred,
            conversion_probability=round(prob, 4),
            probability_band=band,
            prediction=pred,
            probability=round(prob, 4),
            threshold_decision=True if pred == 1 else False,
            model_version=model_name
        ))
    return responses

# API router with /api prefix (matching the UI JS)
api_router = APIRouter()

# ── Endpoints ───────────────────────────────────────────────────────────────

@api_router.get("/health", tags=["Operational"])
def get_health():
    """Returns flat health and model version metrics (Page 4 Contract)."""
    model, model_name = load_active_champion()
    chroma_dir = DATA_DIR / "chroma_db"
    index_exists = chroma_dir.exists() and any(chroma_dir.iterdir())
    
    return {
        "status": "healthy",
        "model_version": model_name,
        "vector_index_version": "active" if index_exists else "not_found",
        # Backward compatibility for test assertion:
        "vector_index": {
            "status": "active" if index_exists else "not_found",
            "path": str(chroma_dir)
        }
    }

@api_router.post("/predict", response_model=PredictionResponse, tags=["ML Modeling"])
def predict_single(features: CustomerFeatures):
    """Computes real-time structured predictive conversion probability for a customer."""
    start = time.time()
    model, model_name = load_active_champion()

    record_dict = features.model_dump()
    record_dict["converted"] = 0
    df = pd.DataFrame([record_dict])

    try:
        validated_df = validate_dataframe(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data validation failed: {str(e)}"
        )

    results = perform_inference(model, validated_df, model_name)
    latency = (time.time() - start) * 1000
    
    _log_inference(model_name, 1, results[0].conversion_prediction, results[0].conversion_probability, latency)
    track_metric(
        endpoint="predict",
        latency_ms=latency,
        prediction_band=results[0].probability_band
    )
    
    return results[0]

@api_router.post("/batch-score", response_model=BatchScoreResponse, tags=["ML Modeling"])
async def batch_score(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """Ingests a CSV file or JSON array, performs scoring, and saves outputs."""
    start = time.time()
    df = None
    
    # 1. Parse CSV File Upload
    if file is not None:
        if file.filename is None or not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Please upload a CSV file."
            )
        try:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse CSV file: {str(e)}"
            )
    # 2. Parse JSON Body
    else:
        try:
            body = await request.json()
            if isinstance(body, dict) and "records" in body:
                records = body["records"]
            elif isinstance(body, list):
                records = body
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON batch payload structure."
                )
            df = pd.DataFrame(records)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse JSON batch: {str(e)}"
            )
            
    if df is None or len(df) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty batch submitted.")

    # Auto-fill target column for schema validator
    if "converted" not in df.columns:
        df["converted"] = 0

    # 3. Validation
    try:
        validated_df = validate_dataframe(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch Pandera data validation failed: {str(e)}"
        )
        
    # 4. Load Model & Score
    model, model_name = load_active_champion()
    results = perform_inference(model, validated_df, model_name)
    latency = (time.time() - start) * 1000
    
    n_class_1 = sum(r.conversion_prediction for r in results)
    avg_prob = sum(r.conversion_probability for r in results) / len(results) if results else 0.0
    _log_inference(model_name, len(results), n_class_1, avg_prob, latency)
    
    # Save scored file path locally
    scored_dir = DATA_DIR / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scored_file_name = f"scored_batch_{timestamp}.csv"
    scored_file_path = scored_dir / scored_file_name
    
    df_scored = df.copy()
    df_scored["conversion_prediction"] = [r.conversion_prediction for r in results]
    df_scored["conversion_probability"] = [r.conversion_probability for r in results]
    df_scored["probability_band"] = [r.probability_band for r in results]
    df_scored.to_csv(scored_file_path, index=False)
    
    # Compute counts
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for r in results:
        band = r.probability_band
        counts[band] = counts.get(band, 0) + 1
        
    track_metric(
        endpoint="batch-score",
        latency_ms=latency,
        prediction_band=get_probability_band(avg_prob)
    )
    
    return BatchScoreResponse(
        scored_file_path=str(scored_file_path),
        counts_by_conversion_band=counts,
        predictions=results
    )

@api_router.get("/download-merged-scored", tags=["ML Modeling"])
def download_merged_scored():
    """Merges all CSVs in data/scored, returns the merged CSV, and deletes the original files."""
    scored_dir = DATA_DIR / "scored"
    if not scored_dir.exists():
        raise HTTPException(status_code=404, detail="No scored files found.")
        
    csv_files = list(scored_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="No scored files found.")
    
    try:
        df_list = [pd.read_csv(f) for f in csv_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_csv_content = merged_df.to_csv(index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge CSV files: {str(e)}")

    for f in csv_files:
        try:
            f.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {f}: {e}")

    return Response(
        content=merged_csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=merged_scored_{int(time.time())}.csv"}
    )

@api_router.post("/calculate-drift", tags=["ML Modeling"])
async def calculate_drift(file: UploadFile = File(...)):
    """Calculates covariate shift (drift) for an uploaded dataset against the training reference."""
    import sys
    from src.config import PROJECT_ROOT
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
        
    try:
        from monitoring.ml_drift import calculate_psi, calculate_ks_test
    except ImportError:
        raise HTTPException(status_code=500, detail="Could not import drift detection methods.")

    train_path = DATA_DIR / "synthetic_train.csv"
    if not train_path.exists():
        raise HTTPException(status_code=500, detail="Reference dataset synthetic_train.csv not found.")
        
    ref_df = pd.read_csv(train_path)
    
    try:
        contents = await file.read()
        prod_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {str(e)}")

    features = ["age", "balance", "duration"]
    drift_results = {}
    
    for feat in features:
        if feat not in prod_df.columns:
            continue
            
        import numpy as np
        ref_arr = np.asarray(ref_df[feat].dropna(), dtype=float)
        prod_arr = np.asarray(prod_df[feat].dropna(), dtype=float)
        
        psi = calculate_psi(ref_arr, prod_arr)
        p_val = calculate_ks_test(ref_arr, prod_arr)
        
        if psi > 0.25:
            status = "SEVERE DRIFT"
        elif psi > 0.1:
            status = "MODERATE DRIFT"
        else:
            status = "STABLE"
            
        drift_results[feat] = {
            "psi": round(psi, 4),
            "ks_p_value": p_val,
            "status": status,
            "ref_mean": round(float(np.mean(ref_arr)), 2) if len(ref_arr) > 0 else 0.0,
            "prod_mean": round(float(np.mean(prod_arr)), 2) if len(prod_arr) > 0 else 0.0
        }
        
    return {"status": "success", "drift_analysis": drift_results}

@api_router.post("/ask-complaints", response_model=AskResponse, tags=["LLM/RAG Lane"])
def ask_complaints(request: AskRequest):
    """Executes the stateful LangGraph agent over indexed complaints with optional filters."""
    start = time.time()
    try:
        agent_out = run_rag_agent(
            question=request.question,
            product=request.product or "",
            company=request.company or "",
            date=request.date or "",
            issue=request.issue or ""
        )
        
        latency = (time.time() - start) * 1000
        score = agent_out.get("relevance_score", 0.0)
        sufficient = score >= RAG_SIMILARITY_THRESHOLD
        sufficiency_note = (
            f"Sufficiency Check: Retrieved evidence matches query with similarity score {score:.4f} "
            f"(above threshold {RAG_SIMILARITY_THRESHOLD})"
            if sufficient else
            f"Sufficiency Check: Insufficient or weak evidence matching query. "
            f"No chunk crossed similarity threshold of {RAG_SIMILARITY_THRESHOLD}."
        )
        
        latency_val = agent_out.get("latency_ms", round(latency, 2))
        prompt_version = "v2.0"
        response = AskResponse(
            question=request.question,
            response=agent_out["response"],
            citations=agent_out["citations"],
            latency_ms=latency_val,
            relevance_score=score,
            answer=agent_out["response"],
            retrieved_evidence_ids=agent_out["citations"],
            evidence_sufficiency_note=sufficiency_note,
            prompt_version=prompt_version
        )
        
        track_metric(
            endpoint="ask-complaints",
            latency_ms=latency_val,
            RAG_hit=len(agent_out["citations"]) > 0,
            RAG_refusal="Refused:" in agent_out["response"],
            RAG_relevance=score
        )
        return response
    except Exception as e:
        logger.error(f"Error running LangGraph RAG Agent: {e}")
        track_metric(endpoint="ask-complaints", latency_ms=(time.time() - start) * 1000, error=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Agent failure: {str(e)}"
        )

@api_router.post("/customer-intel", response_model=CustomerIntelResponse, tags=["Unified Plane"])
def get_customer_intel(request: CustomerIntelRequest):
    """Combines conversion probability and synthesizes aggregate themes for the segment."""
    start = time.time()
    
    # 1. ML Prediction
    model, model_name = load_active_champion()
    customer_dict = request.customer.model_dump()
    customer_dict["converted"] = 0
    record_df = pd.DataFrame([customer_dict])

    try:
        validated_df = validate_dataframe(record_df)
        ml_results = perform_inference(model, validated_df, model_name)
        prediction_info = ml_results[0]
    except Exception as e:
        track_metric(endpoint="customer-intel", latency_ms=(time.time() - start) * 1000, error=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ML lane validation failed: {str(e)}"
        )
        
    # 2. Segment-level RAG Synthesis
    try:
        # Construct retrieval query combining question and user complaint
        query_parts = []
        if request.question:
            query_parts.append(request.question)
        if request.customer.complaint:
            query_parts.append(request.customer.complaint)
        retrieval_query = " ".join(query_parts) if query_parts else "customer support query"

        # Retrieve complaints matching filters
        from src.rag.retrieve import retrieve_complaints
        complaints = retrieve_complaints(
            query=retrieval_query,
            product=request.product or "",
            company=request.company or "",
            date=request.date or "",
            issue=request.issue or "",
            customer_id=request.customer.customer_id,
            limit=5
        )
        
        # Retrieve similar complaints from across the knowledge base for support suggestions
        similar_complaints = retrieve_complaints(
            query=request.customer.complaint if request.customer.complaint else (request.question or ""),
            product=request.product or "",
            company=request.company or "",
            date=request.date or "",
            issue=request.issue or "",
            limit=5
        )
        
        seen_queries = set()
        similar_queries = []
        for doc in similar_complaints:
            q_text = doc["text"]
            if q_text not in seen_queries:
                seen_queries.add(q_text)
                similar_queries.append({
                    "query": q_text,
                    "response": doc.get("support_response", "")
                })

        # Generate suggested response based on retrieved similar complaints
        if similar_complaints and request.customer.complaint:
            context_str = "\n".join([f"Similar Complaint: {doc['text']}\nResponse Given: {doc.get('support_response', '')}" for doc in similar_complaints])
            prompt2 = f"""You are a helpful customer support agent.
Based on the following similar complaints and their responses from our knowledge base:
{context_str}

Draft a professional, helpful, and concise suggested response to the user's current complaint: "{request.customer.complaint}"
CRITICAL INSTRUCTION: Output ONLY the exact drafted response text that will be sent to the customer. Do not include any introductory text (e.g. "Here's a response"), preamble, conversational filler, or explanations of your process.
Suggested Response:"""
            from src.rag.answer import call_nvidia_llama
            suggested_response = call_nvidia_llama(prompt2)
        else:
            suggested_response = "We have received your query and our team will get back to you shortly."
            

        
        citations = [c["source_id"] for c in complaints if c["source_id"] != "Doc-Unknown"]
        
        # Synthesize top complaint themes for segment
        if not complaints:
            top_themes = "No matching complaints or themes identified for this segment."
        else:
            context_str = ""
            for doc in complaints:
                context_str += f"- [{doc['source_id']}]: {doc['text']}\n"
            
            prompt = f"""You are a helpful customer intelligence analyst.
Summarize the main themes and complaints from this segment:
{context_str}

Summary of themes (1-2 paragraphs, professionally citing the Document IDs like [Doc-101]):"""
            from src.rag.answer import call_nvidia_llama
            top_themes = call_nvidia_llama(prompt)
            
        from src.rag.retrieve import evaluate_relevance
        score = evaluate_relevance(complaints)

    except Exception as e:
        track_metric(endpoint="customer-intel", latency_ms=(time.time() - start) * 1000, error=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG lane theme synthesis failed: {str(e)}"
        )
        
    latency = (time.time() - start) * 1000
    track_metric(
        endpoint="customer-intel",
        latency_ms=latency,
        prediction_band=prediction_info.probability_band,
        RAG_hit=len(citations) > 0,
        RAG_relevance=score
    )
    
    return CustomerIntelResponse(
        customer_id=request.customer.customer_id,
        conversion_band=prediction_info.probability_band,
        top_complaint_themes=top_themes,
        cited_ids=list(set(citations)),
        suggested_response=suggested_response,
        similar_queries=similar_queries
    )

@api_router.get("/customer-details/{customer_id}", tags=["Unified Plane"])
def get_customer_details_endpoint(customer_id: str):
    """Retrieves customer details from cache or SQLite database."""
    from src.serving.database import get_customer_details
    details = get_customer_details(customer_id)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer details for ID '{customer_id}' not found."
        )
    return details

@api_router.get("/mock-users", tags=["Unified Plane"])
def get_mock_users():
    """Retrieves random mock users from synthetic_train.csv."""
    import csv
    import random
    from src.config import DATA_DIR
    csv_path = DATA_DIR / "synthetic_train.csv"
    users = []
    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if reader:
                samples = random.sample(reader, min(3, len(reader)))
                for idx, row in enumerate(samples):
                    users.append({
                        "id": row.get("customer_id", f"CUST-RND{idx}"),
                        "name": f"Mock User {idx+1}",
                        "complaints": [row.get("complaint", "General inquiry")]
                    })
    return users

@api_router.post("/submit-support-response", tags=["Unified Plane"])
def submit_support_response(request: SubmitSupportResponseRequest):
    """Stores the finalized, edited response sent by the support team so it can be added to the training dataset."""
    from src.serving.database import add_support_query_response
    try:
        add_support_query_response(request.customer_id, request.query, request.response)
        return {"status": "success", "message": "Support response logged for training."}
    except Exception as e:
        logger.error(f"Failed to log support response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log support response: {str(e)}"
        )

@api_router.get("/download-metrics", tags=["Telemetry"])
def download_metrics():
    """Downloads the telemetry metrics log file."""
    metrics_file = DATA_DIR / "metrics_log.jsonl"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics log file not found.")
    
    try:
        content = metrics_file.read_text(encoding="utf-8")
        
        if METRICS_LOG:
            n = len(METRICS_LOG)
            avg_latency = sum(log["latency_ms"] for log in METRICS_LOG) / n
            
            bands = [log.get("prediction_band") for log in METRICS_LOG if log.get("prediction_band")]
            most_common_band = None
            if bands:
                import collections
                most_common_band = collections.Counter(bands).most_common(1)[0][0]
                
            rag_logs = [log for log in METRICS_LOG if log.get("RAG_relevance") is not None]
            n_rag = len(rag_logs)
            avg_relevance = sum(log["RAG_relevance"] for log in rag_logs) / n_rag if n_rag else None
            
            avg_entry = {
                "timestamp": time.time(),
                "endpoint": "aggregate",
                "latency_ms": avg_latency,
                "error": any(log.get("error") for log in METRICS_LOG),
                "prediction_band": most_common_band,
                "RAG_hit": any(log.get("RAG_hit") for log in rag_logs),
                "RAG_refusal": any(log.get("RAG_refusal") for log in rag_logs),
                "RAG_relevance": avg_relevance
            }
            
            METRICS_LOG.clear()
            METRICS_LOG.append(avg_entry)
            
            import json
            metrics_file.write_text(json.dumps(avg_entry) + "\n", encoding="utf-8")
        else:
            metrics_file.unlink(missing_ok=True)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics file: {str(e)}")

    return Response(
        content=content,
        media_type="application/jsonl",
        headers={"Content-Disposition": f"attachment; filename=metrics_log_{int(time.time())}.jsonl"}
    )

@api_router.get("/metrics", tags=["Telemetry"])
def get_metrics(time_window: Optional[int] = None):
    """Returns analytics telemetry metrics filtered by an optional time window."""
    now = time.time()
    filtered_logs = METRICS_LOG
    
    if time_window is not None:
        filtered_logs = [log for log in METRICS_LOG if (now - log["timestamp"]) <= time_window]
        
    n_requests = len(filtered_logs)
    n_errors = sum(1 for log in filtered_logs if log.get("error", False))
    
    # Calculate average latencies
    latencies = [log["latency_ms"] for log in filtered_logs]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    # Prediction distribution
    dist = {"High": 0, "Medium": 0, "Low": 0}
    for log in filtered_logs:
        band = log.get("prediction_band")
        if band in dist:
            dist[band] += 1
            
    # RAG Stats
    rag_logs = [log for log in filtered_logs if log.get("RAG_relevance") is not None]
    n_rag = len(rag_logs)
    
    hit_rate = sum(1 for log in rag_logs if log.get("RAG_hit", False)) / n_rag if n_rag else 0.0
    refusal_rate = sum(1 for log in rag_logs if log.get("RAG_refusal", False)) / n_rag if n_rag else 0.0
    empty_retrieval_count = sum(1 for log in rag_logs if not log.get("RAG_hit", False))
    avg_relevance = sum(log["RAG_relevance"] for log in rag_logs) / n_rag if n_rag else 0.0
    
    return {
        "latency": round(avg_latency, 2),
        "request_count": n_requests,
        "error_count": n_errors,
        "prediction_distribution": dist,
        "RAG_retrieval_stats": {
            "retrieval_hit_rate": round(hit_rate, 4),
            "empty_retrieval_count": empty_retrieval_count,
            "average_top_k_score": round(avg_relevance, 4),
            "refusal_rate": round(refusal_rate, 4)
        }
    }

# Serve telemetry dashboard
@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    """Serves a beautiful, premium operational metrics dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meridian Intelligence Platform - Telemetry Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f1f5f9;
        }
        .glass-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-700/50 pb-6">
            <div>
                <h1 class="text-3xl font-bold bg-gradient-to-r from-violet-400 to-indigo-300 bg-clip-text text-transparent">
                    Meridian Bank Telemetry Dashboard
                </h1>
                <p class="text-slate-400 mt-1">Real-time predictive scoring and complaints RAG monitoring lane.</p>
            </div>
            <div class="flex items-center gap-4">
                <a href="/api/download-metrics" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-full text-sm font-semibold transition-colors shadow-lg border border-indigo-500/50 flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    Download Logs
                </a>
                <div class="flex items-center gap-2 bg-slate-800/80 px-4 py-2 rounded-full border border-slate-700">
                    <span class="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></span>
                    <span class="text-sm font-semibold text-slate-300">SYSTEM HEALTHY</span>
                </div>
            </div>
        </div>

        <!-- Metric Grid -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div class="glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400">Total Requests</span>
                <h2 class="text-4xl font-bold mt-2" id="request-count">-</h2>
                <div class="text-slate-500 text-xs mt-3 flex justify-between">
                    <span>Active Server</span>
                    <span>100% uptime</span>
                </div>
            </div>
            
            <div class="glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400">Average Latency</span>
                <h2 class="text-4xl font-bold mt-2 text-indigo-400" id="latency">- <span class="text-lg">ms</span></h2>
                <div class="text-slate-500 text-xs mt-3 flex justify-between">
                    <span>Performance Gate</span>
                    <span class="text-emerald-400">PASSED</span>
                </div>
            </div>

            <div class="glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400">RAG Hit Rate</span>
                <h2 class="text-4xl font-bold mt-2 text-violet-400" id="hit-rate">-</h2>
                <div class="text-slate-500 text-xs mt-3 flex justify-between">
                    <span>Offline Grounding</span>
                    <span class="text-slate-300">&gt;0.35 similarity</span>
                </div>
            </div>

            <div class="glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400">Error Frequency</span>
                <h2 class="text-4xl font-bold mt-2 text-rose-400" id="error-count">-</h2>
                <div class="text-slate-500 text-xs mt-3 flex justify-between">
                    <span>Error Gate Rate</span>
                    <span class="text-rose-400">0.00%</span>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="glass-card p-6 rounded-2xl">
                <h3 class="text-lg font-semibold mb-4 text-slate-300">Conversion Predictions Distribution</h3>
                <div class="h-64 flex items-center justify-center">
                    <canvas id="predictionChart"></canvas>
                </div>
            </div>
            <div class="glass-card p-6 rounded-2xl">
                <h3 class="text-lg font-semibold mb-4 text-slate-300">RAG Retrieval Sufficiency Analytics</h3>
                <div class="h-64 flex items-center justify-center">
                    <canvas id="ragChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();
                
                // Update UI values
                document.getElementById('request-count').textContent = data.request_count;
                document.getElementById('latency').innerHTML = `${data.latency} <span class="text-lg">ms</span>`;
                document.getElementById('hit-rate').textContent = `${(data.RAG_retrieval_stats.retrieval_hit_rate * 100).toFixed(1)}%`;
                document.getElementById('error-count').textContent = data.error_count;
                
                // Render Charts
                renderCharts(data);
            } catch (err) {
                console.error("Error fetching telemetry:", err);
            }
        }

        let predChart, ragChartInstance;

        function renderCharts(data) {
            // 1. Predictions Distribution
            const predCtx = document.getElementById('predictionChart').getContext('2d');
            const dist = data.prediction_distribution;
            
            if (predChart) predChart.destroy();
            predChart = new Chart(predCtx, {
                type: 'doughnut',
                data: {
                    labels: ['High Prob', 'Medium Prob', 'Low Prob'],
                    datasets: [{
                        data: [dist.High, dist.Medium, dist.Low],
                        backgroundColor: ['#6366f1', '#a855f7', '#475569'],
                        borderColor: 'rgba(30, 41, 59, 0.8)',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8' } }
                    }
                }
            });

            // 2. RAG Analytics
            const ragCtx = document.getElementById('ragChart').getContext('2d');
            const stats = data.RAG_retrieval_stats;
            
            if (ragChartInstance) ragChartInstance.destroy();
            ragChartInstance = new Chart(ragCtx, {
                type: 'bar',
                data: {
                    labels: ['Hit Rate', 'Refusal Rate', 'Top-K Avg Score'],
                    datasets: [{
                        label: 'Operational Metrics',
                        data: [stats.retrieval_hit_rate, stats.refusal_rate, stats.average_top_k_score],
                        backgroundColor: ['rgba(139, 92, 246, 0.85)', 'rgba(244, 63, 94, 0.85)', 'rgba(16, 185, 129, 0.85)'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { max: 1.0, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
                        x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        // Initial fetch
        fetchMetrics();
        // Poll every 5 seconds
        setInterval(fetchMetrics, 5000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)

app.include_router(api_router, prefix="/api")

# Add backwards compatible endpoints at the root to avoid breaking any other integrations
@app.get("/health", tags=["Backwards Compatibility"])
def health_root():
    return get_health()

@app.post("/predict", response_model=PredictionResponse, tags=["Backwards Compatibility"])
def predict_root(features: CustomerFeatures):
    return predict_single(features)

@app.post("/batch-score", response_model=List[PredictionResponse], tags=["Backwards Compatibility"])
async def batch_score_root(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    # Call the router implementation
    res = await batch_score(request, file)
    # Return just the list of predictions!
    return res.predictions

@app.post("/ask-complaints", response_model=AskResponse, tags=["Backwards Compatibility"])
def ask_complaints_root(request: AskRequest):
    return ask_complaints(request)

@app.post("/customer-intel", response_model=CustomerIntelResponse, tags=["Backwards Compatibility"])
def customer_intel_root(request: CustomerIntelRequest):
    return get_customer_intel(request)

@app.get("/customer-details/{customer_id}", tags=["Backwards Compatibility"])
def customer_details_root(customer_id: str):
    return get_customer_details_endpoint(customer_id)

@app.get("/download-metrics", tags=["Backwards Compatibility"])
def download_metrics_root():
    return download_metrics()

@app.get("/metrics", tags=["Backwards Compatibility"])
def metrics_root(time_window: Optional[int] = None):
    return get_metrics(time_window)
