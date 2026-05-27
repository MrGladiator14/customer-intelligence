"""Meridian Customer Intelligence Platform - Pydantic Schemas."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# ── Prediction Schemas ──────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer ID, must start with CUST", examples=["CUST101"])
    age: int = Field(..., gt=0, description="Age of the customer, must be positive", examples=[35])
    education: str = Field(..., description="Education level", examples=["tertiary"])
    job: str = Field(..., description="Job sector", examples=["management"])
    balance: float = Field(..., description="Current account balance", examples=[2000.50])
    duration: int = Field(..., ge=0, description="Last contact duration in seconds", examples=[220])
    complaint: str = Field(..., description="Customer complaint text", examples=["Unexpected transaction fees charged."])

class PredictionResponse(BaseModel):
    customer_id: str
    conversion_prediction: int = Field(..., description="Binary prediction: 1 if customer likely to convert, 0 otherwise")
    conversion_probability: float = Field(..., description="Raw conversion probability value")
    probability_band: str = Field(..., description="Probability band: High (>= 0.7), Medium (0.3 - 0.7), Low (< 0.3)")
    
    # Mandated flat API contract fields (Optional for backward-compatible mock safety)
    prediction: Optional[int] = Field(None, description="Same as conversion_prediction")
    probability: Optional[float] = Field(None, description="Same as conversion_probability")
    threshold_decision: Optional[bool] = Field(None, description="True if prediction == 1 (convert) else False")
    model_version: Optional[str] = Field(None, description="Active model name/version used for scoring")

class BatchPredictRequest(BaseModel):
    records: List[CustomerFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResponse]

class BatchScoreResponse(BaseModel):
    scored_file_path: str = Field(..., description="The local or server file path of the scored CSV batch")
    counts_by_conversion_band: Dict[str, int] = Field(..., description="Counts of customers categorized by conversion bands")
    predictions: List[PredictionResponse] = Field(..., description="The list of prediction outputs")

# ── RAG / LangGraph Agent Schemas ───────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The query regarding customer complaints", examples=["Why was my debit card blocked?"])
    
    # Optional filters (Page 4 of PDF)
    product: Optional[str] = Field(None, description="Optional product filter", examples=["Credit card"])
    company: Optional[str] = Field(None, description="Optional company filter", examples=["Meridian Bank"])
    date: Optional[str] = Field(None, description="Optional date filter", examples=["2026-05-27"])
    issue: Optional[str] = Field(None, description="Optional issue filter", examples=["Fees"])

class AskResponse(BaseModel):
    question: str
    response: str = Field(..., description="Grounded answer from the LangGraph agent or refusal string")
    citations: List[str] = Field(..., description="Structural source documents cited, e.g. ['Doc-101']")
    latency_ms: float = Field(..., description="Total execution latency in milliseconds")
    relevance_score: float = Field(..., description="Computed maximum relevance score for retrieved complaints")
    
    # Mandated flat API contract fields (Optional for backward-compatible mock safety)
    answer: Optional[str] = Field(None, description="Grounded answer from the LangGraph agent or refusal string")
    retrieved_evidence_ids: Optional[List[str]] = Field(None, description="Structural source documents cited, same as citations")
    evidence_sufficiency_note: Optional[str] = Field(None, description="One-line evidence-sufficiency check note")
    prompt_version: Optional[str] = Field(None, description="Prompt version used inside the LLM generation step")

# ── Unified Combined Schema ─────────────────────────────────────────────────
class CustomerIntelRequest(BaseModel):
    customer: CustomerFeatures
    question: str = Field(..., min_length=3, description="Query for aggregate complaint insights", examples=["What was their specific complaint about card lockages?"])
    
    # Optional filters (Page 4 of PDF)
    product: Optional[str] = Field(None, description="Optional product filter")
    company: Optional[str] = Field(None, description="Optional company filter")
    date: Optional[str] = Field(None, description="Optional date filter")
    issue: Optional[str] = Field(None, description="Optional issue filter")

class CustomerIntelResponse(BaseModel):
    customer_id: str
    
    # Mandated flat API contract fields (Optional for backward-compatible mock safety)
    conversion_band: Optional[str] = Field(None, description="Conversion band for the customer, e.g. High, Medium, Low")
    top_complaint_themes: Optional[str] = Field(None, description="Synthesized complaint themes for this segment")
    cited_ids: Optional[List[str]] = Field(None, description="Evidence document IDs referenced in theme compilation")
    
    # Legacies kept for backwards compatibility with existing UI/tests
    conversion_info: PredictionResponse
    complaint_insights: AskResponse
