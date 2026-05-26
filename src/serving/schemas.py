"""Meridian Customer Intelligence Platform - Pydantic Schemas."""

from typing import List, Optional
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

class BatchPredictRequest(BaseModel):
    records: List[CustomerFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResponse]

# ── RAG / LangGraph Agent Schemas ───────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The query regarding customer complaints", examples=["Why was my debit card blocked?"])

class AskResponse(BaseModel):
    question: str
    response: str = Field(..., description="Grounded answer from the LangGraph agent or refusal string")
    citations: List[str] = Field(..., description="Structural source documents cited, e.g. ['Doc-101']")
    latency_ms: float = Field(..., description="Total execution latency in milliseconds")
    relevance_score: float = Field(..., description="Computed maximum relevance score for retrieved complaints")

# ── Unified Combined Schema ─────────────────────────────────────────────────
class CustomerIntelRequest(BaseModel):
    customer: CustomerFeatures
    question: str = Field(..., min_length=3, description="Query for aggregate complaint insights", examples=["What was their specific complaint about card lockages?"])

class CustomerIntelResponse(BaseModel):
    customer_id: str
    conversion_info: PredictionResponse
    complaint_insights: AskResponse
