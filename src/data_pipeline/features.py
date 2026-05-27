"""Meridian Customer Intelligence Platform - Pipeline Feature Engineering."""

import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)

# Categorical mapping dictionaries
JOB_MAPPING = {
    "management": 0, "technician": 1, "self-employed": 2, "blue-collar": 3,
    "services": 4, "retired": 5, "admin.": 6, "student": 7, "entrepreneur": 8
}

EDUCATION_MAPPING = {
    "primary": 0, "secondary": 1, "tertiary": 2, "unknown": -1
}

def clean_complaint_text(text: str) -> str:
    """Cleans complaint text while preserving metadata, redacting obvious personal details.

    E.g., redacts email addresses, SSNs, credit card numbers, and phone numbers.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Redact Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", text)
    
    # 2. Redact Phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", text)
    
    # 3. Redact Social Security Numbers (SSN)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    
    # 4. Redact standard 16 digit Credit Card numbers
    text = re.sub(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[REDACTED_CARD]", text)

    # 5. Clean extra whitespace but preserve original letters/casing
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_dataframe(df: pd.DataFrame):
    """Encodes categorical columns and returns feature matrix X and target y."""
    processed = df.copy()
    
    # Clean complaints while preserving metadata
    if "complaint" in processed.columns:
        processed["complaint"] = processed["complaint"].apply(clean_complaint_text)
        
    processed["job_code"] = processed["job"].map(JOB_MAPPING).fillna(-1).astype(int)
    processed["edu_code"] = processed["education"].map(EDUCATION_MAPPING).fillna(-1).astype(int)
    
    feature_cols = ["age", "balance", "duration", "job_code", "edu_code"]
    X = processed[feature_cols]
    y = processed["converted"] if "converted" in processed.columns else None
    return X, y
