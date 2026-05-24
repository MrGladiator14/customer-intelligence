"""Unit tests for Pandera Schema Validation."""

import pandas as pd
import pytest
import pandera as pa
from src.data_pipeline.validate import validate_dataframe

def test_validation_success():
    # Valid customer record
    valid_data = pd.DataFrame([{
        "customer_id": "CUST123",
        "age": 35,
        "education": "tertiary",
        "job": "management",
        "balance": 2500.50,
        "duration": 180,
        "complaint": "Valid complaint text regarding credit card billing. [Doc-999]",
        "converted": 1
    }])
    
    validated_df = validate_dataframe(valid_data)
    assert validated_df is not None
    assert len(validated_df) == 1
    assert validated_df["age"].iloc[0] == 35

def test_validation_negative_age():
    # Invalid: negative age
    invalid_data = pd.DataFrame([{
        "customer_id": "CUST123",
        "age": -5,  # Violates age > 0 check
        "education": "tertiary",
        "job": "management",
        "balance": 2500.50,
        "duration": 180,
        "complaint": "Valid complaint text.",
        "converted": 1
    }])
    
    with pytest.raises(pa.errors.SchemaErrors):
        validate_dataframe(invalid_data)

def test_validation_invalid_education():
    # Invalid: education category not in standard list
    invalid_data = pd.DataFrame([{
        "customer_id": "CUST123",
        "age": 35,
        "education": "phd",  # Violates isin check
        "job": "management",
        "balance": 2500.50,
        "duration": 180,
        "complaint": "Valid complaint text.",
        "converted": 1
    }])
    
    with pytest.raises(pa.errors.SchemaErrors):
        validate_dataframe(invalid_data)

def test_validation_empty_complaint():
    # Invalid: empty complaint string
    invalid_data = pd.DataFrame([{
        "customer_id": "CUST123",
        "age": 35,
        "education": "tertiary",
        "job": "management",
        "balance": 2500.50,
        "duration": 180,
        "complaint": "   ",  # Violates non-empty string check
        "converted": 1
    }])
    
    with pytest.raises(pa.errors.SchemaErrors):
        validate_dataframe(invalid_data)

def test_validation_negative_duration():
    # Invalid: negative duration
    invalid_data = pd.DataFrame([{
        "customer_id": "CUST123",
        "age": 35,
        "education": "tertiary",
        "job": "management",
        "balance": 2500.50,
        "duration": -20,  # Violates ge(0) check
        "complaint": "Valid complaint text.",
        "converted": 1
    }])
    
    with pytest.raises(pa.errors.SchemaErrors):
        validate_dataframe(invalid_data)
