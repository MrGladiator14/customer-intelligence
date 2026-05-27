"""Unit and integration tests for SQLite DB storage and Redis Caching layer."""

import os
import pytest
from src.serving.database import init_db, get_customer_details, REDIS_AVAILABLE, IN_MEMORY_CACHE

def test_database_initialization_and_ingest():
    """Verify database initialization and pre-population work correctly."""
    # Ensure database is initialized
    init_db()
    
    # Retrieve details for a known customer from train.csv (CUST001)
    details = get_customer_details("CUST001")
    assert details is not None
    assert details["customer_id"] == "CUST001"
    assert details["age"] == 34
    assert details["education"] == "tertiary"
    assert details["job"] == "management"
    assert details["balance"] == 1500.5
    assert details["duration"] == 240
    assert "Extremely disappointed with the high transaction fees" in details["complaint"]

def test_caching_hit_and_miss_behavior():
    """Verify caching hit/miss state transition works correctly."""
    # Ensure database is initialized
    init_db()
    
    # Target CUST002 from train.csv
    customer_id = "CUST002"
    
    # Clear cache for this customer first to force a cache MISS
    if REDIS_AVAILABLE:
        from src.serving.database import redis_client
        if redis_client is not None:
            redis_client.delete(f"customer:{customer_id}")
    else:
        if customer_id in IN_MEMORY_CACHE:
            del IN_MEMORY_CACHE[customer_id]
            
    # First request: should be a Cache MISS
    res1 = get_customer_details(customer_id)
    assert res1 is not None
    assert res1["cache_status"] == "MISS"
    
    # Second request: should be a Cache HIT
    res2 = get_customer_details(customer_id)
    assert res2 is not None
    assert res2["cache_status"] == "HIT"
    assert res2["age"] == 45
    assert res2["job"] == "technician"

def test_non_existent_customer():
    """Verify querying a non-existent customer correctly returns None."""
    details = get_customer_details("CUST999_NON_EXISTENT")
    assert details is None
