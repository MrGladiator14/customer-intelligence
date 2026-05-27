"""Meridian Customer Intelligence Platform - Database & Caching Services."""

import csv
import json
import sqlite3
import logging
from typing import Optional, Dict, Any
from src.config import DATA_DIR, SQLITE_DB_PATH, REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger(__name__)

# Redis connection setup with graceful in-memory fallback
REDIS_AVAILABLE = False
redis_client = None
IN_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}

try:
    import redis
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=1
    )
    # Ping to verify active connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}. Caching enabled.")
except Exception as e:
    logger.warning(f"Redis is not available ({e}). Falling back to in-memory dictionary caching.")

def init_db():
    """Initializes the SQLite database and pre-populates it with data from CSV files."""
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    cursor = conn.cursor()
    
    # Create customers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            age INTEGER,
            education TEXT,
            job TEXT,
            balance REAL,
            duration INTEGER,
            complaint TEXT
        )
    """)
    conn.commit()
    
    # Ingest existing records from train.csv and test.csv if table is empty
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        logger.info("Customers database table is empty. Pre-populating from local CSV files...")
        
        for filename in ["train.csv", "test.csv"]:
            csv_path = DATA_DIR / filename
            if csv_path.exists():
                try:
                    with open(csv_path, mode="r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cust_id = row.get("customer_id")
                            if not cust_id:
                                continue
                            
                            age = int(row["age"]) if row.get("age") else 0
                            edu = row.get("education", "")
                            job = row.get("job", "")
                            bal = float(row["balance"]) if row.get("balance") else 0.0
                            dur = int(row["duration"]) if row.get("duration") else 0
                            comp = row.get("complaint", "")
                            
                            cursor.execute("""
                                INSERT OR IGNORE INTO customers (customer_id, age, education, job, balance, duration, complaint)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (cust_id, age, edu, job, bal, dur, comp))
                    conn.commit()
                    logger.info(f"Loaded records from {filename} into SQLite.")
                except Exception as e:
                    logger.error(f"Error pre-populating database from {filename}: {e}")
    else:
        logger.info("SQLite database table already populated.")
        
    conn.close()

def get_customer_details(customer_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves customer details by customer_id using Redis cache or SQLite DB."""
    cache_key = f"customer:{customer_id}"
    
    # 1. Try Cache
    if REDIS_AVAILABLE and redis_client is not None:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache HIT for customer key: '{cache_key}'")
                parsed = json.loads(cached_data)
                parsed["cache_status"] = "HIT"
                return parsed
        except Exception as e:
            logger.warning(f"Error reading from Redis cache: {e}")
    else:
        if customer_id in IN_MEMORY_CACHE:
            logger.info(f"In-Memory Cache HIT for customer key: '{cache_key}'")
            parsed = IN_MEMORY_CACHE[customer_id].copy()
            parsed["cache_status"] = "HIT"
            return parsed
            
    # 2. Try DB
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT customer_id, age, education, job, balance, duration, complaint
            FROM customers
            WHERE customer_id = ?
        """, (customer_id,))
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        logger.error(f"SQLite query failed: {e}")
        return None
        
    if row:
        data = {
            "customer_id": row[0],
            "age": row[1],
            "education": row[2],
            "job": row[3],
            "balance": row[4],
            "duration": row[5],
            "complaint": row[6]
        }
        logger.info(f"Cache MISS for customer key: '{cache_key}'. Loaded from SQLite.")
        
        # Write to cache
        if REDIS_AVAILABLE and redis_client is not None:
            try:
                redis_client.setex(cache_key, 3600, json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to save to Redis cache: {e}")
        else:
            IN_MEMORY_CACHE[customer_id] = data.copy()
            
        data["cache_status"] = "MISS"
        return data
        
    logger.info(f"Customer '{customer_id}' not found in SQLite DB.")
    return None
