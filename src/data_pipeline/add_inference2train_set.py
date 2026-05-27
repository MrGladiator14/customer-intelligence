import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import sqlite3
from src.config import SQLITE_DB_PATH, DATA_DIR

def process_pending_queries():
    """
    Reads pending support queries from the database, appends them to train.csv,
    and clears them from the database.
    """
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    cursor = conn.cursor()
    
    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            query TEXT,
            response TEXT
        )
    """)
    
    cursor.execute("SELECT id, customer_id, query, response FROM support_queries")
    rows = cursor.fetchall()
    
    if not rows:
        print("No pending queries to add.")
        conn.close()
        return

    csv_path = DATA_DIR / "train.csv"
    if not csv_path.exists():
        print(f"Error: Dataset {csv_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    new_doc_id = df["doc_id"].max() + 1 if "doc_id" in df.columns and len(df) > 0 else 0
    
    new_rows = []
    ids_to_delete = []

    for row in rows:
        record_id, customer_id, query, response = row
        
        # Look up customer
        cursor.execute("""
            SELECT age, education, job, balance, duration 
            FROM customers WHERE customer_id = ?
        """, (customer_id,))
        cust = cursor.fetchone()
        
        if cust:
            age, education, job, balance, duration = cust
        else:
            age, education, job, balance, duration = 0, "unknown", "unknown", 0.0, 0
            
        new_rows.append({
            "doc_id": new_doc_id,
            "customer_id": customer_id,
            "age": age,
            "education": education,
            "job": job,
            "balance": balance,
            "duration": duration,
            "complaint": query,
            "support_response": response,
            "converted": 0
        })
        new_doc_id += 1
        ids_to_delete.append(record_id)

    # Append to CSV
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Delete processed from DB
    cursor.executemany("DELETE FROM support_queries WHERE id = ?", [(i,) for i in ids_to_delete])
    conn.commit()
    conn.close()
    
    print(f"Successfully processed and added {len(new_rows)} queries to {csv_path}.")

if __name__ == "__main__":
    process_pending_queries()
