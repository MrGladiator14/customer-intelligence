#!/usr/bin/env python3
"""
Meridian Customer Intelligence Platform - Local Service Test Script.
This script performs verification checks against the locally running service
to ensure all API endpoints are fully operational and conform to specs.
"""

import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://127.0.0.1:8000"

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def send_request(method, path, data=None, headers=None, is_json=True):
    url = f"{BASE_URL}{path}"
    req_headers = headers or {}
    if is_json and data is not None:
        req_data = json.dumps(data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    elif data is not None:
        req_data = data
    else:
        req_data = None
        
    req = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            latency = (time.time() - start_time) * 1000
            res_data = response.read()
            status_code = response.status
            
            print(f"[{method}] {path} -> Status: {status_code} ({latency:.1f}ms)")
            
            # Try parsing JSON
            try:
                parsed_json = json.loads(res_data.decode("utf-8"))
                return status_code, parsed_json
            except json.JSONDecodeError:
                return status_code, res_data.decode("utf-8")
                
    except urllib.error.HTTPError as e:
        latency = (time.time() - start_time) * 1000
        err_data = e.read().decode("utf-8")
        print(f"[{method}] {path} -> Failed! Status: {e.code} ({latency:.1f}ms)")
        try:
            parsed_err = json.loads(err_data)
            print("Error Details:", json.dumps(parsed_err, indent=2))
        except json.JSONDecodeError:
            print("Error Body:", err_data)
        return e.code, None
    except urllib.error.URLError as e:
        print(f"[{method}] {path} -> Network Error: {e.reason}")
        print("Is the service running on http://127.0.0.1:8000?")
        return None, None

def run_tests():
    print_section("1. Testing Health Endpoint (/api/health)")
    status, res = send_request("GET", "/api/health")
    if res:
        print(json.dumps(res, indent=2))
        
    print_section("2. Testing Single Prediction (/api/predict)")
    sample_customer = {
        "customer_id": "CUST1001",
        "age": 35,
        "education": "tertiary",
        "job": "management",
        "balance": 2500.50,
        "duration": 420,
        "complaint": "Unexpected charge on my statement without prior notification. [Doc-101]"
    }
    status, res = send_request("POST", "/api/predict", data=sample_customer)
    if res:
        print(json.dumps(res, indent=2))

    print_section("3. Testing RAG Complaint Query (/api/ask-complaints)")
    sample_query = {
        "question": "What issues do customers have with unexpected charges or fees?",
        "product": "Credit card"
    }
    status, res = send_request("POST", "/api/ask-complaints", data=sample_query)
    if res:
        print(json.dumps(res, indent=2))

    print_section("4. Testing Unified Customer Intelligence (/api/customer-intel)")
    sample_intel = {
        "customer": sample_customer,
        "question": "Does this customer have complaints about statement fees?",
        "product": "Credit card"
    }
    status, res = send_request("POST", "/api/customer-intel", data=sample_intel)
    if res:
        print(json.dumps(res, indent=2))

    print_section("5. Testing Batch JSON Scoring (/api/batch-score)")
    batch_json = {
        "records": [
            sample_customer,
            {
                "customer_id": "CUST1002",
                "age": 45,
                "education": "secondary",
                "job": "blue-collar",
                "balance": 500.00,
                "duration": 90,
                "complaint": "Mobile application crashes immediately when selecting transfer option."
            }
        ]
    }
    status, res = send_request("POST", "/api/batch-score", data=batch_json)
    if res:
        # Extract predictions array (handles both BatchScoreResponse and flat array)
        predictions = res if isinstance(res, list) else res.get("predictions", [])
        predictions_count = len(predictions)
        summary = {
            "scored_file_path": None if isinstance(res, list) else res.get("scored_file_path"),
            "counts_by_conversion_band": None if isinstance(res, list) else res.get("counts_by_conversion_band"),
            "predictions_count": predictions_count
        }
        print("Batch Response Summary:")
        print(json.dumps(summary, indent=2))

    print_section("6. Testing Telemetry Metrics Endpoint (/api/metrics)")
    status, res = send_request("GET", "/api/metrics")
    if res:
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    print("=== Meridian Local Service Test Suite ===")
    run_tests()
