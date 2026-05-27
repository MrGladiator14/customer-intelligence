# Meridian Customer Intelligence Platform - Service Hardening Plan

This document outlines the concrete steps taken and planned to harden the Customer Intelligence services for enterprise production deployment.

---

## 1. PII Redaction & Data Auditing
* **Current Safeguard**: We implemented a regex-based `clean_complaint_text` function in `src/data_pipeline/features.py` redacting standard emails, SSNs, credit card numbers, and phone numbers.
* **Hardening Action**: Replace regex filters with a dedicated Named Entity Recognition (NER) model (e.g. Presidio Analyzer) to catch names, addresses, and physical locations before data storage.

---

## 2. API Scalability & Concurrency
* **Current Safeguard**: Lightweight async FastAPI framework executing inference in standard process pools.
* **Hardening Action**:
  * Run FastAPI behind a high-performance proxy like NGINX or Traefik.
  * Use Gunicorn with Uvicorn workers (`uvicorn.workers.UvicornWorker`) to load-balance across multiple CPU cores.
  * Deploy horizontal pod autoscaling (HPA) on Kubernetes, scaling on CPU/latency thresholds.

---

## 3. RAG & LLM Upstream Resilience
* **Current Safeguard**: Stateful LangGraph conditional routes and graceful mock fallbacks when API keys are exhausted.
* **Hardening Action**:
  * Implement standard retry logic with exponential backoff on NVIDIA AI Foundation Endpoints.
  * Establish a circuit breaker pattern: if upstream LLM latency crosses 2000ms or returns 502/504 errors repeatedly, fail-over to a lightweight local model or cached response database.
