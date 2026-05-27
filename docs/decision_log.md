# Meridian Customer Intelligence Platform - Architecture Decision Log

This document records the engineering decisions, trade-offs, and reasoning behind our architectural design, including rejected approaches.

---

## 1. Decision: Decoupled Modular RAG Architecture
* **Context**: The RAG lane was initially embedded inline in `langgraph_agent.py`.
* **Decision**: We decoupled RAG retrieval and response generation into standalone modules: [retrieve.py](file:///c:/Developer/Codebase/customer-intelligence-main/src/rag/retrieve.py) and [answer.py](file:///c:/Developer/Codebase/customer-intelligence-main/src/rag/answer.py).
* **Rationale**: Separating concerns allows debugging, profiling, and unit-testing the retrieval accuracy separately from the LLM generation quality.
* **Rejected Alternative**: Inline LangGraph agent nodes. Rejected because it binds operational execution logic to the graph orchestration library, rendering unit testing and metric capture highly complex.

---

## 2. Decision: Hybrid Python-Chroma Filtering for Segment Queries
* **Context**: The `POST /ask-complaints` and `POST /customer-intel` endpoints require filtering complaints on `product`, `company`, `date`, and `issue`.
* **Decision**: We query ChromaDB for a larger semantic candidate set (e.g., top 20-50 matches) and apply precise, case-insensitive keyword filtering on text and metadata in Python.
* **Rationale**: The CFPB consumer complaint dummy corpus is highly sparse and lacks explicit structural metadata headers inside local databases. Keyword-based matching over semantic retrieval candidate pools ensures high-fidelity results without database schema lock-in.
* **Rejected Alternative**: Restricting queries using rigid metadata dict matches in Chroma's `where` clause. Rejected because sparse entries fail strict exact matches, leading to empty retrievals.

---

## 3. Decision: Centralized Feature Pipeline Module
* **Context**: Categorical mappings and encoding were inline in the training script.
* **Decision**: Created a unified [features.py](file:///c:/Developer/Codebase/customer-intelligence-main/src/data_pipeline/features.py) sharing preprocessing pipelines.
* **Rationale**: Eliminates train-serve feature engineering skew, guaranteeing identical preprocessing pipelines at train and serve time.

---

## 4. Decision: Custom Pure-Python Telemetry and Dashboard
* **Context**: Needed live telemetry of latencies, predictions, and RAG hit rates, served in a dashboard.
* **Decision**: Implemented an async `.jsonl` logging telemetry engine in `serve.py` and rendered a stunning dark-mode tailwind dashboard on `/dashboard`.
* **Rationale**: Highly responsive, zero external microservice dependencies, extremely lightweight, and runs flawlessly inside local container environments.
* **Rejected Alternative**: Promoted external ELK stack or Prometheus. Rejected due to configuration overhead, resource footprints, and failure to run "out of the box" from fresh checkout.
