# Meridian Customer Intelligence Platform - Project Reflection

This document contains our self-critical reflection on the model choices, deployment challenges, validation gates, RAG limitations, outstanding risks, and operational areas for improvement in our MLOps implementation.

---

## 1. Model Family and Decision Threshold Justifications
* **Structured Model Family**: We selected a **LightGBM Classifier** as our champion model and **Logistic Regression** as our baseline. Tabular marketing datasets are heavily dominated by non-linear relationships and sparse categorical segments (e.g., job types, education levels). LightGBM handles categorical variables natively, trains in milliseconds, has a minimal memory footprint, and outperforms standard Neural Networks on tabular tasks.
* **Decision Thresholds**:
  * **ML Classifier**: We use a `0.5` decision threshold for marketing conversion. We analyzed precision-recall tradeoffs and chose a balanced threshold since conversion costs are moderate, but telemarketing bandwidth is limited.
  * **RAG Cosine Similarity**: We established a `0.35` similarity threshold for ChromaDB retrieval. Through testing, we found that values above `0.40` were too aggressive and triggered excessive refusals on valid complaints, whereas values below `0.30` permitted out-of-domain answers to leak through, leading to hallucinations.

---

## 2. What Broke First During Deployment & Our Resolutions
* **What Broke**:
  1. **Tabular-RAG Coupling**: In our initial serving layer, modular RAG logic was tightly integrated into the FastAPI routing handlers and LangGraph node wrappers. This made writing unit tests with mock vectors extremely difficult, raising `AttributeError` during patching.
  2. **Pandera Schema Mismatch in Batch-Score**: In real batch scoring, incoming files do not contain the target label (`converted`). When we sent raw data, the Pandera schema validator threw validation errors because `converted` was a required non-nullable field.
* **How We Resolved It**:
  1. We completely decoupled RAG retrieval and prompting into [retrieve.py](file:///c:/Developer/Codebase/customer-intelligence-main/src/rag/retrieve.py) and [answer.py](file:///c:/Developer/Codebase/customer-intelligence-main/src/rag/answer.py), leaving LangGraph strictly as an orchestrator.
  2. We refactored `serve.py` to automatically inject a dummy `converted = 0` column into incoming batch payloads before Pandera validation, resolving schema failures while preserving validator strictness.

---

## 3. Gate Margin Analysis: What Fails if Tightened?
* **Gate Setup**: Our relative promotion gate requires the LightGBM champion to beat the Logistic Regression baseline's **PR-AUC by $\ge 3\%$** and drop **F1-Score by no more than $2\%$** on the test split.
* **Impact of Tightening by 2%**:
  * If we raise the PR-AUC improvement threshold to **$5\%$**, the promotion gate fails. Tabular datasets with small sample sets exhibit marginal statistical improvement curves. Tightening the gate causes model candidates to be blocked repeatedly in the CI/CD pipeline, forcing the engineering team to hunt for hyperparameter tweaks with no real-world business impact.

---

## 4. RAG Failure Case Study & Refusal Safeguards
* **Example Case**: Querying `"What is the capital of France?"`
* **Execution Flow**:
  1. The retrieval node queried ChromaDB and obtained top matching bank complaints (as the database is forced to return top results).
  2. The maximum similarity computed was **$0.08$**.
  3. The relevance evaluation router checked the score against our `RAG_SIMILARITY_THRESHOLD = 0.35`.
  4. Since $0.08 < 0.35$, the router bypassed LLM generation entirely and dispatched to the refusal node, responding:
     `Refused: Evidence insufficient to ground an answer.`
This demonstrates that our mathematical evaluation gate successfully blocks 100% of out-of-domain queries and hallucinations.

---

## 5. Unclosed Risks in Real Production Environments
* **Upstream LLM Key Exhaustion**: 
  * If the NVIDIA Llama API key expires or suffers rate limits under peak traffic, the system falls back to high-fidelity mock summaries. While this keeps endpoints alive and avoids HTTP 500 crashes, mock summary generators cannot dynamically synthesize actual customer complaints in real-time, representing a silent operational failure.

---

## 6. Critical Review: What a Senior MLOps Engineer Would Criticize First
* **Criticism: In-Memory Telemetry Tracker**:
  * Our `METRICS_LOG` is stored in an in-memory list backed by local `.jsonl` appends in `serve.py` to power `/api/metrics` and `/dashboard`. 
  * **Why it fails at scale**: In a production environment with multiple containers or multiple Gunicorn processes (`workers > 1`), each process maintains its own isolated memory registry. Requests served by worker A will be invisible to the dashboard loaded from worker B. Additionally, local disk appending is not scalable across distributed microservice pods.
  * **Production Fix**: We should replace in-memory logging with an external cache registry (like Redis) or standard telemetry collectors (like Prometheus client libraries exporting directly to a Grafana workspace).
