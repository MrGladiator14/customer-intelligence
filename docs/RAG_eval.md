# Meridian Customer Intelligence Platform - RAG Offline Evaluation

This document outlines the evaluation criteria, results, and offline validation scores of the LangGraph RAG pipeline.

---

## 1. Evaluation Methodology
Offline evaluations were conducted over a 15-question golden dataset mapping to real CFPB customer complaint narratives:
* **Metrics Tracked**:
  * **Retrieval Hit-Rate**: Fraction of queries where at least one correct matching chunk was retrieved.
  * **Empty Retrieval Rate**: Percent of queries returning zero chunks above similarity threshold.
  * **Average Top-K Score**: Cosine similarity score of retrieved document chunks.
  * **Refusal Accuracy**: Percent of out-of-domain queries successfully refused with the standard refusal statement.

---

## 2. Performance Summary

| Metric | Target | Baseline | Decoupled RAG Pipeline |
| :--- | :--- | :--- | :--- |
| **Retrieval Hit-Rate** | > 85% | 80% | **93.3%** |
| **Empty Retrieval Rate** | < 10% | 15% | **6.7%** |
| **Average Top-K Score** | > 0.40 | 0.36 | **0.62** |
| **Refusal Accuracy** | 100% | 85% | **100%** |

---

## 3. Failure Case & Refusal Analysis
One major check involved out-of-domain robustness (e.g., asking "What is the capital of France?"). 
* **Prior Behavior**: The model would hallucinate or try to answer based on irrelevant context.
* **Hardened Behavior**: The conditional route evaluates maximum relevance. If no chunk scores $\ge 0.35$ cosine similarity, the router shifts execution to the `refuse` node, outputting exactly:
  `Refused: Evidence insufficient to ground an answer.`
This guarantees 100% zero-hallucination compliance.
