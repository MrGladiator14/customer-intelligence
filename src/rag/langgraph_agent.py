"""Meridian Customer Intelligence Platform - LangGraph RAG Agent."""

import logging
import time
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
import mlflow

import chromadb
from src.config import RAG_SIMILARITY_THRESHOLD
from src.rag.retrieve import retrieve_complaints, evaluate_relevance
from src.rag.answer import generate_answer, rewrite_query
from src.rag.build_index import get_embedding_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── LangGraph Agent State Definition ────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    product: Optional[str]
    company: Optional[str]
    date: Optional[str]
    issue: Optional[str]
    customer_id: Optional[str]
    rewritten_question: Optional[str]
    documents: List[Dict[str, Any]]
    relevance_score: float
    response: str
    citations: List[str]
    latency_ms: float
    nodes_visited: List[str]

# ── Graph Nodes ─────────────────────────────────────────────────────────────

@mlflow.trace(span_type="RETRIEVER")
def node_retrieve(state: AgentState) -> Dict[str, Any]:
    """Retrieves top-K complaints based on query and filters."""
    logger.info(f"Node [Retrieve]: Querying ChromaDB for '{state['question']}' with customer_id '{state.get('customer_id')}'")
    docs = retrieve_complaints(
        query=state["question"],
        product=state.get("product"),
        company=state.get("company"),
        date=state.get("date"),
        issue=state.get("issue"),
        customer_id=state.get("customer_id")
    )
    return {
        "documents": docs,
        "nodes_visited": state.get("nodes_visited", []) + ["retrieve"]
    }

@mlflow.trace(span_type="TOOL")
def node_evaluate_relevance(state: AgentState) -> Dict[str, Any]:
    """Evaluates the relevance of the retrieved documents."""
    logger.info("Node [Evaluate Relevance]: Assessing retrieved documents...")
    score = evaluate_relevance(state.get("documents", []))
    return {
        "relevance_score": score,
        "nodes_visited": state.get("nodes_visited", []) + ["evaluate_relevance"]
    }

def route_relevance(state: AgentState) -> str:
    """Routes based on relevance score vs threshold."""
    score = state.get("relevance_score", 0.0)
    logger.info(f"Routing check: Relevance Score={score:.4f} (Threshold={RAG_SIMILARITY_THRESHOLD:.4f})")
    
    if score >= RAG_SIMILARITY_THRESHOLD:
        return "generate"
    else:
        return "refuse"

@mlflow.trace(span_type="TOOL")
def node_refuse(state: AgentState) -> Dict[str, Any]:
    """Node that handles out-of-domain / irrelevant questions by refusing."""
    logger.info("Node [Refuse]: Executing refusal response...")
    return {
        "response": "Refused: Evidence insufficient to ground an answer.",
        "citations": [],
        "nodes_visited": state.get("nodes_visited", []) + ["refuse"]
    }

@mlflow.trace(span_type="TOOL")
def node_rewrite_query(state: AgentState) -> Dict[str, Any]:
    """Node that rewrites the query for better retrieval."""
    logger.info("Node [Rewrite Query]: Rewriting question...")
    new_question = rewrite_query(state["question"])
    return {
        "rewritten_question": new_question,
        "nodes_visited": state.get("nodes_visited", []) + ["rewrite_query"]
    }

@mlflow.trace(span_type="RETRIEVER")
def node_re_retrieve(state: AgentState) -> Dict[str, Any]:
    """Node that re-retrieves documents using the rewritten query."""
    query = state.get("rewritten_question") or state["question"]
    logger.info(f"Node [Re-Retrieve]: Querying ChromaDB for rewritten query '{query}'")
    docs = retrieve_complaints(
        query=query,
        product=state.get("product"),
        company=state.get("company"),
        date=state.get("date"),
        issue=state.get("issue"),
        customer_id=state.get("customer_id")
    )
    return {
        "documents": docs,
        "nodes_visited": state.get("nodes_visited", []) + ["re_retrieve"]
    }

@mlflow.trace(span_type="LLM")
def node_generate(state: AgentState) -> Dict[str, Any]:
    """Node that generates a response strictly citing retrieved source documents."""
    logger.info("Node [Generate]: Synthesizing answer...")
    out = generate_answer(state["question"], state.get("documents", []))
    return {
        "response": out["response"],
        "citations": out["citations"],
        "nodes_visited": state.get("nodes_visited", []) + ["generate"]
    }

# ── Assemble LangGraph ──────────────────────────────────────────────────────

def get_rag_agent_graph():
    """Compiles and returns the LangGraph stateful workflow."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("evaluate_relevance", node_evaluate_relevance)
    workflow.add_node("refuse", node_refuse)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("re_retrieve", node_re_retrieve)
    workflow.add_node("generate", node_generate)
    
    # Set Entry Point
    workflow.set_entry_point("retrieve")
    
    # Add Edges
    workflow.add_edge("retrieve", "evaluate_relevance")
    
    # Conditional router
    workflow.add_conditional_edges(
        "evaluate_relevance",
        route_relevance,
        {
            "generate": "generate",
            "refuse": "rewrite_query"
        }
    )
    
    workflow.add_edge("rewrite_query", "re_retrieve")
    workflow.add_edge("re_retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse", END)
    
    return workflow.compile()

@mlflow.trace(name="LangGraph RAG Agent", span_type="AGENT")
def run_rag_agent(
    question: str,
    product: str = None,
    company: str = None,
    date: str = None,
    issue: str = None,
    customer_id: str = None
) -> Dict[str, Any]:
    """Wrapper function to run the compiled LangGraph and track overall latency."""
    try:
        from src.config import MLFLOW_TRACKING_URI, MLFLOW_INFERENCE_EXPERIMENT_NAME
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_INFERENCE_EXPERIMENT_NAME)
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow client inside RAG agent: {e}")
        
    start_time = time.time()
    
    graph = get_rag_agent_graph()
    initial_state = {
        "question": question,
        "product": product,
        "company": company,
        "date": date,
        "issue": issue,
        "customer_id": customer_id,
        "rewritten_question": None,
        "documents": [],
        "relevance_score": 0.0,
        "response": "",
        "citations": [],
        "latency_ms": 0.0,
        "nodes_visited": []
    }
    
    final_output = graph.invoke(initial_state)
    
    latency_ms = (time.time() - start_time) * 1000.0
    final_output["latency_ms"] = round(latency_ms, 2)
    
    logger.info(f"Agent finished in {latency_ms:.2f}ms. Node path: {' -> '.join(final_output['nodes_visited'])}")
    return final_output
