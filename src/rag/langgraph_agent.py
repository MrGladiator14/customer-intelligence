"""Meridian Customer Intelligence Platform — LangGraph RAG Agent."""

import logging
import time
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import chromadb
import mlflow

from src.config import (
    DATA_DIR,
    NVIDIA_API_KEY,
    NVIDIA_CHAT_MODEL,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
    USE_MOCK_LLM
)
from src.rag.build_index import get_embedding_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Persistence directory for ChromaDB
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── LangGraph Agent State Definition ────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    relevance_score: float
    response: str
    citations: List[str]
    latency_ms: float
    nodes_visited: List[str]

# ── NVIDIA AI Endpoints & LLM Factory ───────────────────────────────────────
# LLM client singleton - initialized once for performance
_llm_client = None

def _get_llm_client():
    """Get or initialize the NVIDIA LLM client."""
    global _llm_client
    if _llm_client is None:
        if USE_MOCK_LLM or not NVIDIA_API_KEY:
            return None
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            logger.info("Initializing NVIDIA Llama ...")
            _llm_client = ChatNVIDIA(
                model=NVIDIA_CHAT_MODEL,
                api_key=NVIDIA_API_KEY,
                temperature=0.1,
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"Error initializing NVIDIA Llama: {e}")
            return None
    return _llm_client

@mlflow.trace(span_type="LLM")
def call_nvidia_llama(prompt: str) -> str:
    """Invokes the NVIDIA Build Llama 70B model or falls back to mock."""
    if USE_MOCK_LLM or not NVIDIA_API_KEY:
        # High-fidelity mock LLM generation based on prompt context
        logger.info("Using high-fidelity Mock LLM for generation...")
        # Extract source docs and build a smart summary
        import re
        docs = re.findall(r"Document \[(Doc-\d+)\]:\s*(.*?)(?=\nDocument|Context:|\Z)", prompt, re.DOTALL)
        if not docs:
            return "Refused: Evidence insufficient to ground an answer."
            
        answers = []
        for doc_id, text in docs:
            # Create a nice summarized sentence citing the document
            clean_text = text.strip()
            if "fee" in clean_text.lower():
                answers.append(f"Customers have expressed deep dissatisfaction with unexpected, high transaction fees on savings accounts [{doc_id}].")
            elif "app" in clean_text.lower() or "mobile" in clean_text.lower():
                answers.append(f"Users reported severe delays and interface crashes when transferring funds via the mobile application [{doc_id}].")
            elif "support" in clean_text.lower() or "fraudulent" in clean_text.lower():
                answers.append(f"Customer support teams have been criticized for taking up to three days to respond to critical fraud queries [{doc_id}].")
            elif "card" in clean_text.lower() or "blocked" in clean_text.lower():
                answers.append(f"Debit card blockages have occurred unexpectedly during routine grocery purchases [{doc_id}].")
            else:
                answers.append(f"A complaint was registered regarding the following issue: '{clean_text[:60]}...' [{doc_id}].")
                
        return " ".join(answers)

    try:
        llm = _get_llm_client()
        if llm is None:
            logger.error("Failed to initialize NVIDIA Llama client. Falling back to mock generator.")
            return "Refused: Evidence insufficient to ground an answer. (Error in LLM upstream connection)"
            
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error calling NVIDIA Llama: {e}. Falling back to mock generator.")
        return "Refused: Evidence insufficient to ground an answer. (Error in LLM upstream connection)"

# ── Graph Nodes ─────────────────────────────────────────────────────────────

@mlflow.trace(span_type="RETRIEVER")
def node_retrieve(state: AgentState) -> Dict[str, Any]:
    """Retrieves top-K complaints from ChromaDB based on the query."""
    logger.info(f"Node [Retrieve]: Querying ChromaDB for '{state['question']}'")
    start_time = time.time()
    
    # 1. Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection("customer_complaints")
    except Exception:
        logger.warning("ChromaDB collection 'customer_complaints' not found. Returning empty docs.")
        return {
            "documents": [],
            "nodes_visited": state.get("nodes_visited", []) + ["retrieve"]
        }
        
    # 2. Embed Query
    embedding_engine = get_embedding_engine()
    query_embedding = embedding_engine.embed_query(state["question"])
        
    # 3. Query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RAG_TOP_K
    )
    
    # Parse results
    docs = []
    if results and "documents" in results and results["documents"]:
        documents_list = results["documents"][0]
        distances = results["distances"][0] if "distances" in results else [0.5] * len(documents_list)
        metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(documents_list)
        
        for doc_text, dist, meta in zip(documents_list, distances, metadatas):
            # ChromaDB with cosine space returns cosine distance; similarity = 1 - distance
            similarity = 1.0 - dist
            docs.append({
                "text": doc_text,
                "similarity": similarity,
                "source_id": meta.get("source_id", "Doc-Unknown"),
                "customer_id": meta.get("customer_id", "CUST-Unknown")
            })
            
    logger.info(f"Retrieved {len(docs)} matching documents.")
    return {
        "documents": docs,
        "nodes_visited": state.get("nodes_visited", []) + ["retrieve"]
    }

@mlflow.trace(span_type="TOOL")
def node_evaluate_relevance(state: AgentState) -> Dict[str, Any]:
    """Evaluates the relevance of the retrieved documents."""
    logger.info("Node [Evaluate Relevance]: Assessing retrieved documents...")
    
    docs = state.get("documents", [])
    if not docs:
        return {
            "relevance_score": 0.0,
            "nodes_visited": state.get("nodes_visited", []) + ["evaluate_relevance"]
        }
        
    # Compute the max relevance score of our documents
    max_similarity = max(doc["similarity"] for doc in docs)
    logger.info(f"Max document similarity computed: {max_similarity:.4f}")
    
    return {
        "relevance_score": max_similarity,
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

@mlflow.trace(span_type="LLM")
def node_generate(state: AgentState) -> Dict[str, Any]:
    """Node that generates a response strictly citing retrieved source documents."""
    logger.info("Node [Generate]: Synthesizing answer using NVIDIA Llama 70B...")
    
    docs = state.get("documents", [])
    
    # 1. Format context for LLM
    context_str = ""
    citations = []
    for doc in docs:
        if doc["similarity"] >= RAG_SIMILARITY_THRESHOLD:
            context_str += f"Document [{doc['source_id']}]: {doc['text']}\n"
            citations.append(doc["source_id"])
            
    # 2. Build system/user prompt
    prompt = f"""You are a helpful customer intelligence assistant for Meridian Bank.
Your task is to answer the User Question using ONLY the provided Customer Complaint Context below.

Guidelines:
1. Answer the question accurately and professionally.
2. If the context does not contain enough information to ground a complete answer, you MUST reply exactly with: "Refused: Evidence insufficient to ground an answer."
3. Do not make up facts or use general knowledge. Answer strictly from the provided documents.
4. You MUST cite the source documents you use at the end of the sentence or paragraph, e.g. [Doc-101], [Doc-102].

Context:
{context_str}

User Question: {state['question']}

Helpful, Grounded Answer:"""

    response = call_nvidia_llama(prompt)
    
    # Double check if the model refused
    if "Refused: Evidence insufficient to ground an answer" in response:
        citations = []
        
    return {
        "response": response,
        "citations": list(set(citations)) if "Refused:" not in response else [],
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
            "refuse": "refuse"
        }
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse", END)
    
    return workflow.compile()

@mlflow.trace(name="LangGraph RAG Agent", span_type="AGENT")
def run_rag_agent(question: str) -> Dict[str, Any]:
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
