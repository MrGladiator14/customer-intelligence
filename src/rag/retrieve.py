"""Meridian Customer Intelligence Platform - RAG Retrieval Lane."""

import logging
import time
from typing import List, Dict, Any
import chromadb
from src.config import DATA_DIR, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD
from src.rag.build_index import get_embedding_engine

logger = logging.getLogger(__name__)

CHROMA_DIR = DATA_DIR / "chroma_db"

def retrieve_complaints(
    query: str,
    product: str = None,
    company: str = None,
    date: str = None,
    issue: str = None,
    limit: int = RAG_TOP_K
) -> List[Dict[str, Any]]:
    """Retrieves top-K complaints from ChromaDB based on query and optional filters."""
    logger.info(f"Retrieving complaints for query: '{query}' with filters (product={product}, company={company}, date={date}, issue={issue})")
    
    # 1. Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection("customer_complaints")
    except Exception:
        logger.warning("ChromaDB collection 'customer_complaints' not found. Returning empty docs.")
        return []
        
    # 2. Embed Query
    embedding_engine = get_embedding_engine()
    query_embedding = embedding_engine.embed_query(query)
    
    # 3. Query (Request a larger batch so we can filter in Python and still return up to top-K)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(limit * 4, 20)
    )
    
    docs = []
    if results and "documents" in results and results["documents"]:
        documents_list = results["documents"][0]
        distances = results["distances"][0] if "distances" in results else [0.5] * len(documents_list)
        metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(documents_list)
        
        for doc_text, dist, meta in zip(documents_list, distances, metadatas):
            similarity = 1.0 - dist
            
            # Formulate the document dictionary
            doc_dict = {
                "text": doc_text,
                "similarity": similarity,
                "source_id": meta.get("source_id", "Doc-Unknown"),
                "customer_id": meta.get("customer_id", "CUST-Unknown"),
                # We can store extra mocked/deduced metadata or fetch from text
                "product": meta.get("product", ""),
                "company": meta.get("company", ""),
                "date": meta.get("date", ""),
                "issue": meta.get("issue", "")
            }
            
            # Apply optional filters in Python (case-insensitive keyword matching on text)
            if product and product.lower() not in doc_text.lower():
                continue
            if company and company.lower() not in doc_text.lower():
                continue
            if date and date.lower() not in doc_text.lower():
                continue
            if issue and issue.lower() not in doc_text.lower():
                continue
                
            docs.append(doc_dict)
            
            if len(docs) >= limit:
                break
                
    logger.info(f"Retrieved {len(docs)} matching documents after filtering.")
    return docs

def evaluate_relevance(docs: List[Dict[str, Any]]) -> float:
    """Evaluates the relevance of the retrieved documents by returning the max similarity score."""
    if not docs:
        return 0.0
    max_similarity = max(doc["similarity"] for doc in docs)
    logger.info(f"Max similarity score of retrieved documents: {max_similarity:.4f}")
    return max_similarity
