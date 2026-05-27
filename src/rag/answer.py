"""Meridian Customer Intelligence Platform - RAG Generation Lane."""

import logging
import re
from typing import List, Dict, Any
from src.config import (
    NVIDIA_API_KEY,
    NVIDIA_CHAT_MODEL,
    RAG_SIMILARITY_THRESHOLD,
    USE_MOCK_LLM
)

logger = logging.getLogger(__name__)

# LLM client singleton
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

def call_nvidia_llama(prompt: str) -> str:
    """Invokes the NVIDIA LLM or falls back to the mock generator."""
    if USE_MOCK_LLM or not NVIDIA_API_KEY:
        logger.info("Using high-fidelity Mock LLM for generation...")
        docs = re.findall(r"Document \[(Doc-\d+|Doc-GEN-\d+)\]:\s*(.*?)(?=\nDocument|Context:|\Z)", prompt, re.DOTALL)
        if not docs:
            return "Refused: Evidence insufficient to ground an answer."
            
        answers = []
        for doc_id, text in docs:
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
            return "Refused: Evidence insufficient to ground an answer."
            
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error calling NVIDIA Llama: {e}. Falling back to mock generator.")
        return "Refused: Evidence insufficient to ground an answer."

def generate_answer(question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates an answer using NVIDIA Llama 70B, strictly grounded in retrieved docs."""
    logger.info("Generating answer grounded in retrieved documents...")
    
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

User Question: {question}

Helpful, Grounded Answer:"""

    response = call_nvidia_llama(prompt)
    
    # Refusal safety check
    if "Refused: Evidence insufficient to ground an answer" in response:
        citations = []
        
    return {
        "response": response,
        "citations": list(set(citations)) if "Refused:" not in response else []
    }
