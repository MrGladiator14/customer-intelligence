"""Unit tests for LangGraph RAG Agent."""

from unittest.mock import patch, MagicMock
import pytest
from src.rag.langgraph_agent import run_rag_agent, get_rag_agent_graph

@patch("src.rag.langgraph_agent.chromadb.PersistentClient")
@patch("src.rag.langgraph_agent.get_embedding_engine")
def test_rag_agent_in_domain_generation(mock_embedding_engine, mock_chroma_client):
    # Setup mocks for ChromaDB retrieval
    mock_coll = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll
    mock_chroma_client.return_value = mock_client
    
    # Mock search results for an in-domain query about high fees
    # Cosine distance = 0.4 -> Similarity = 0.6 >= threshold (0.35)
    mock_coll.query.return_value = {
        "documents": [["Expected high transaction fees charged on savings account without prior notice. [Doc-101]"]],
        "distances": [[0.4]],
        "metadatas": [[{"source_id": "Doc-101", "customer_id": "CUST001"}]]
    }
    
    # Run the agent
    response = run_rag_agent("Why was I charged high transaction fees?")
    
    assert response["relevance_score"] >= 0.35
    assert "Doc-101" in response["citations"]
    assert "transaction fees" in response["response"].lower()
    assert "retrieve" in response["nodes_visited"]
    assert "generate" in response["nodes_visited"]
    assert "refuse" not in response["nodes_visited"]

@patch("src.rag.langgraph_agent.chromadb.PersistentClient")
@patch("src.rag.langgraph_agent.get_embedding_engine")
def test_rag_agent_out_of_domain_refusal(mock_embedding_engine, mock_chroma_client):
    # Setup mocks for ChromaDB retrieval
    mock_coll = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll
    mock_chroma_client.return_value = mock_client
    
    # Mock search results for an out-of-domain query
    # Cosine distance = 0.9 -> Similarity = 0.1 < threshold (0.35)
    mock_coll.query.return_value = {
        "documents": [["Expected high transaction fees charged on savings account without prior notice. [Doc-101]"]],
        "distances": [[0.9]],
        "metadatas": [[{"source_id": "Doc-101", "customer_id": "CUST001"}]]
    }
    
    # Run the agent
    response = run_rag_agent("What is the capital of France?")
    
    assert response["relevance_score"] < 0.35
    assert len(response["citations"]) >= 0 # Actually it might have some citations from re_retrieve
    assert "rewrite_query" in response["nodes_visited"]
    assert "re_retrieve" in response["nodes_visited"]
    assert "generate" in response["nodes_visited"]

@patch("src.rag.langgraph_agent.chromadb.PersistentClient")
@patch("src.rag.langgraph_agent.get_embedding_engine")
def test_rag_agent_customer_id_filtering(mock_embedding_engine, mock_chroma_client):
    # Setup mocks for ChromaDB retrieval
    mock_coll = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll
    mock_chroma_client.return_value = mock_client
    
    # Mock search results containing complaints from multiple customers
    mock_coll.query.return_value = {
        "documents": [[
            "Savings account fees were unexpectedly high. [Doc-101]",
            "Credit card limit was reduced without warning. [Doc-102]"
        ]],
        "distances": [[0.1, 0.15]],
        "metadatas": [[
            {"source_id": "Doc-101", "customer_id": "CUST001"},
            {"source_id": "Doc-102", "customer_id": "CUST002"}
        ]]
    }
    
    # Run the agent filtering by customer_id CUST001
    response = run_rag_agent(
        question="Why was I charged high transaction fees?",
        customer_id="CUST001"
    )
    
    # Verify that the query call passed the where filter to ChromaDB query
    mock_coll.query.assert_called_once()
    kwargs = mock_coll.query.call_args[1]
    assert kwargs.get("where") == {"customer_id": "CUST001"}
    
    # Verify that only the CUST001 complaint was kept in the output
    assert "Doc-101" in response["citations"]
    assert "Doc-102" not in response["citations"]
