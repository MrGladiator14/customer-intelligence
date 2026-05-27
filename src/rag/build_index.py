import logging
import re
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_ROOT, DATA_DIR
from src.data_pipeline.ingest import ingest_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Persistence directory for ChromaDB
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── HuggingFace Embeddings Helper ────────────────────────────────────────
class HuggingFaceEmbeddings:
    """Uses sentence-transformers for embeddings."""
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
    
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

_embedding_engine = None

def get_embedding_engine():
    """Returns HuggingFace embeddings."""
    global _embedding_engine
    if _embedding_engine is None:
        logger.info("Initializing HuggingFace Embeddings (first time)...")
        _embedding_engine = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_engine

# ── Regex Helper to extract structural Doc ID ──────────────────────────────
def extract_doc_id(text: str, default_id: str) -> str:
    """Extracts a structural source ID like [Doc-101] from complaint text."""
    match = re.search(r"\[(Doc-\d+)\]", text)
    if match:
        return match.group(1)
    return default_id

def build_chroma_index():
    import chromadb
    logger.info("Building ChromaDB Vector Index...")
    
    # 1. Ingest train dataset containing complaints
    train_df = ingest_csv(PROJECT_ROOT / "data" / "train.csv")
    synthetic_df = ingest_csv(PROJECT_ROOT / "data" / "synthetic_train.csv")
    train_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    
    # Filter rows with valid complaints
    complaint_data = train_df[train_df["complaint"].notna() & (train_df["complaint"].str.strip() != "")]
    logger.info(f"Loaded {len(complaint_data)} complaints to index.")
    
    # 2. Initialize Chroma Client
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Re-create collection to ensure clean state
    collection_name = "customer_complaints"
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 3. Get Embedding Engine
    embedding_engine = get_embedding_engine()
    
    # 4. Prepare data for insertion
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    for idx, row in complaint_data.iterrows():
        text = str(row["complaint"])
        cust_id = str(row["customer_id"])
        
        # Extract unique structural source ID, e.g. [Doc-101]
        doc_id = extract_doc_id(text, f"Doc-GEN-{idx}")
        
        # Clean text of the document marker for vector storage if desired,
        # but leaving it makes it clear. We'll store the clean text or the whole text.
        documents.append(text)
        support_response = row.get("support_response")
        support_response_str = str(support_response) if pd.notna(support_response) else ""

        metadatas.append({
            "customer_id": cust_id,
            "source_id": doc_id,
            "age": int(row["age"]),
            "job": str(row["job"]),
            "education": str(row["education"]),
            "balance": float(row["balance"]),
            "support_response": support_response_str
        })
        ids.append(f"{cust_id}_{idx}")
    
    # Compute embeddings
    logger.info("Computing embeddings for complaints...")
    computed_embeddings = embedding_engine.embed_documents(documents)
        
    embeddings.extend(computed_embeddings)
    
    # 5. Insert into ChromaDB
    logger.info(f"Persisting {len(documents)} items in ChromaDB collection '{collection_name}'...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info("ChromaDB Index build complete.")

if __name__ == "__main__":
    build_chroma_index()
