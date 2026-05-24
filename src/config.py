import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

for d in [DATA_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
# minimaxai/minimax-m2.7
NVIDIA_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

USE_MOCK_LLM = not bool(NVIDIA_API_KEY)

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "meridian-bank-marketing")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")
MLFLOW_INFERENCE_EXPERIMENT_NAME = os.getenv("MLFLOW_INFERENCE_EXPERIMENT_NAME", "meridian-bank-inference")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.35"))
RAG_SAMPLE_SIZE = int(os.getenv("RAG_SAMPLE_SIZE", "5000"))

PROMOTION_PR_AUC_MIN_IMPROVEMENT = float(os.getenv("PROMOTION_PR_AUC_MIN_IMPROVEMENT", "0.03"))
PROMOTION_F1_MAX_DROP = float(os.getenv("PROMOTION_F1_MAX_DROP", "0.02"))
