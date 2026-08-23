from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / "AboutMLSC"
DB_PATH = ROOT_DIR / "knowledge_base.db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Below this cosine similarity, we treat the question as unanswerable
# from the knowledge base rather than risk a weakly-grounded answer.
RETRIEVAL_SCORE_THRESHOLD = 0.3
RETRIEVAL_TOP_K = 3

# Among the top-k, drop any chunk scoring more than this far below the best
# match. A fixed top-k always pulls in k documents even when only 1-2 are
# actually relevant, which hurts precision; this trims the clearly-worse
# stragglers while still keeping close runners-up for multi-doc questions.
RETRIEVAL_RELATIVE_MARGIN = 0.12
