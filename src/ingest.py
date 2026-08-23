"""Load the MLSC knowledge base, embed it, and persist it to SQLite.

Each source file is embedded as a single whole-document chunk rather than
split into paragraphs. The files are small (under ~1.5KB each), easily within
any LLM's context window, and paragraph-level splitting was tested and found
to actively hurt retrieval: it separated an introductory sentence from the
list it introduced, and the chunk containing the actual answer (a list of
domain names) ranked 20th out of 47 for a direct question asking for exactly
that list, because the intro sentence had higher literal word-overlap with
the query. Keeping each document whole avoids this failure mode entirely.

Run from the project root with the venv active:
    python -m src.ingest
"""

from sentence_transformers import SentenceTransformer

from src.config import DB_PATH, EMBEDDING_MODEL_NAME, KNOWLEDGE_BASE_DIR
from src.store import clear_chunks, get_connection, init_db, insert_chunks


def load_documents() -> dict[str, str]:
    documents = {}
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")
    return documents


def main() -> None:
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No .txt files found in {KNOWLEDGE_BASE_DIR}")

    texts: list[str] = []
    source_files: list[str] = []
    chunk_indices: list[int] = []

    for filename, content in documents.items():
        texts.append(content.strip())
        source_files.append(filename)
        chunk_indices.append(0)
        print(f"{filename}: 1 chunk (whole document)")

    print(f"\nEmbedding {len(texts)} chunks with '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    conn = get_connection()
    init_db(conn)
    clear_chunks(conn)
    insert_chunks(conn, texts, source_files, chunk_indices, embeddings)
    conn.close()

    print(f"Stored {len(texts)} chunks in {DB_PATH}")


if __name__ == "__main__":
    main()
