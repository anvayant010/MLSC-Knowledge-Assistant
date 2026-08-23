"""SQLite-backed chunk store with NumPy cosine-similarity search.

Chunks and their embeddings are persisted in SQLite (a real, inspectable
database file) so the knowledge base survives across runs. Similarity
search itself is done in-memory with NumPy rather than a dedicated vector
index (e.g. HNSW) -- at a few dozen chunks, brute-force cosine similarity
over a matrix is effectively instant, so an ANN index would add complexity
without any real benefit.
"""

import sqlite3
from dataclasses import dataclass

import numpy as np

from src.config import DB_PATH


@dataclass
class Chunk:
    id: int
    text: str
    source_file: str
    chunk_index: int


def get_connection(db_path=DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()


def clear_chunks(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks")
    conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    texts: list[str],
    source_files: list[str],
    chunk_indices: list[int],
    embeddings: np.ndarray,
) -> None:
    rows = [
        (source_files[i], chunk_indices[i], texts[i], embeddings[i].astype(np.float32).tobytes())
        for i in range(len(texts))
    ]
    conn.executemany(
        "INSERT INTO chunks (source_file, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def load_all(conn: sqlite3.Connection) -> tuple[list[Chunk], np.ndarray]:
    """Load every chunk and its embedding into memory for search."""
    cursor = conn.execute("SELECT id, source_file, chunk_index, text, embedding FROM chunks")
    chunks: list[Chunk] = []
    vectors: list[np.ndarray] = []
    for row_id, source_file, chunk_index, text, embedding_blob in cursor.fetchall():
        chunks.append(Chunk(id=row_id, text=text, source_file=source_file, chunk_index=chunk_index))
        vectors.append(np.frombuffer(embedding_blob, dtype=np.float32))

    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)

    return chunks, np.vstack(vectors)


def cosine_similarity_search(
    query_embedding: np.ndarray, embedding_matrix: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Return (row_index, score) pairs for the top_k most similar rows."""
    if embedding_matrix.shape[0] == 0:
        return []

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    matrix_norms = embedding_matrix / (
        np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
    )
    scores = matrix_norms @ query_norm

    top_k = min(top_k, len(scores))
    top_indices = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]
