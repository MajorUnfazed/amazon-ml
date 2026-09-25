"""TF-IDF candidate generation retrievers for LinkSure."""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Set, Optional

def sparse_top_k(matrix: sparse.csr_matrix, k: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extracts top-k column indices and scores for each row in a sparse CSR matrix.
    Vectorized and highly memory efficient.
    """
    row_indices = []
    row_scores = []

    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz == 0:
            row_indices.append(np.array([], dtype=int))
            row_scores.append(np.array([], dtype=float))
            continue

        data = row.data
        cols = row.indices

        if len(data) <= k:
            sort_order = np.argsort(-data)
            row_indices.append(cols[sort_order])
            row_scores.append(data[sort_order])
        else:
            top_k_part = np.argpartition(-data, k)[:k]
            sort_order = top_k_part[np.argsort(-data[top_k_part])]
            row_indices.append(cols[sort_order])
            row_scores.append(data[sort_order])

    return row_indices, row_scores

class TfidfRetriever:
    """
    Char or Word N-gram TF-IDF Retriever with chunked sparse top-k cosine retrieval.
    """
    def __init__(
        self,
        name: str,
        analyzer: str = "char_wb",
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True
    ):
        self.name = name
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            dtype=np.float32
        )
        self.partner_matrix: Optional[sparse.csr_matrix] = None
        self.partner_ids: List[str] = []

    def fit_index(self, corpus_texts: List[str], partner_ids: List[str]):
        """Fits vectorizer and builds index for candidate partner pool (S2 + S3)."""
        self.partner_ids = list(partner_ids)
        self.partner_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(
        self,
        query_texts: List[str],
        query_ids: List[str],
        top_k: int = 25,
        chunk_size: int = 1000
    ) -> List[Tuple[str, str, float, str]]:
        """
        Retrieves top-k candidates for each query ID in chunks.
        Returns list of (query_id, candidate_id, score, retriever_name).
        """
        if self.partner_matrix is None:
            raise RuntimeError("Index not fitted. Call fit_index first.")

        results = []
        n_queries = len(query_texts)

        for start_idx in range(0, n_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, n_queries)
            q_chunk_texts = query_texts[start_idx:end_idx]
            q_chunk_ids = query_ids[start_idx:end_idx]

            q_matrix = self.vectorizer.transform(q_chunk_texts)
            sim_matrix = q_matrix.dot(self.partner_matrix.T)

            row_indices, row_scores = sparse_top_k(sim_matrix, top_k)

            for q_id, top_indices, top_scores in zip(q_chunk_ids, row_indices, row_scores):
                for idx, score in zip(top_indices, top_scores):
                    if score > 0.01:
                        results.append((q_id, self.partner_ids[idx], float(score), self.name))

        return results
