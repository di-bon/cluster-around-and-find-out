from data.document_record import DocumentRecord
import numpy as np

class DocumentRegistry:
    """
    Single source of truth that keeps documents, embeddings, and cluster
    labels in sync. Supports retrieval in both directions:
      - by document id  → get the embedding / cluster
      - by embedding    → recover the original document (nearest neighbour)
    """

    def __init__(self, documents: list[str], embeddings: np.ndarray, labels: np.ndarray):
        assert len(documents) == len(embeddings) == len(labels), \
            "documents, embeddings and labels must have the same length"

        self._records: list[DocumentRecord] = [
            DocumentRecord(id=i, text=doc, embedding=emb, cluster=int(lbl))
            for i, (doc, emb, lbl) in enumerate(zip(documents, embeddings, labels))
        ]
        # Pre-build a normalised matrix for fast cosine nearest-neighbour lookup
        self._matrix = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # ── Forward lookup (id / cluster) ────────────────────────────────────────

    def get_by_id(self, doc_id: int) -> DocumentRecord:
        return self._records[doc_id]

    def get_by_cluster(self, cluster_id: int) -> list[DocumentRecord]:
        return [r for r in self._records if r.cluster == cluster_id]

    @property
    def cluster_ids(self) -> list[int]:
        return sorted(set(r.cluster for r in self._records))

    # ── Reverse lookup (embedding → document) ────────────────────────────────

    def find_nearest(self, query_embedding: np.ndarray, top_k: int = 1) -> list[DocumentRecord]:
        """
        Given an arbitrary embedding vector, return the top-k closest
        documents by cosine similarity. Useful when you only have an
        embedding and need to recover the source text.
        """
        normed_query = query_embedding / np.linalg.norm(query_embedding)
        scores = self._matrix @ normed_query          # cosine similarities
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._records[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._records)