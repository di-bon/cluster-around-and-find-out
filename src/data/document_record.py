from dataclasses import dataclass

import numpy as np


@dataclass
class DocumentRecord:
    """Bundles a document with its embedding and assigned cluster."""

    id: int
    text: str
    embedding: np.ndarray
    cluster: int = -1

    def preview(self, max_chars: int = 120) -> str:
        return self.text[:max_chars] + ("…" if len(self.text) > max_chars else "")

