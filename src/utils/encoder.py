import os

import numpy as np
import ollama
from openai import OpenAI


def make_encoder_client(
    model_name: str, nvidia_api_key: str, nvidia_base_url: str
) -> tuple:
    """Returns (client, backend) where backend is 'nvidia', 'openrouter', or 'ollama'."""

    # 1. Check if the model belongs to the Nvidia family
    if model_name.startswith("nvidia/") or model_name.startswith("NV-Embed"):
        client = OpenAI(api_key=nvidia_api_key, base_url=nvidia_base_url)
        return client, "nvidia"

    # 2. If not Nvidia, check for OpenRouter configuration
    if "OPENROUTER_API_URL" in os.environ:
        client = OpenAI(
            base_url=os.environ["OPENROUTER_API_URL"],
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
        return client, "openrouter"

    # 3. Fallback if neither applies
    return None, "ollama"


class Encoder:
    def __init__(
        self,
        embedding_model: str,
        nvidia_api_key: str = "",
        nvidia_base_url: str = "",
    ):
        self.embedding_model = embedding_model
        self.client, self.backend = make_encoder_client(
            embedding_model, nvidia_api_key, nvidia_base_url
        )

    def _embed_batch_nvidia(self, batch: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=batch,
            model=self.embedding_model,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"},
        )
        return [item.embedding for item in response.data]

    def _embed_batch_ollama(self, batch: list[str]) -> list[list[float]]:
        response = ollama.embed(
            model=self.embedding_model, input=batch, dimensions=4096
        )
        return response["embeddings"]

    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 50,
        cache_path: str = "embeddings_cache.npy",
        use_cache: bool = False,
    ) -> np.ndarray:
        """
        Embeds documents in batches using either NVIDIA or Ollama.
        Loads from cache if available and matches the document count.
        """
        if use_cache and os.path.exists(cache_path):
            cached_embeddings = np.load(cache_path)
            if len(cached_embeddings) == len(documents):
                print(
                    f"💾 Loaded {len(cached_embeddings)} embeddings from local cache ('{cache_path}')."
                )
                return cached_embeddings
            else:
                print(
                    "⚠️ Cache size mismatch with current documents. Regenerating from API..."
                )

        embed_batch = (
            self._embed_batch_nvidia
            if self.backend == "nvidia"
            else self._embed_batch_ollama
        )

        all_embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(
                f"  🔄 Batch {batch_num}/{total_batches} ({len(batch)} docs)… [{self.backend}]"
            )
            all_embeddings.extend(embed_batch(batch))

        print(f"✅ Embedded {len(all_embeddings)} documents.")

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        np.save(cache_path, embeddings_np)
        print(f"💾 Saved embeddings to cache ('{cache_path}').")

        return embeddings_np
