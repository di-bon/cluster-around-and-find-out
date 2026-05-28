import numpy as np
from openai import OpenAI
import os

class Encoder:
    def __init__(self, NVIDIA_API_KEY: str, NVIDIA_BASE_URL: str, EMBEDDING_MODEL: str):
        self.nvidia_client = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
        )
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
    
    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 50,
        cache_path: str = "embeddings_cache.npy"
    ) -> np.ndarray:
        """
        Embeds documents in batches using the NVIDIA API.
        Loads from cache if available and matches the document count.
        """
        # for testing ONLY! Don't use cached embeddings for experiment evaluation
        # they depend on the user preferences
        if os.path.exists(cache_path):
            cached_embeddings = np.load(cache_path)
            if len(cached_embeddings) == len(documents):
                print(f"💾 Loaded {len(cached_embeddings)} embeddings from local cache ('{cache_path}').")
                return cached_embeddings
            else:
                print("⚠️ Cache size mismatch with current documents. Regenerating from API...")

        # 2. Cache miss: Fetch from the API
        all_embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  🔄 Batch {batch_num}/{total_batches} ({len(batch)} docs)…")

            response = self.nvidia_client.embeddings.create(
                input=batch,
                model=self.EMBEDDING_MODEL,
                encoding_format="float",
                extra_body={
                    "input_type": "passage",   
                    "truncate": "END",         
                },
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        print(f"✅ Embedded {len(all_embeddings)} documents.")
        
        # 3. Convert to array, save to cache, and return
        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        np.save(cache_path, embeddings_np)
        print(f"💾 Saved embeddings to cache ('{cache_path}').")
        
        return embeddings_np