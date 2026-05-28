import ollama
from utils.prompt_loader import get_prompt
import json
from data.document_record import DocumentRecord
from data.document_registry import DocumentRegistry
from utils.encoder import Encoder
from utils.clustering import run_clustering
import numpy as np

class ClusteringInterviewAgent:
    def __init__(self, dataset_summary: str, model_name: str = "qwen3.6:35b"):
        self.model_name = model_name
        self.messages   = [{"role": "system", "content": get_prompt("interview.md").format(dataset_summary=dataset_summary)}]
        self.ready_to_summarize = False
        self._label_cache: dict[frozenset, dict] = {}

    def _cluster_cache_key(self, records: list[DocumentRecord]) -> frozenset:
        return frozenset(r.id for r in records)
    
    def invalidate_label_cache(self, records: list[DocumentRecord]) -> None:
        key = self._cluster_cache_key(records)
        self._label_cache.pop(key, None)

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response       = ollama.chat(model=self.model_name, messages=self.messages)
        assistant_reply = response["message"]["content"]
        self.messages.append({"role": "assistant", "content": assistant_reply})

        if "[READY_TO_SUMMARIZE]" in assistant_reply:
            self.ready_to_summarize = True
        return assistant_reply
    
    def summarize_preferences(self) -> str:
        """
        Reuses the agent's conversation history and asks the LLM to distill
        everything it learned into a single NV-Embed-v2 instruction string.
        """
        # We append the summarizer prompt to a copy of the history 
        # so we don't permanently contaminate self.messages with the meta-prompt.
        summarizer_messages = self.messages + [
            {"role": "user", "content": get_prompt("summarize.md")}
        ]

        response = ollama.chat(model=self.model_name, messages=summarizer_messages)
        summary = response['message']['content'].strip()
        return summary
    
    def select_clustering_algorithm(self, n_documents: int, user_preference: str) -> dict:
        """
        Asks the LLM to pick an algorithm + params given the conversation history
        and basic dataset statistics.
        """
        user_turn = (
            f"User preference summary:\n{user_preference}\n\n"
            f"Dataset statistics:\n"
            f"- Number of documents: {n_documents}\n"
            f"- Embedding model: NV-Embed-v1 (4096-dim, cosine similarity)\n\n"
            f"{get_prompt("select_algorithm.md")}"
        )

        selector_messages = self.messages + [{"role": "user", "content": user_turn}]
        response = ollama.chat(model=self.model_name, messages=selector_messages)
        raw = response['message']['content'].strip()

        # Strip accidental markdown fences the model might add despite instructions
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        config = json.loads(raw)
        print(f"\n🤖 Algorithm chosen: {config['algorithm']}")
        print(f"   Rationale: {config['rationale']}")
        print(f"   Params:    {config['params']}")
        return config
    
    def label_cluster(
        self,
        user_preference: str,
        records: list[DocumentRecord],
        sample_size: int = 5,
    ) -> dict:
        """Asks the LLM to name a cluster based on a sample of its documents."""
        key = self._cluster_cache_key(records)

        if key in self._label_cache:
            print(f"   💾 Cache hit — reusing label '{self._label_cache[key]['label']}'")
            return self._label_cache[key]

        sample_texts = "\n".join(
            f"- {r.preview()}" for r in records[:sample_size]
        )
        user_turn = (
            f"User preference summary:\n{user_preference}\n\n"
            f"Documents in this cluster (sample of {min(sample_size, len(records))}):\n"
            f"{sample_texts}\n\n"
            f"{get_prompt('cluster_labeller.md')}"
        )
        messages = self.messages + [{"role": "user", "content": user_turn}]
        response = ollama.chat(model=self.model_name, messages=messages)
        raw = response["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        self._label_cache[key] = result
        return result
    
    def print_clustering_report(
        self,
        user_preference: str,
        registry: DocumentRegistry,
        docs_per_cluster: int | None = 5,
    ) -> None:
        """
        Prints a structured report for a human evaluator:
        one section per cluster, with an LLM-generated label and document previews.
        """
        total = len(registry)
        cluster_ids = registry.cluster_ids
        print("\n" + "═" * 70)
        print(f"  CLUSTERING REPORT — {total} documents, {len(cluster_ids)} cluster(s)")
        print("═" * 70)
        for cid in cluster_ids:
            records = registry.get_by_cluster(cid)
            pct = len(records) / total * 100
            if cid == -1:
                # HDBSCAN noise bucket — skip LLM labelling
                print(f"\n  🔇  NOISE  ({len(records)} docs, {pct:.1f}%)")
                print("  " + "─" * 66)
                for r in records[:docs_per_cluster]:
                    print(f"    [{r.id:>4}] {r.preview()}")
                if docs_per_cluster is not None and len(records) > docs_per_cluster:
                    print(f"           … and {len(records) - docs_per_cluster} more")
                continue
            # Ask LLM to name the cluster
            meta = self.label_cluster(user_preference, records)
            print(f"\n  📂  Cluster {cid}  ·  {meta['label']}  ({len(records)} docs, {pct:.1f}%)")
            print(f"      {meta['description']}")
            print("  " + "─" * 66)
            for r in records[:docs_per_cluster]:
                print(f"    [{r.id:>4}] {r.preview()}")
            if docs_per_cluster is not None and len(records) > docs_per_cluster:
                print(f"           … and {len(records) - docs_per_cluster} more")
        print("\n" + "═" * 70)
        print("  Review complete. Use registry.get_by_id(id) to inspect any document.")
        print("═" * 70 + "\n")

    def refine_instruction_for_cluster(
        self,
        user_preference: str,
        records: list[DocumentRecord],
        user_request: str,
        n_examples: int = 10,
    ) -> str:
        """Asks the LLM to write a targeted embedding instruction for a cluster refinement."""
        sample = "\n".join(f"  - {r.preview()}" for r in records[:n_examples])
        user_turn = (
            f"Original embedding instruction:\n{user_preference}\n\n"
            f"User's refinement request: {user_request}\n\n"
            f"Sample documents from this cluster:\n{sample}\n\n"
            f"{get_prompt("instruction_refinement.md")}"
        )
        messages = self.messages + [{"role": "user", "content": user_turn}]
        response = ollama.chat(model=self.model_name, messages=messages)
        instruction = response["message"]["content"].strip()
        print(f"\n📐 New embedding instruction:\n   {instruction}")
        return instruction
    
    def semantic_split_cluster(
        self,
        registry: DocumentRegistry,
        cluster_id: int,
        user_request: str,
        n_splits: int,
        encoder: Encoder,
        user_preference: str,
    ) -> DocumentRegistry:
        """
        Splits a cluster by:
        1. Asking the LLM to write a targeted embedding instruction
        2. Re-embedding only the cluster's documents with that instruction
        3. Clustering the new embeddings into n_splits parts
        4. Updating the registry records in place
        """
        records = registry.get_by_cluster(cluster_id)
        if len(records) < n_splits:
            print(f"⚠️  Only {len(records)} docs in cluster {cluster_id}, can't split into {n_splits}.")
            return registry

        new_instruction = self.refine_instruction_for_cluster(user_preference, records, user_request)

        # Step 2 — re-embed with the new instruction
        documents = [f"{new_instruction}{r.text}" for r in records]
        new_embeddings = encoder.embed_documents(documents)

        # Step 3 — cluster the new embeddings
        sub_config = {"algorithm": "kmeans", "params": {"n_clusters": n_splits, "random_state": 42}}
        sub_labels = run_clustering(new_embeddings, sub_config)

        # Step 4 — update registry: new embeddings + new cluster ids
        existing_ids = set(r.cluster for r in registry._records)
        next_id = max(existing_ids) + 1
        id_map = {i: next_id + i for i in range(n_splits)}
        id_map[0] = cluster_id  # reuse original id for first sub-cluster

        for record, sub_label, new_emb in zip(records, sub_labels, new_embeddings):
            record.embedding = new_emb          # update the embedding in place
            record.cluster   = id_map[int(sub_label)]

        # Rebuild the registry's internal matrix to keep find_nearest consistent
        all_embeddings = np.array([r.embedding for r in registry._records])
        registry._matrix = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

        self.invalidate_label_cache(records)

        print(f"✅ Semantically split cluster {cluster_id} → {list(id_map.values())}.")
        return registry
    
    def semantic_merge_clusters(
        self,
        user_preference: str,
        registry: DocumentRegistry,
        cluster_ids: list[int],
        user_request: str,
        encoder: Encoder,
    ) -> DocumentRegistry:
        """
        Merges clusters and re-embeds the combined set with an instruction
        that reflects why the user thinks they belong together.
        """
        records = []
        for cid in cluster_ids:
            records.extend(registry.get_by_cluster(cid))

        # Step 1 — get an instruction that reflects the user's merging rationale
        new_instruction = self.refine_instruction_for_cluster(user_preference, records, user_request)

        # Step 2 — re-embed the combined set
        documents = [f"{new_instruction}{r.text}" for r in records]
        new_embeddings = encoder.embed_documents(documents)

        # Step 3 — reassign all to the lowest cluster id, update embeddings
        target = min(cluster_ids)
        for record, new_emb in zip(records, new_embeddings):
            record.embedding = new_emb
            record.cluster   = target

        # Rebuild the registry's internal matrix
        all_embeddings = np.array([r.embedding for r in registry._records])
        registry._matrix = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

        self.invalidate_label_cache(records)

        print(f"✅ Semantically merged clusters {cluster_ids} → cluster {target}.")
        return registry
    
    def parse_user_command(self, user_input: str) -> dict:
        messages = [
            {"role": "system", "content": get_prompt("command_parser.md")},
            {"role": "user",   "content": user_input},
        ]
        response = ollama.chat(model=self.model_name, messages=messages)
        raw = response["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)