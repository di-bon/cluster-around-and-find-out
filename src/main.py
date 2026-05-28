import json
import os
from dataclasses import dataclass, field

import hdbscan
import numpy as np
import ollama
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from agents.clustering_interview_agent import ClusteringInterviewAgent
from data.document_record import DocumentRecord
from data.document_registry import DocumentRegistry
from utils.clustering import build_cluster_index, merge_clusters, run_clustering
from utils.data_loader import build_dataset_summary, download_dataset, load_dataset
from utils.encoder import Encoder

# ── Step 0: Download the dataset ─────────────────────────────────────────────

load_dotenv()

# Dataset

DATASET_DIR = "datasets/stackoverflow"
TITLES_URL = "https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/title_StackOverflow.txt"
TITLES_PATH = os.path.join(DATASET_DIR, "title_StackOverflow.txt")

# Embeddings

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_MODEL = "nvidia/nv-embed-v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

download_dataset(dataset_dir=DATASET_DIR, url_path_pairs=[(TITLES_URL, TITLES_PATH)])
documents = load_dataset(TITLES_PATH, sample_size=500)  # use None for all 20k
dataset_summary = build_dataset_summary(documents)
agent = ClusteringInterviewAgent(dataset_summary=dataset_summary)

print("Agent: Hello! Here's a quick overview of your dataset:\n")
print(dataset_summary)
print("\nHow would you like these documents grouped?")

while not agent.ready_to_summarize:
    user_msg = input("\nYou: ")
    if user_msg.lower() == "exit":
        break
    reply = agent.chat(user_msg)
    print(f"\nAgent: {reply}")

# print("\n🎉 Step 1 Complete — moving to embedding and clustering.")

user_preference = agent.summarize_preferences()
print(f"\n📋 Embedding instruction:\n{user_preference}\n")

# user_preference = """Instruct: Embed 500 StackOverflow question titles to dynamically infer and cluster them by their primary underlying programming language, routing SQL/VBA to Database, defaulting to the most common language for multi-binding frameworks, and placing agnostic queries in Other/Misc, with each cluster named strictly by the exact inferred language without prefixes.
# Query:"""

# user_preference = """Instruct: Embed and cluster StackOverflow question titles into broad programming language groupings (PHP, C#, SQL/Database, CSS/HTML, dynamically generated languages, Multi-language, Language Agnostic, and Unknown) by mapping explicit language mentions, inferring primary languages from associated platforms or frameworks, routing database queries to SQL/Database, frontend markup to CSS/HTML, and assigning unmatched or multi-stack titles to their respective fallback buckets.
# Query:"""

# prepend the user preferences to each document
extended_documents = [f"{user_preference}{doc}" for doc in documents]

encoder = Encoder(NVIDIA_API_KEY, NVIDIA_BASE_URL, EMBEDDING_MODEL)

print("\n🔄 Generating vectors...")
document_embeddings = encoder.embed_documents(extended_documents)
print(f"   Embedding matrix shape: {document_embeddings.shape}")

config = agent.select_clustering_algorithm(
    n_documents=len(documents), user_preference=user_preference
)
labels = run_clustering(document_embeddings, config)
clusters = build_cluster_index(documents, labels)

print("\n📦 Cluster contents:")
for cluster_id, docs in sorted(clusters.items()):
    label_str = "🔇 Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"\n  {label_str} ({len(docs)} docs):")
    for doc in docs:
        print(f"    • {doc[:80]}")  # truncate long documents for readability

# config   = agent.select_clustering_algorithm(n_documents=len(documents), user_preference=user_preference)
# labels   = run_clustering(document_embeddings, config)

registry = DocumentRegistry(documents, document_embeddings, labels)

# agent.print_clustering_report(user_preference, registry)

# Example reverse lookup: given a raw embedding, find its document
# sample_vec = document_embeddings[0]
# nearest = registry.find_nearest(sample_vec, top_k=3)
# print("Nearest documents to document_embeddings[0]:")
# for r in nearest:
#     print(f"  [{r.id}] cluster={r.cluster}  {r.preview()}")

# ── Cluster editing operations ────────────────────────────────────────────────


def rename_cluster(registry: DocumentRegistry, cluster_id: int, name: str) -> None:
    """Attaches a human-chosen name to a cluster (stored in a side dict)."""
    if not hasattr(registry, "_names"):
        registry._names = {}
    registry._names[cluster_id] = name
    print(f"✅ Cluster {cluster_id} renamed to '{name}'.")


# ── Interactive refinement loop ───────────────────────────────────────────────


def run_refinement_loop(
    registry: DocumentRegistry,
    agent: ClusteringInterviewAgent,
) -> DocumentRegistry:
    """
    Lets the user iteratively refine clusters via natural language commands
    until they type 'done' or are satisfied.
    """
    print("\n" + "═" * 70)
    print("  CLUSTER REFINEMENT — type your changes in plain English.")
    print(
        "  Examples: 'merge 1 and 3', 'split cluster 2 base on the X', 'show current clustering assignment', 'done'"
    )
    print("═" * 70)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        try:
            command = agent.parse_user_command(user_input)
        except json.JSONDecodeError:
            print(
                "⚠️  Couldn't parse that. Try rephrasing (e.g. 'merge clusters 1 and 2')."
            )
            continue

        action = command.get("action")

        if action == "merge":
            user_request = command.get(
                "reason", f"merge clusters {command['clusters']}"
            )
            registry = agent.semantic_merge_clusters(
                user_preference, registry, command["clusters"], user_request, encoder
            )
            agent.print_clustering_report(user_preference, registry)

        elif action == "split":
            user_request = command.get("reason", f"split cluster {command['cluster']}")
            registry = agent.semantic_split_cluster(
                registry,
                command["cluster"],
                user_request,
                command["n_splits"],
                encoder,
                user_preference,
            )
            agent.print_clustering_report(user_preference, registry)

        elif action == "rename":
            rename_cluster(registry, command["cluster"], command["name"])

        elif action == "show":
            agent.print_clustering_report(user_preference, registry, None)

        elif action == "done":
            print("\n✅ Refinement complete.")
            print("--- Final clustering assignment ---")
            agent.print_clustering_report(user_preference, registry, None)
            break

        elif action == "unknown":
            print(f"⚠️  {command.get('reason', 'Unknown command.')} Try rephrasing.")

    return registry


# ── Wire it in after the initial clustering ───────────────────────────────────

registry = DocumentRegistry(documents, document_embeddings, labels)
agent.print_clustering_report(user_preference, registry)

registry = run_refinement_loop(registry, agent)

