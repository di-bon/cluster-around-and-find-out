import argparse
import os
import secrets
import time

import tiktoken
from dotenv import load_dotenv
from loguru import logger
from transformers.models.auto.tokenization_auto import AutoTokenizer

from agents.cluster_judge import ClusterJudge
from agents.clustering_interview_agent import ClusteringInterviewAgent
from data.document_registry import DocumentRegistry
from utils.clustering import build_cluster_index, run_clustering
from utils.compute_baseline import run_baseline
from utils.data_loader import build_dataset_summary, download_dataset, load_dataset
from utils.encoder import Encoder
from utils.log_step import ExperimentLogger


def run_experiment(interview_type: str):

    # --- CONFIG ---
    RUN_ID = secrets.token_hex(4)  # Generates a clean, short unique ID like "a3f2b1c0"
    SEED = secrets.randbits(32)

    DATASET_DIR = "datasets/stackoverflow"
    TITLES_URL = "https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/title_StackOverflow.txt"
    TITLES_PATH = os.path.join(DATASET_DIR, "title_StackOverflow.txt")
    SAMPLE_SIZE = None  # set it to None to use the full dataset
    N_CLUSTERS_SUMMARY = 10
    EXAMPLES_PER_CLUSTER_SUMMARY = 3
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6:35b")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:8b")
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

    # Tokenizer

    if "gpt" in LLM_MODEL:
        gpt_tokenizer = tiktoken.encoding_for_model(LLM_MODEL)
    else:
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")

    CONFIG = {
        "dataset_url": TITLES_URL,
        "dataset_path": TITLES_PATH,
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "N_CLUSTERS_SUMMARY": N_CLUSTERS_SUMMARY,
        "EXAMPLES_PER_CLUSTER_SUMMARY": EXAMPLES_PER_CLUSTER_SUMMARY,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger = ExperimentLogger(timestamp, RUN_ID, SEED, CONFIG, "logs")

    download_dataset(
        dataset_dir=DATASET_DIR, url_path_pairs=[(TITLES_URL, TITLES_PATH)]
    )
    documents = load_dataset(
        TITLES_PATH, sample_size=SAMPLE_SIZE
    )  # use None for the entire dataset

    dataset_summary = build_dataset_summary(
        documents,
        n_clusters=N_CLUSTERS_SUMMARY,
        examples_per_cluster=EXAMPLES_PER_CLUSTER_SUMMARY,
        seed=SEED,
    )
    agent = ClusteringInterviewAgent(
        dataset_summary=dataset_summary,
        model_name=LLM_MODEL,
        interview_type=interview_type,
    )

    print("Agent: Hello! Here's a quick overview of your dataset:\n")
    print(dataset_summary)
    print("\nHow would you like these documents grouped?")

    turns = 0
    total_tokens = 0
    total_user_tokens = 0

    while not agent.ready_to_summarize:
        user_msg = input("\nYou: ")
        logger.log_step(inputs={"user": user_msg})
        if user_msg.lower() == "exit":
            break
        reply = agent.chat(user_msg)
        logger.log_step(inputs={}, outputs={"agent": reply})
        print(f"\nAgent: {reply}")

        turns += 1
        if "gpt" in LLM_MODEL:
            # tiktoken encode
            user_tokens = len(gpt_tokenizer.encode(user_msg))
            reply_tokens = (
                len(gpt_tokenizer.encode(reply))
                if "[READY_TO_SUMMARIZE]" not in reply
                else 0
            )
        else:
            # Hugging Face encode (disabling automatic special tokens)
            user_tokens = len(qwen_tokenizer.encode(user_msg, add_special_tokens=False))
            reply_tokens = (
                len(qwen_tokenizer.encode(reply, add_special_tokens=False))
                if "[READY_TO_SUMMARIZE]" not in reply
                else 0
            )

        total_user_tokens += user_tokens
        total_tokens += user_tokens + reply_tokens

    instruction = agent.get_embedding_instruction()
    print(f"\n📋 Embedding instruction:\n{instruction}\n")
    logger.log_step(inputs={}, outputs={"instruction": instruction})

    # prepend the user preferences to each document
    extended_documents = [f"{instruction}{doc}" for doc in documents]

    user_preference = agent.get_user_preference()
    print(f"\n 🦆 User preference: \n{user_preference}\n")
    logger.log_step(inputs={}, outputs={"user_preference": user_preference})

    encoder = Encoder(EMBEDDING_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL)

    print("\n🔄 Generating vectors...")
    document_embeddings = encoder.embed_documents(extended_documents, use_cache=False)
    print(f"   Embedding matrix shape: {document_embeddings.shape}")

    has_rows = document_embeddings.shape[0] > 0
    vector_preview = document_embeddings[0][:5].tolist() if has_rows else []
    logger.log_step(
        inputs={},
        outputs={
            "document_embeddings": {
                "matrix_shape": list(document_embeddings.shape),
                "data_type": str(document_embeddings.dtype),
                "vector_preview": vector_preview,
            }
        },
    )

    config = agent.select_clustering_algorithm(
        n_documents=len(documents), user_preference=user_preference
    )
    logger.log_step(
        inputs={"user_preference": user_preference}, outputs={"config": config}
    )

    labels = run_clustering(document_embeddings, config)
    labels_preview = labels[:5].tolist()
    logger.log_step(
        inputs={
            "document_embeddings": {
                "matrix_shape": list(document_embeddings.shape),
                "data_type": str(document_embeddings.dtype),
                "vector_preview": vector_preview,
            },
            "config": config,
        },
        outputs={"labels_preview": labels_preview},
    )

    clusters = build_cluster_index(documents, labels)
    documents_preview = documents[:5]
    logger.log_step(
        inputs={
            "documents_preview": documents_preview,
            "labels_preview": labels_preview,
        },
        outputs={"clusters": clusters},
    )

    print("\n📦 Cluster contents:")
    for cluster_id, docs in sorted(clusters.items()):
        label_str = "🔇 Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n  {label_str} ({len(docs)} docs):")
        for doc in docs:
            print(f"    • {doc[:80]}")  # truncate long documents for readability

    # registry = DocumentRegistry(documents, document_embeddings, labels)
    # agent.print_clustering_report(user_preference, registry)

    judge = ClusterJudge(model_name=LLM_MODEL)
    judgement = judge.evaluate(user_preference=user_preference, clusters=clusters)

    print("\n" + "=" * 70)
    print("\nMETRICS\n")
    print(f"\nTurns: {turns}")
    print(f"\nTotal tokens: {total_tokens}")
    print(f"\nTotal user tokens: {total_user_tokens}")
    print(f"\nJudgement:")
    print(f"\n  Coherence: {judgement.coherence}")
    print(f"\n  Alignment: {judgement.alignment}")
    print(f"\n  Separation: {judgement.separation}")
    print(f"\n  Mean score: {judgement.mean_score}")
    print(f"\n  Reasoning: {judgement.reasoning}")
    print("\n" + "=" * 70)

    logger.log_step(
        outputs={
            "metrics": {
                "turns": turns,
                "total_tokens": total_tokens,
                "total_user_tokens": total_user_tokens,
                "judgement": {
                    "coherence": judgement.coherence,
                    "alignment": judgement.alignment,
                    "separation": judgement.separation,
                    "reasoning": judgement.reasoning,
                    "mean_score": judgement.mean_score,
                },
            }
        }
    )

    run_baseline(
        documents=documents,
        encoder=encoder,
        agent=agent,
        judge=judge,
        logger=logger,
        user_preference=user_preference,
    )


def main():
    parser = argparse.ArgumentParser(description="Interview process automation script.")

    parser.add_argument(
        "--interview_type",
        type=str,
        required=False,
        default="free",
        choices=["yes_no", "multiple_choice", "open_questions", "free"],
        help="The type of interview to conduct (default: %(default)s).",
    )

    args = parser.parse_args()

    load_dotenv()

    run_experiment(args.interview_type)


if __name__ == "__main__":
    main()
