"""
run_batch.py — Run the clustering pipeline in parallel,
executing across all pairs of (persona, goal).

Usage:
    python run_batch.py                                     # Runs ALL (persona, goal) combinations
    python run_batch.py --workers 6                         # Runs all combinations with 6 parallel workers
    python run_batch.py --runs 10                           # Caps the run at the first 10 combinations
    python run_batch.py --interview_type multiple_choice     # Specific interview type
"""

import argparse
import itertools
import os
import pathlib
import secrets
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import jsonlines
from dotenv import load_dotenv
from loguru import logger as root_logger

load_dotenv()

# ---------------------------------------------------------------------------
# Single-run function (must be importable at top level for multiprocessing)
# ---------------------------------------------------------------------------


def run_once(
    run_index: int, interview_type: str, persona_index: int, goal_index: int
) -> dict:
    """
    Execute one full pipeline run with a SimulatedUser driving the chat loop.
    Returns a summary dict with run metadata and metrics.
    """
    import secrets
    import time

    import tiktoken
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    from agents.cluster_judge import ClusterJudge
    from agents.clustering_interview_agent import ClusteringInterviewAgent
    from data.document_registry import DocumentRegistry
    from prompts.simulated_user.config import GOALS, PERSONAS
    from utils.clustering import build_cluster_index, run_clustering
    from utils.compute_baseline import run_baseline
    from utils.data_loader import build_dataset_summary, download_dataset, load_dataset
    from utils.encoder import Encoder
    from utils.log_step import ExperimentLogger
    from utils.simulated_user import SimulatedUser

    # ---- CONFIG --------------------------------------------------------------
    RUN_ID = secrets.token_hex(4)
    SEED = secrets.randbits(32)

    DATASET_DIR = "datasets/stackoverflow"
    TITLES_URL = "https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/title_StackOverflow.txt"
    TITLES_PATH = os.path.join(DATASET_DIR, "title_StackOverflow.txt")
    SAMPLE_SIZE = None
    N_CLUSTERS_SUMMARY = 10
    EXAMPLES_PER_CLUSTER_SUMMARY = 3
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6:35b")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:8b")
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

    CONFIG = {
        "dataset_url": TITLES_URL,
        "dataset_path": TITLES_PATH,
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "N_CLUSTERS_SUMMARY": N_CLUSTERS_SUMMARY,
        "EXAMPLES_PER_CLUSTER_SUMMARY": EXAMPLES_PER_CLUSTER_SUMMARY,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "INTERVIEW_TYPE": interview_type,
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_logger = ExperimentLogger(
        timestamp, RUN_ID, SEED, CONFIG, f"logs/{interview_type}"
    )

    # ---- Tokenizer -----------------------------------------------------------
    if "gpt" in LLM_MODEL:
        tokenizer = tiktoken.encoding_for_model(LLM_MODEL)

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

    # ---- Data ----------------------------------------------------------------
    download_dataset(
        dataset_dir=DATASET_DIR, url_path_pairs=[(TITLES_URL, TITLES_PATH)]
    )
    documents = load_dataset(TITLES_PATH, sample_size=SAMPLE_SIZE)

    dataset_summary = build_dataset_summary(
        documents,
        n_clusters=N_CLUSTERS_SUMMARY,
        examples_per_cluster=EXAMPLES_PER_CLUSTER_SUMMARY,
        seed=SEED,
    )

    # ---- Agents --------------------------------------------------------------
    agent = ClusteringInterviewAgent(
        dataset_summary=dataset_summary,
        model_name=LLM_MODEL,
        interview_type=interview_type,
    )

    # Safely index goals and personas using modulo just in case, though combinations should be precise
    goal = GOALS[goal_index % len(GOALS)]
    persona = PERSONAS[persona_index % len(PERSONAS)]

    user = SimulatedUser(model_name=LLM_MODEL, seed=SEED, goal=goal, persona=persona)

    print(
        f"[run {run_index} | {RUN_ID}] SimulatedUser (Persona Index: {persona_index}, Goal Index: {goal_index}): {user}"
    )
    exp_logger.log_step(
        outputs={
            "simulated_user": {
                "persona": user.persona,
                "persona_index": persona_index,
                "goal": user.goal,
                "goal_index": goal_index,
            }
        }
    )

    opening = (
        f"Hello! Here's a quick overview of your dataset:\n\n{dataset_summary}"
        "\n\nHow would you like these documents grouped?"
    )

    # ---- Conversation loop ---------------------------------------------------
    turns = 0
    total_tokens = 0
    total_user_tokens = 0
    agent_message = opening
    MAX_TURNS = 20  # safety cap

    while not agent.ready_to_summarize:
        user_reply = user.respond(agent_message)
        print(f"[run {run_index} | {RUN_ID}] User turn {turns + 1}: {user_reply}")
        exp_logger.log_step(inputs={"user": user_reply})

        if user_reply.lower() in ("exit", "quit"):
            break

        agent_message = agent.chat(user_reply)
        exp_logger.log_step(inputs={}, outputs={"agent": agent_message})
        print(f"[run {run_index} | {RUN_ID}] Agent turn {turns + 1}: {agent_message}")

        turns += 1
        total_tokens += count_tokens(user_reply)
        total_user_tokens += count_tokens(user_reply)
        if "[READY_TO_SUMMARIZE]" not in agent_message:
            total_tokens += count_tokens(agent_message)

        if turns >= MAX_TURNS:
            break

    # ---- Preference & instruction --------------------------------------------
    instruction = agent.get_embedding_instruction()
    exp_logger.log_step(inputs={}, outputs={"instruction": instruction})
    print(f"[run {run_index} | {RUN_ID}] Embedding instruction: {instruction}")

    user_preference = agent.get_user_preference()
    exp_logger.log_step(inputs={}, outputs={"user_preference": user_preference})

    # ---- Embeddings ----------------------------------------------------------
    extended_documents = [f"{instruction}{doc}" for doc in documents]
    encoder = Encoder(EMBEDDING_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL)
    document_embeddings = encoder.embed_documents(extended_documents, use_cache=False)

    has_rows = document_embeddings.shape[0] > 0
    vector_preview = document_embeddings[0][:5].tolist() if has_rows else []
    exp_logger.log_step(
        inputs={},
        outputs={
            "document_embeddings": {
                "matrix_shape": list(document_embeddings.shape),
                "data_type": str(document_embeddings.dtype),
                "vector_preview": vector_preview,
            }
        },
    )

    # ---- Clustering ----------------------------------------------------------
    config = agent.select_clustering_algorithm(
        n_documents=len(documents), user_preference=user_preference
    )
    exp_logger.log_step(
        inputs={"user_preference": user_preference}, outputs={"config": config}
    )

    labels = run_clustering(document_embeddings, config)
    labels_preview = labels[:5].tolist()
    exp_logger.log_step(
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
    exp_logger.log_step(
        inputs={
            "documents_preview": documents_preview,
            "labels_preview": labels_preview,
        },
        outputs={"clusters": clusters},
    )

    # ---- Judge ---------------------------------------------------------------
    judge = ClusterJudge(model_name=LLM_MODEL)
    judgement = judge.evaluate(user_preference=user_preference, clusters=clusters)

    # ---- Metrics -------------------------------------------------------------
    n_clusters = len([k for k in clusters if k != -1])
    n_noise = len(clusters.get(-1, []))

    metrics = {
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

    exp_logger.log_step(
        inputs={
            "condition": f"interview_{interview_type}",
            "n_documents": len(documents),
            "instruction": instruction,
            "user_preference": user_preference,
            "clusters": exp_logger.format_clusters_for_logging(clusters),
        },
        outputs={
            "metrics": metrics,
        },
    )

    print(
        f"\n[run {run_index} | {RUN_ID}] turns={turns} tokens={total_tokens} "
        f"mean_score={judgement.mean_score:.2f}"
    )

    run_baseline(
        documents=documents,
        encoder=encoder,
        agent=agent,
        judge=judge,
        logger=exp_logger,
        user_preference=user_preference,
    )

    return {
        "run_index": run_index,
        "run_id": RUN_ID,
        "seed": SEED,
        "interview_type": interview_type,
        "persona_index": persona_index,
        "goal_index": goal_index,
        "user_goal": user.goal,
        "user_persona": user.persona,
        "user_preference": user_preference,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "clustering_config": config,
        "metrics": metrics,
        "status": "ok",
    }


def run_once_safe(
    run_index: int, interview_type: str, persona_index: int, goal_index: int
) -> dict:
    """Wrapper that catches exceptions so one failed run doesn't abort the batch."""
    try:
        return run_once(run_index, interview_type, persona_index, goal_index)
    except Exception as exc:  # noqa: BLE001
        return {
            "run_index": run_index,
            "persona_index": persona_index,
            "goal_index": goal_index,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run the clustering pipeline across all persona and goal combinations."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Cap maximum number of runs (defaults to all combinations)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel workers (default: 1)"
    )
    parser.add_argument(
        "--interview_type",
        type=str,
        default="free",
        choices=["yes_no", "multiple_choice", "open_questions", "free"],
        help="Interview type passed to ClusteringInterviewAgent (default: free)",
    )
    args = parser.parse_args()

    # Import configuration directly here to setup the Cartesian combinations
    from prompts.simulated_user.config import GOALS, PERSONAS

    # Generate all pairs of (persona_index, goal_index)
    all_combinations = list(itertools.product(range(len(PERSONAS)), range(len(GOALS))))
    total_combinations = len(all_combinations)

    # Determine how many tasks we actually execute
    if args.runs is not None:
        tasks_to_run = all_combinations[: args.runs]
        total_runs = len(tasks_to_run)
    else:
        tasks_to_run = all_combinations
        total_runs = total_combinations

    root_logger.info(
        f"Starting batch: {total_runs} runs requested (out of {total_combinations} total persona-goal pairs), "
        f"{args.workers} workers, interview_type={args.interview_type}"
    )
    t0 = time.time()

    failed = 0
    out_path = pathlib.Path("logs") / f"batch_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            # Submit each distinct combination to the pool
            futures = {
                pool.submit(run_once_safe, idx, args.interview_type, p_idx, g_idx): idx
                for idx, (p_idx, g_idx) in enumerate(tasks_to_run)
            }

            for future in as_completed(futures):
                result = future.result()
                writer.write(result)  # flushed immediately — safe even if batch crashes

                if result["status"] == "error":
                    failed += 1
                    root_logger.error(
                        f"Run {result['run_index']} (Persona: {result['persona_index']}, Goal: {result['goal_index']}) FAILED: {result['error']}"
                    )
                else:
                    m = result["metrics"]
                    root_logger.info(
                        f"Run {result['run_index']} ({result['run_id']}) done — "
                        f"Persona={result['persona_index']} "
                        f"Goal={result['goal_index']} — "
                        f"clusters={result['n_clusters']} "
                        f"turns={m['turns']} "
                        f"tokens={m['total_tokens']} "
                        f"mean_score={m['judgement']['mean_score']:.2f} "
                        f"goal={result['user_goal'][:50]}"
                    )

    elapsed = time.time() - t0
    root_logger.info(
        f"Batch complete in {elapsed:.1f}s — "
        f"{total_runs - failed}/{total_runs} succeeded, {failed} failed. "
        f"Results written to {out_path}"
    )


if __name__ == "__main__":
    main()
