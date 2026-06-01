from agents.cluster_judge import ClusterJudge
from agents.clustering_interview_agent import ClusteringInterviewAgent
from utils.clustering import build_cluster_index, run_clustering
from utils.encoder import Encoder
from utils.log_step import ExperimentLogger


def run_baseline(
    documents: list[str],
    encoder: Encoder,
    agent: ClusteringInterviewAgent,
    judge: ClusterJudge,
    logger: ExperimentLogger,
    user_preference: str,  # collected from the conversational run on the same documents
) -> dict:
    # Step 1: embed with no instruction
    embeddings = encoder.embed_documents(documents)

    # Step 2: pick algorithm with no user preference — use a neutral placeholder
    config = agent.select_clustering_algorithm(
        n_documents=len(documents),
        user_preference="No user preference available. Choose a sensible default.",
    )

    # Step 3: cluster
    labels = run_clustering(embeddings, config)

    # Step 4: build cluster dict for the judge
    clusters = build_cluster_index(documents, labels)

    # Step 5: judge using the user preferences from the conversational run
    judgement = judge.evaluate(
        user_preference=user_preference,
        clusters=clusters,
    )

    # Step 6: log
    logger.log_step(
        inputs={
            "condition": "baseline",
            "n_documents": len(documents),
            "instruction": None,
            "user_preference": user_preference,
        },
        outputs={
            "metrics": {
                "turns": 0,
                "total_tokens": 0,
                "total_user_tokens": 0,
                "judgement": {
                    "coherence": judgement.coherence,
                    "alignment": judgement.alignment,
                    "separation": judgement.separation,
                    "reasoning": judgement.reasoning,
                    "mean_score": judgement.mean_score,
                },
            }
        },
    )

    return {
        "labels": labels,
        "config": config,
        "judgement": judgement,
    }
