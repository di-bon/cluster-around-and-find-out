import hashlib
import json

from loguru import logger


class ExperimentLogger:
    def __init__(
        self,
        timestamp: str,
        run_id: str,
        seed: int,
        config: dict,
        log_dir: str = "logs",
    ):
        """
        Initializes an independent structured log file bound to this object instance.
        """
        self.run_id = run_id
        self.seed = seed

        # 1. Process and store configuration metadata
        self.llm_model = config.get("LLM_MODEL", "unknown_LLM_model")
        self.embedding_model = config.get("EMBEDDING_MODEL", "unknown_embedding_model")

        # Compute a stable config hash
        config_string = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.sha256(config_string.encode("utf-8")).hexdigest()[:8]

        # 2. Strip default handlers so the user's terminal stays clean
        logger.remove()

        log_filepath = f"{log_dir}/run_{timestamp}_{self.run_id}_{self.seed}.jsonl"

        def jsonl_sink(message):
            record = message.record
            record_data = {**record["extra"]}  # copy so we don't mutate the live record
            record_data["timestamp"] = record["time"].strftime("%Y%m%d-%H%M%S")
            with open(log_filepath, "a") as f:
                f.write(json.dumps(record_data) + "\n")

        self._handler_id = logger.add(jsonl_sink, level="INFO")

    def log_step(self, inputs: dict = {}, outputs: dict = {}, errors: str = ""):
        """
        Logs a structured record using the parameters passed during initialization.
        """
        logger.bind(
            run_id=self.run_id,
            seed=self.seed,
            config_hash=self.config_hash,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            inputs=inputs,
            outputs=outputs,
            errors=errors,
        ).info("")

    def log_metrics(self, turns: int, total_tokens: int):
        """
        Logs the metrics for the current run
        """
        logger.bind(run_id=self.run_id, turns=turns, total_tokens=total_tokens).info("")

    def format_clusters_for_logging(self, clusters: dict[int, list[str]]):
        """
        Transforms a dictionary of clusters into a structured list of tuples
        suitable for heavy logging.

        Parameters:
        - clusters (dict): A dictionary where keys are cluster IDs (int) and
                           values are lists of documents (strings).

        Returns:
        - list of tuples: Each tuple is (cluster_id, cluster_title, [documents])
        """
        logged_clusters = []

        # Sort by cluster_id to ensure consistent, readable log order
        for cluster_id, docs in sorted(clusters.items()):
            # Determine the user-friendly title
            title = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"

            # Append the full data structure (keeping entire doc strings intact)
            logged_clusters.append((cluster_id, title, list(docs)))

        return logged_clusters
