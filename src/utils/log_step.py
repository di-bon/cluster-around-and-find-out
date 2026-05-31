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
