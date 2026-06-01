import json
import os

from openai import OpenAI

from data.judgement_result import JudgementResult
from utils.prompt_loader import get_prompt


def make_client(model_name: str) -> tuple[OpenAI, str]:
    openai_models = {"gpt-4o"}
    if model_name in openai_models:
        return OpenAI(
            base_url=os.environ["GITHUB_BASE_URL"],
            api_key=os.environ["GITHUB_TOKEN"],
        ), model_name

    if "OPENROUTER_API_URL" in os.environ:
        return OpenAI(
            base_url=os.environ["OPENROUTER_API_URL"],
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        ), model_name

    # Fallback to local Ollama instance
    return OpenAI(base_url=os.environ["OLLAMA_BASE_URL"], api_key="ollama"), model_name


class ClusterJudge:
    """
    Evaluates clustering quality using an LLM as judge.

    Parameters
    ----------
    model_name : str
        OpenAI-compatible model to use for judgement.
    examples_per_cluster : int
        Number of document examples shown per cluster. Keep this low to
        avoid blowing the context window on large clusters.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        examples_per_cluster: int = 20,
    ):
        self.model_name = model_name
        self.examples_per_cluster = examples_per_cluster
        self.client = make_client(self.model_name)

    def evaluate(
        self,
        user_preference: str,
        clusters: dict[int, list[str]],
    ) -> JudgementResult:
        """
        Parameters
        ----------
        user_preference : str
            The summarised clustering goal produced by the interview agent.
        clusters : dict[int, list[str]]
            Mapping of cluster_id -> list of document strings.
            Cluster id -1 is treated as noise and excluded from evaluation.

        Returns
        -------
        JudgementResult with coherence, alignment, separation scores and reasoning.
        """
        meaningful_clusters = {k: v for k, v in clusters.items() if k != -1}
        n_noise = len(clusters.get(-1, []))

        cluster_text = self._format_clusters(meaningful_clusters)

        prompt = get_prompt("judge_user_prompt.md").format(
            user_preference=user_preference,
            n_clusters=len(meaningful_clusters),
            n_noise=n_noise,
            cluster_text=cluster_text,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": get_prompt("judge_system_prompt.md")},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # deterministic scoring
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_clusters(self, clusters: dict[int, list[str]]) -> str:
        lines = []
        for cluster_id, docs in sorted(clusters.items()):
            sample = docs[: self.examples_per_cluster]
            lines.append(f"Cluster {cluster_id} ({len(docs)} docs):")
            lines.extend(f"  - {doc[:100]}" for doc in sample)
            if len(docs) > self.examples_per_cluster:
                lines.append(f"  ... and {len(docs) - self.examples_per_cluster} more")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> JudgementResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Judge returned non-JSON response:\n{raw}") from exc

        try:
            return JudgementResult(
                coherence=int(data["coherence"]),
                alignment=int(data["alignment"]),
                separation=int(data["separation"]),
                reasoning=data["reasoning"],
            )
        except KeyError as exc:
            raise ValueError(
                f"Judge response missing expected field {exc}:\n{data}"
            ) from exc
