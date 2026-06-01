"""
SimulatedUser: uses an LLM to act as a user in the ClusteringInterviewAgent conversation loop.

The simulated user is given a randomly sampled persona + clustering goal at init time,
so that each run produces meaningfully different clustering preferences.
"""

import os
import random

from openai import OpenAI

from prompts.simulated_user.config import GOALS, PERSONAS
from utils.prompt_loader import get_prompt

# ---------------------------------------------------------------------------
# Persona pool — feel free to extend these
# ---------------------------------------------------------------------------


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


class SimulatedUser:
    """
    Wraps an OpenAI-compatible LLM to simulate a human user in the
    ClusteringInterviewAgent conversation loop.

    Parameters
    ----------
    model_name : str
        The name of the LLM used (e.g. "qwen3.6:35b")
    goal : str | None
        Explicit clustering goal. If None, one is sampled at random.
    persona : str | None
        Explicit persona description. If None, one is sampled at random.
    seed : int | None
        Random seed for reproducible persona/goal sampling.
    """

    def __init__(
        self,
        model_name: str = "qwen3.6:35b",
        goal: str | None = None,
        persona: str | None = None,
        seed: int | None = None,
    ):
        rng = random.Random(seed)
        self.goal = goal or rng.choice(GOALS)
        self.persona = persona or rng.choice(PERSONAS)
        self.model_name = model_name
        self.client, _ = make_client(self.model_name)
        self._history: list[dict] = []
        self._system = get_prompt("simulated_user/system_prompt.md").format(
            persona=self.persona, goal=self.goal
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(self, agent_message: str) -> str:
        """
        Given the agent's latest message, return the simulated user's reply.
        """
        self._history.append({"role": "user", "content": agent_message})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system},
                *self._history,
            ],
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def __repr__(self) -> str:
        return f"SimulatedUser(persona={self.persona!r}, goal={self.goal!r})"
