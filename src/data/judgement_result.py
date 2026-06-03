from dataclasses import dataclass


@dataclass
class JudgementResult:
    coherence: int
    alignment: int
    separation: int
    reasoning: str

    @property
    def mean_score(self) -> float:
        return (self.coherence + self.alignment + self.separation) / 3
