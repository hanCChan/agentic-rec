"""Episode trajectory serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    step_idx: int
    action: Dict[str, Any]
    observation: Optional[str] = None
    ndcg: Optional[float] = None
    recall: Optional[float] = None
    delta_ndcg: Optional[float] = None
    step_reward: Optional[float] = None
    invalid: bool = False
    invalid_reason: Optional[str] = None
    penalty: float = 0.0


@dataclass
class EpisodeTrajectory:
    qid: str
    original_query: str
    target_items: List[str]
    mode: str
    steps: List[StepRecord] = field(default_factory=list)
    final_query: Optional[str] = None
    final_ndcg: float = 0.0
    final_recall: float = 0.0
    baseline_ndcg: Optional[float] = None
    num_search_calls: int = 0
    penalties: Dict[str, float] = field(default_factory=dict)
    total_penalty: float = 0.0
    process_reward_sum: float = 0.0
    final_reward: float = 0.0
    total_reward: float = 0.0
    terminated_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [asdict(s) for s in self.steps]
        return payload


def save_trajectories_jsonl(trajectories: List[EpisodeTrajectory], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for traj in trajectories:
            fout.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
