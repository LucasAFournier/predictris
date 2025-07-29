from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from uuid import UUID


class MetricsRegistry:
    """Very small *dict-like* pub-sub container with metric classes as keys."""

    def __init__(self) -> None:
        self._metrics: Dict[Type, object] = {}

    def __getitem__(self, cls: Type) -> object:
        return self._metrics[cls]

    def __contains__(self, cls: Type) -> bool:
        return cls in self._metrics

    def __iter__(self):
        return iter(self._metrics.values())

    def register(self, metric: object) -> None:
        """Attach a metric instance to the registry."""
        self._metrics[type(metric)] = metric

    def emit(self, event: str, **kwargs) -> None:
        """Broadcast an event with arbitrary payload."""
        handler_name = f"on_{event}"
        for metric in self._metrics.values():
            handler = getattr(metric, handler_name, None)
            if handler:
                handler(**kwargs)


@dataclass
class PredMetric:
    """Counts total predictions and how many were correct."""
    preds: int = 0
    correct_preds: int = 0

    # event handler -----------------------------------------------------
    def on_prediction(self, *, correct: bool, confidence: float) -> None:
        self.preds += 1
        self.correct_preds += int(correct)


@dataclass
class PathsMetric:
    """
    Records life-cycles of “active paths” through prediction trees.

    - `step` advances each `on_step`
    - `active_paths_by_pred` keeps {prediction: {node_id: start_step}}
    - `paths` stores all finished (start, end) tuples
    """
    step: int = 0
    paths = list[tuple[int, int]]()
    active_paths_by_pred = dict[tuple, dict[UUID, int]]()

    def on_test_step(self) -> None:
        self.step += 1

    def on_path_open(self, *, pred: tuple, node: UUID) -> None:
        self.active_paths_by_pred.setdefault(pred, {})[node] = self.step

    def on_path_close(self, *, pred: tuple, node: UUID) -> None:
        start = self.active_paths_by_pred.get(pred, {}).pop(node, None)
        if start is not None:
            self.paths.append((start, self.step))

    def on_path_transfer(
        self, *, pred: tuple, from_node: UUID, to_node: UUID
    ) -> None:
        start = self.active_paths_by_pred.get(pred, {}).pop(from_node, None)
        if start is not None:
            self.active_paths_by_pred[pred][to_node] = start


@dataclass
class BestPredErrorRate:
    """Records prediction success rate for the highest confidence prediction at each step."""
    step: int = 0
    episodes: int = 0
    best_preds = dict[int, int]()
    current_best_conf: float = 0.0
    current_best_pred: bool = False
    
    def on_prediction(self, *, correct: bool, confidence: float) -> None:
        if confidence > self.current_best_conf:
            self.current_best_conf = confidence
            self.current_best_pred = correct

    def on_test_step(self) -> None:
        self.best_preds[self.step] = self.best_preds.get(self.step, 0) + int(self.current_best_pred)
        self.current_best_conf = 0.0
        self.current_best_pred = False
        self.step += 1

    def on_test_episode(self) -> None:
        self.episodes += 1
        self.step = 0

    def reset(self) -> None:
        self.step = 0
        self.episodes = 0
        self.best_preds.clear()
        self.current_best_conf = 0.0
        self.current_best_pred = False

    def result(self) -> float:
        result = {
            step: round((self.episodes - self.best_preds[step])/ self.episodes, 4)
            for step in self.best_preds
        }
        self.reset()
        return result


@dataclass
class TimePerLearnStep:
    """Records time spent per step."""
    times = list[float]()

    def on_learn_step(self, time: float) -> None:
        self.times.append(time)

    def reset(self) -> None:
        self.times.clear()

    def result(self) -> float:
        result = round(sum(self.times) / len(self.times), 4)
        self.reset()
        return result