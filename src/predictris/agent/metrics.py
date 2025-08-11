from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type
import random


CONFIDENCES_SAMPLE_SIZE = 1000


class MetricsRegistry:
    """Very small *dict-like* pub-sub container with metric classes as keys."""

    def __init__(self, agent=None) -> None:
        self._metrics: Dict[Type, object] = {}
        self.agent = agent

    def __getitem__(self, key: Type | str) -> object:
        if isinstance(key, str):
            key = METRIC_MAP[key]
        return self._metrics[key]

    def __contains__(self, cls: Type) -> bool:
        return cls in self._metrics

    def __iter__(self):
        return iter(self._metrics.values())

    def register(self, metric: str) -> None:
        """Create and attach a metric instance from its name."""
        if metric_cls := METRIC_MAP.get(metric):
            metric_instance = metric_cls()
            metric_instance.registry = self
            self._metrics[metric_cls] = metric_instance

    def emit(self, event: str, **kwargs) -> None:
        """Broadcast an event with payload."""
        handler_name = f"on_{event}"
        for metric in self._metrics.values():
            handler = getattr(metric, handler_name, None)
            if handler:
                handler(**kwargs)


@dataclass
class ErrorRates:
    """Records prediction success rate for the highest confidence prediction at each step."""

    step: int = 0
    episodes: int = 0
    error_rates = dict[int, int]()
    current_conf: float = 0.0
    current_pred: bool = False

    def on_prediction(self, *, correct: bool, confidence: float) -> None:
        if confidence > self.current_conf:
            self.current_conf = confidence
            self.current_pred = correct

    def on_test_step(self) -> None:
        self.error_rates[self.step] = self.error_rates.get(self.step, 0) + int(
            self.current_pred
        )
        self.current_conf = 0.0
        self.current_pred = False
        self.step += 1

    def on_test_episode(self) -> None:
        self.episodes += 1
        self.step = 0

    def reset(self) -> None:
        self.step = 0
        self.episodes = 0
        self.error_rates.clear()
        self.current_conf = 0.0
        self.current_pred = False

    def result(self) -> float:
        result = {
            step: round(
                (self.episodes - self.error_rates[step]) / self.episodes, 4
            )
            for step in self.error_rates
        }
        self.reset()
        return result


@dataclass
class TimePerStep:
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


@dataclass
class NodesCount:
    """Records the number of nodes in the prediction tree."""

    registry: MetricsRegistry = field(init=False, repr=False)

    def reset(self) -> None:
        return

    def result(self) -> int:
        agent = self.registry.agent
        trees = agent.prediction_forest.values()
        nodes_count = sum(len(t) for t in trees)

        self.reset()
        return nodes_count


@dataclass
class Confidences:
    """Records the confidences of all nodes in the prediction trees."""

    registry: MetricsRegistry = field(init=False, repr=False)

    def reset(self) -> None:
        return

    def result(self) -> list[float]:
        confidences = []
        agent = self.registry.agent
        trees = agent.prediction_forest.values()
        for tree in trees:
            confidences.extend(
                [
                    data["confidence"]
                    for _, data in tree.nodes(data=True)
                    if "confidence" in data
                ]
            )

        if len(confidences) > CONFIDENCES_SAMPLE_SIZE:
            confidences = random.sample(confidences, k=CONFIDENCES_SAMPLE_SIZE)

        self.reset()
        return confidences


METRIC_MAP = {
    "nodes_count": NodesCount,
    "confidences": Confidences,
    "time_per_step": TimePerStep,
    "error_rate": ErrorRates,
}
