from __future__ import annotations

import glob
import random
from collections import deque
from pathlib import Path
from typing import Callable
from uuid import UUID
import time
from dataclasses import dataclass
from tqdm import tqdm

from predictris.learning import Context, PredictionTree

from .metrics import MetricsRegistry


@dataclass
class ActiveSequence:
    current_node: UUID
    entry_node: UUID
    context: Context

    def __hash__(self):
        return hash(self.entry_node)

    def __eq__(self, other):
        if not isinstance(other, ActiveSequence):
            return False
        return self.entry_node == other.entry_node


class Agent:
    """
    Learning agent that explores an environment and builds a forest of
    prediction trees.
    """

    def __init__(
        self,
        action_dict: dict[int, Callable[["Agent"], tuple]],
        perception_dict: dict[int, Callable[["Agent"], None]],
        depth: int,
        verbose: bool = False,
        metrics: list[str] | None = None,
    ):
        self.action_dict = action_dict
        self.perception_dict = perception_dict
        self.depth = depth
        self.verbose = verbose

        self.prediction_forest: dict[tuple, PredictionTree] = {}
        self.next_actions: deque[int] = deque()
        self.active_sequences_by_pred: dict[tuple, set[ActiveSequence]] = {}

        # metrics initialization
        self._metrics: MetricsRegistry = MetricsRegistry(self)
        if metrics:
            for metric in metrics:
                self._metrics.register(metric)

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    def observe(self) -> tuple:
        if len(self.perception_dict) == 1:
            return next(iter(self.perception_dict.values()))(self)
        return tuple(
            (k, perceive(self)) for k, perceive in self.perception_dict.items()
        )

    def perform(self, action: int):
        self.action_dict[action](self)

    def init_episode(
        self,
        action_choice: str,
        activation: str,
        learn: bool = False,
        test: bool = False,
    ):
        """Prime the agent and reset metrics."""
        self.action_choice = action_choice  # 'random' | 'from_active'
        self.activation = activation  # 'all' | 'by_confidence'

        # update
        obs = self.observe()
        self.prediction_forest.setdefault(
            obs, PredictionTree(obs, depth=self.depth)
        )
        self.prev_obs = obs

        if test:
            self._metrics.emit("test_episode")

    def update(self, learn: bool = False, test: bool = False):
        start = time.time()

        # act 
        action = self._get_next_action()
        self.perform(action)
        self.current_context = Context(self.prev_obs, action)

        # perceive 
        obs = self.observe()

        if learn:
            # reinforce prediction node
            tree = self.prediction_forest.setdefault(
                obs, PredictionTree(obs, depth=self.depth)
            )
            tree.reinforce_correct_prediction(
                tree.pred_node, self.current_context
            )

        # context propagation
        new_active: dict[tuple, set[ActiveSequence]] = {}

        for pred, tree in self.prediction_forest.items():
            matching = tree.get_nodes_from_obs(obs).copy()
            updated: set[ActiveSequence] = set()

            for sequence in self.active_sequences_by_pred.get(pred, set()):
                _, next_node, edge_action = next(
                    iter(tree.out_edges(sequence.current_node, data="action"))
                )
                if edge_action != action:
                    continue

                # reached prediction node => evaluate
                if next_node == tree.pred_node:
                    correct = obs == pred
                    if learn:
                        tree.update_prediction(
                            sequence.entry_node,
                            sequence.context,
                            correct_pred=correct,
                        )
                    if test and tree.nodes[sequence.entry_node].get(
                        "confident", False
                    ):
                        confidence = tree.nodes[sequence.entry_node][
                            "confidence"
                        ]
                        self._metrics.emit(
                            "prediction",
                            correct=correct,
                            confidence=confidence,
                        )

                # ordinary transition
                elif next_node in matching:
                    sequence.current_node = next_node
                    updated.add(sequence)
                    matching.remove(next_node)

            # spawn new contexts
            for node in matching:
                if self.activation == "all" or (
                    self.activation == "by_confidence"
                    and random.random() < tree.nodes[node]["confidence"]
                ):
                    updated.add(
                        ActiveSequence(node, node, self.current_context)
                    )

            new_active[pred] = updated

        self.active_sequences_by_pred = new_active
        self.prev_obs = obs

        # metrics step
        if learn:
            self._metrics.emit("learn_step", time=time.time() - start)
        if test:
            self._metrics.emit("test_step")

    def _get_next_action(self) -> int:
        if not self.next_actions:
            self._choose_actions()
        return self.next_actions.popleft()

    def _choose_actions(self):
        if self.action_choice == "from_active" and (
            current := self._get_current_nodes()
        ):
            pred, node = random.choice(list(current))
            self.next_actions.extend(
                self.prediction_forest[pred].get_actions_to_pred(node)
            )
        else:
            self.next_actions.append(random.choice(list(self.action_dict)))

    def _get_current_nodes(self) -> set[tuple[tuple, UUID]]:
        return {
            (pred, seq.current_node)
            for pred, sequences in self.active_sequences_by_pred.items()
            for seq in sequences
        }

    def load(self, dir: Path, verbose: bool = False):
        if not dir.exists():
            raise FileNotFoundError(f"Directory not found: {dir}")

        paths = glob.glob(str(dir / "*.gpickle"))
        for fp in tqdm(paths, desc="Loading trees", leave=False):
            tree = PredictionTree.load(Path(fp))
            self.prediction_forest[tree.pred_obs] = tree
        if verbose:
            print(f"Loaded {len(paths)} trees from {dir}")

    def save(self, dir: Path, verbose: bool = False):
        if verbose:
            print(f"\nSaving trees to {dir}")
        for tree in self.prediction_forest.values():
            tree.save(dir)
