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

from .prediction_tree import Context, PredictionTree, update_node_metrics
from .metrics import MetricsRegistry


@dataclass
class Sequence:
    current_node: UUID
    source_node: UUID
    context: Context

    def __hash__(self):
        return hash((self.current_node, self.source_node))


class Agent:
    """
    Learning agent that explores an environment and builds a forest of
    prediction trees.
    """

    def __init__(
        self,
        action_dict: dict[int, Callable[["Agent"], None]],
        perception_dict: dict[int, Callable[["Agent"], tuple]],
        depth: int,
        verbose: bool = False,
        metrics: list[str] | None = None,
    ):
        self.action_dict = action_dict
        self.perception_dict = perception_dict
        self.depth = depth
        self.verbose = verbose

        self.prediction_forest = dict[tuple, PredictionTree]()
        self.active_sequences = dict[PredictionTree, set[Sequence]]()
        self.next_actions = deque[int]()

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

        # create tree if it doesn't exist for the observation
        self.prev_obs = self.observe()
        self.prediction_forest.setdefault(
            self.prev_obs, PredictionTree(self.prev_obs)
        )

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
            # create tree if it doesn't exist for the observation
            tree = self.prediction_forest.setdefault(
                obs, PredictionTree(obs)
            )
            # reinforce prediction node if the prev_obs - action - obs
            # combination is new
            tree.reinforce_correct_prediction(
                tree.pred_node, self.current_context
            )

        # context propagation
        new_active = dict[PredictionTree, set[Sequence]]()

        for tree in self.prediction_forest.values():
            new_active[tree] = self._update_sequences_for_tree(
                tree, obs, action, learn, test
            )

        self.active_sequences = new_active
        self.prev_obs = obs

        # metrics step
        if learn:
            self._metrics.emit("learn_step", time=time.time() - start)
        if test:
            self._metrics.emit("test_step")

    def _update_sequences_for_tree(
        self,
        tree: PredictionTree,
        obs: tuple,
        action: int,
        learn: bool,
        test: bool,
    ) -> set[Sequence]:
        matching = tree.get_nodes_from_obs(obs).copy()
        new_active = set[Sequence]()

        for sequence in self.active_sequences.get(tree, set()):
            _, next_node, edge_action = next(
                iter(tree.out_edges(sequence.current_node, data="action"))
            )
            # deactivate if the action does not match
            if edge_action != action:
                continue

            # reached prediction node => evaluate
            if next_node == tree.pred_node:
                correct = (obs == tree.pred_obs)
                if learn:
                    self._update_prediction(
                        tree,
                        sequence.source_node,
                        sequence.context,
                        correct_pred=correct,
                    )
                if test and tree.nodes[sequence.source_node]["confident"]:
                    self._metrics.emit(
                        "prediction",
                        correct=correct
                    )

            # ordinary transition
            elif next_node in matching:
                sequence.current_node = next_node
                new_active.add(sequence)
                matching.remove(next_node)

        # spawn new contexts for remaining nodes
        for node in matching:
            if self.activation == "all" or (
                self.activation == "by_confidence"
                and random.random() < tree.nodes[node]["confidence"]
            ):
                new_active.add(
                    Sequence(node, node, self.current_context)
                )

        return new_active
    
    def _update_prediction(
        self,
        tree: PredictionTree,
        source_node: UUID,
        context: Context,
        correct_pred: bool,
    ):
        """Update prediction tree based on context and correctness of prediction."""
        source_node = tree.nodes[source_node]
        if not correct_pred:
            source_node["confident"] = False
        elif not source_node["confident"]:
            if source_node["level"] < self.depth:
                tree.reinforce_correct_prediction(source_node, context)
            source_node["confident"] = True

        update_node_metrics(source_node, correct_pred)

    def _get_next_action(self) -> int:
        if not self.next_actions:
            self._choose_actions()
        return self.next_actions.popleft()

    def _choose_actions(self):
        if self.action_choice == "from_active" and (
            current := self._get_current_nodes()
        ):
            tree, node = random.choice(list(current))
            self.next_actions.extend(tree.get_actions_to_pred(node))
        else:
            self.next_actions.append(random.choice(list(self.action_dict)))

    def _get_current_nodes(self) -> set[tuple[PredictionTree, UUID]]:
        return {
            (tree, seq.current_node)
            for tree, sequences in self.active_sequences.items()
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
