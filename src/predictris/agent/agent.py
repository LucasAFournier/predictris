# predictris/learning/agent.py
from __future__ import annotations

import glob
import random
from collections import deque
from pathlib import Path
from typing import Callable
from uuid import UUID
import time

from tqdm import tqdm

from predictris.learning.prediction_tree import (
    Context,
    ObservationAction as OA,
    PredictionTree,
)

# centralised metrics hub -------------------------------------------------
from .metrics import MetricsRegistry


class Agent:
    """Learning agent that explores an environment and builds a forest of prediction trees."""

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

        self.trees_by_pred: dict[tuple, PredictionTree] = {}
        self.next_actions: deque[int] = deque()
        self.active_contexts_by_pred: dict[tuple, dict[UUID, Context]] = {}

        # ---------- metrics initialization --------------------------------
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

    def init_learn_episode(self, action: str, activation: str):
        """Prime the agent and reset metrics."""
        self.action_choice = action          # 'random' | 'from_active'
        self.activation = activation                # 'all' | 'by_confidence'

        # update -----------------------------------------------------------
        obs = self.observe()
        self.trees_by_pred.setdefault(obs, PredictionTree(obs, depth=self.depth))
        self.prev_obs = obs

    def learn(self):
        start = time.time()
        
        # ---------- act ---------------------------------------------------
        action = self._get_next_action()
        self.perform(action)
        self.last_oa = OA(self.prev_obs, action)

        # ---------- perceive ---------------------------------------------
        obs = self.observe()

        tree = self.trees_by_pred.setdefault(obs, PredictionTree(obs, depth=self.depth))
        tree.reinforce_correct_prediction(Context(self.last_oa, tree.pred_node))

        # ---------- context propagation ----------------------------------
        new_active: dict[tuple, dict[UUID, Context]] = {}

        for pred, tree in self.trees_by_pred.items():
            matching = tree.get_nodes_from_obs(obs).copy()
            updated: dict[UUID, Context] = {}

            for prev_node, ctx in self.active_contexts_by_pred.get(pred, {}).items():
                _, next_node, edge_action = next(
                    iter(tree.out_edges(prev_node, data="action"))
                )
                if edge_action != action:
                    continue

                # reached prediction node => evaluate
                if next_node == tree.pred_node:
                    correct = (obs == pred)
                    tree.update_prediction(ctx, correct_pred=correct)                       

                # ordinary transition
                elif next_node in matching:
                    updated[next_node] = ctx
                    matching.remove(next_node)

            # spawn new contexts
            for node in matching:
                if self.activation == "all" or (
                    self.activation == "by_confidence"
                    and random.random() < tree.nodes[node]["confidence"]
                ):
                    updated[node] = Context(self.last_oa, node)

            new_active[pred] = updated

        self.active_contexts_by_pred = new_active
        self.prev_obs = obs

        # ---------- metrics step -----------------------------------------
        self._metrics.emit("learn_step", time=time.time() - start)

    def init_test_episode(self, action_choice: str, activation: str):
        """Prime the agent and reset metrics."""
        self.action_choice = action_choice
        self.activation = activation

        # update -----------------------------------------------------------
        obs = self.observe()
        self.trees_by_pred.setdefault(obs, PredictionTree(obs, depth=self.depth))
        self.prev_obs = obs

        self._metrics.emit("test_episode")

    def test(self):
        # ---------- act ---------------------------------------------------
        action = self._get_next_action()
        self.perform(action)
        self.last_oa = OA(self.prev_obs, action)

        # ---------- perceive ---------------------------------------------
        obs = self.observe()

        tree = self.trees_by_pred.setdefault(obs, PredictionTree(obs, depth=self.depth))
        tree.reinforce_correct_prediction(Context(self.last_oa, tree.pred_node))

        # ---------- context propagation ----------------------------------
        new_active: dict[tuple, dict[UUID, Context]] = {}

        for pred, tree in self.trees_by_pred.items():
            matching = tree.get_nodes_from_obs(obs).copy()
            updated: dict[UUID, Context] = {}

            for prev_node, ctx in self.active_contexts_by_pred.get(pred, {}).items():
                _, next_node, edge_action = next(
                    iter(tree.out_edges(prev_node, data="action"))
                )
                if edge_action != action:
                    continue

                # reached prediction node => evaluate
                if next_node == tree.pred_node:
                    correct = (obs == pred)
                    self._metrics.emit("path_close", pred=pred, node=prev_node)
                    if tree.nodes[ctx.node].get("confident", False):
                        confidence = tree.nodes[ctx.node]["confidence"]
                        self._metrics.emit("prediction", correct=correct, confidence=confidence)                        

                # ordinary transition
                elif next_node in matching:
                    updated[next_node] = ctx
                    matching.remove(next_node)
                    self._metrics.emit(
                        "path_transfer",
                        pred=pred,
                        from_node=prev_node,
                        to_node=next_node,
                    )

            # spawn new contexts
            for node in matching:
                if self.activation == "all" or (
                    self.activation == "by_confidence"
                    and random.random() < tree.nodes[node]["confidence"]
                ):
                    updated[node] = Context(self.last_oa, node)
                    self._metrics.emit("path_open", pred=pred, node=node)

            new_active[pred] = updated

        self.active_contexts_by_pred = new_active
        self.prev_obs = obs

        # ---------- metrics step -----------------------------------------
        self._metrics.emit("test_step")

    def _get_next_action(self) -> int:
        if not self.next_actions:
            self._choose_actions()
        return self.next_actions.popleft()

    def _choose_actions(self):
        if (
            self.action_choice == "from_active"
            and (active := self._get_active_nodes())
        ):
            pred, node = random.choice(list(active))
            self.next_actions.extend(
                self.trees_by_pred[pred].get_actions_to_pred(node)
            )
        else:
            self.next_actions.append(random.choice(list(self.action_dict)))

    def _get_active_nodes(self) -> set[tuple[tuple, UUID]]:
        return {
            (pred, node)
            for pred, ctxs in self.active_contexts_by_pred.items()
            for node in ctxs
        }

    def load(self, dir: Path, verbose: bool = False):
        if not dir.exists():
            raise FileNotFoundError(f"Directory not found: {dir}")

        paths = glob.glob(str(dir / "*.gpickle"))
        for fp in tqdm(paths, desc="Loading trees", leave=False):
            tree = PredictionTree.load(Path(fp))
            self.trees_by_pred[tree.pred_obs] = tree
        if verbose:
            print(f"Loaded {len(paths)} trees from {dir}")

    def save(self, dir: Path, verbose: bool = False):
        if verbose:
            print(f"\nSaving trees to {dir}")
        for tree in self.trees_by_pred.values():
            tree.save(dir)