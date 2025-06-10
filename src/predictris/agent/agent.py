from uuid import UUID
from collections import deque
import glob
from pathlib import Path
import random
from tqdm import tqdm
from typing import Callable
        
from predictris.learning.prediction_tree import (
    PredictionTree,
    Context,
    ObservationAction as OA,
)

from .agent_utils import is_valid


class Agent:
    """
    Learning agent that interacts with environment using prediction trees.
    
    The agent maintains active nodes and uses prediction trees to learn patterns
    from sequences of actions and observations.
    """
    def __init__(
        self,
        action_dict: dict[int, Callable[["Agent"], tuple]],
        perception_dict: dict[int, Callable[["Agent"], None]],
        verbose: bool = False,
    ):
        """Initialize agent with action and perception capabilities."""
        self.action_dict = action_dict
        self.perception_dict = perception_dict
        self.trees_by_pred = dict[tuple, PredictionTree]()
        self.verbose = verbose
        
    #region Interaction

    def act(self, action: int = None):
        """Execute an action in the environment.
        
        Args:
            action (int, optional): Action to execute. Defaults to None.
        """
        if action is None and len(self.action_dict) == 1:
            action = next(iter(self.action_dict.keys()))

        self.action_dict[action](self)

    def observe(self) -> tuple:
        """Get perception from the environment.
        
        Returns:
            tuple: Observation from the environment.
        """
        if len(self.perception_dict) == 1:
            return next(iter(self.perception_dict.values()))(self)
        else:
            return tuple(
                (perception, perceive(self))
                for perception, perceive in self.perception_dict.items()
            )
        
    #endregion
    #region Learning

    def init_learn(self, action_choice: str, activation: str, metrics: bool = False) -> tuple:
        """Initialize agent for learning mode.
        
        Args:
            action_choice (str): Action choice strategy.
            activation (str): Activation strategy.
        
        Returns:
            tuple: Initial observation or 'aborted' if observation is invalid.
        """
        if metrics:
            self.metrics = {
                'preds': 0,
                'correct_preds': 0,
            }
        else:
            self.metrics = None
        
        self.action_choice = action_choice
        self.activation = activation
        
        obs = self.observe()
        self.last_oa: OA = None
        self.next_actions = deque[int]()
    
        self.active_contexts_by_pred = dict[tuple, dict[UUID, Context]]()
        
        if not is_valid(obs):
            return 'abort'
        
        self._update_current_tree(obs)
        self._update_active_contexts(obs, None)
        self.prev_obs = obs 

    def learn(self):
        """Learn from environment interaction."""

        action, obs = self._apply_action_and_observe()
        if not is_valid(obs):
            return 'abort'

        self._update_current_tree(obs)
        self._update_active_contexts(obs, action)
        self.prev_obs = obs

    def _apply_action_and_observe(self):
        action = self._get_next_action()
        self.act(action)
        self.last_oa = OA(self.prev_obs, action)
        obs = self.observe()

        return action, obs

    def _update_current_tree(self, obs: tuple):
        # Add new tree if observation is new
        current_tree = self.trees_by_pred.setdefault(obs, PredictionTree(obs))
        # Add new node if sequence is new
        current_tree.reinforce_correct_prediction(Context(self.last_oa, current_tree.pred_node))

    def _update_active_contexts(self, obs: tuple, action: int):
        # Prepare updated active contexts
        new_active_contexts_by_pred = dict[tuple, dict[UUID, Context]]()

        for pred, tree in self.trees_by_pred.items():
            candidate_nodes = tree.get_nodes_from_obs(obs).copy()
            updated_contexts = {}
            
            # Iterate over previous active contexts
            for prev_node, context in self.active_contexts_by_pred.get(pred, {}).items():
                # Get next node and action
                _, next_node, other_action = next(iter(tree.out_edges(prev_node, data='action')))
                if other_action == action:
                    if next_node == tree.pred_node:
                        # Update prediction if next node is the prediction node
                        tree.update_prediction(
                            context,
                            correct_pred=(obs == pred),
                        )
                        if self.metrics:
                            self.metrics['preds'] += 1
                            self.metrics['correct_preds'] += int(obs == pred)

                    elif next_node in candidate_nodes:
                        # Transfer context to next node if it matches previous observation and action
                        updated_contexts[next_node] = context
                        candidate_nodes.remove(next_node)

            # Create new contexts for remaining candidate nodes
            for node in candidate_nodes:
                if self.activation == 'all':
                    updated_contexts[node] = Context(self.last_oa, node)

                elif self.activation == 'by_confidence':
                    if random.random() < tree.nodes[node]['confidence']:
                        updated_contexts[node] = Context(self.last_oa, node)

            new_active_contexts_by_pred[pred] = updated_contexts

        self.active_contexts_by_pred = new_active_contexts_by_pred
                    
    def _get_next_action(self) -> int:
        """Get next action."""
        if not self.next_actions:
            self._choose_actions()
        return self.next_actions.popleft()

    def _choose_actions(self):
        """Choose next actions."""
        if self.action_choice == 'from_active':
            active_nodes = self._get_active_nodes()
            if active_nodes:
                pred, node = random.choice(active_nodes)
                tree = self.trees_by_pred[pred]
                self.next_actions.extend(tree.get_actions_to_pred(node))
            else:
                self.next_actions.append(random.choice(list(self.action_dict.keys())))

        elif self.action_choice == 'random':
            self.next_actions.append(random.choice(list(self.action_dict.keys())))
     
    def _get_active_nodes(self) -> set[tuple[tuple, UUID]]:
        """Get active nodes."""
        active = list[tuple[tuple, UUID]]()
        for pred, contexts in self.active_contexts_by_pred.items():
            for node in contexts:
                active.append((pred, node))
        return active

    #endregion
    
    def load(self, dir: Path, verbose: bool = False):
        """Load agent with prediction trees from a directory."""
        if not dir.exists():
            raise FileNotFoundError(f"Directory not found: {dir}")
            
        filepaths = glob.glob(str(dir / "*.gpickle"))
        for filepath in tqdm(filepaths, desc="Loading trees", leave=False):
            tree = PredictionTree.load(filepath)
            self.trees_by_pred[tree.pred] = tree
            
        if verbose:
            print(f"Loaded {len(filepaths)} trees from {dir}")

    def save(self, dir: Path, verbose: bool = False):
        if verbose:
            print(f"\nSaving trees to {dir}")
        for tree in self.trees_by_pred.values():
                    tree.save(dir)

    def get_confidence_between(self, from_obs: tuple, to_obs: tuple) -> float:
        """Calculate confidence of transitioning from one observation to another."""
        if to_obs not in self.trees_by_pred:
            return 0.0
            
        tree = self.trees_by_pred[to_obs]
        ids = tree.get_nodes_from_obs(from_obs)
        
        if not ids:
            return 0.0
            
        # Return confidence of most confident matching node
        return max(tree.nodes[id_]['confidence'] for id_ in ids)
    
    def get_nodes_count(self, filter: float = None) -> int:
        """Get total number of nodes in all trees."""
        if filter:
            return sum(
                len(
                    [node for node, confidence in tree.nodes(data='confidence', default=0.0)
                     if confidence > filter]
                )
                for tree in self.trees_by_pred.values()
            )
        
        return sum(len(tree) for tree in self.trees_by_pred.values())