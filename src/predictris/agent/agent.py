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
        depth: int,
        verbose: bool = False,
    ):
        """Initialize agent with action and perception capabilities."""
        self.action_dict = action_dict
        self.perception_dict = perception_dict
        self.trees_by_pred = dict[tuple, PredictionTree]()
        self.verbose = verbose
        self.depth = depth

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
        
    def init_learn(self, priming_steps: int, action_choice: str, activation: str, metrics: str = None) -> tuple:
        """Initialize agent for learning mode."""
        self.init_metrics(metrics)
        
        # self.action_choice = 'random'
        self.action_choice = action_choice
        self.activation = activation
        
        self.next_actions = deque[int]()    
        self.active_contexts_by_pred = dict[tuple, dict[UUID, Context]]()

        self.update(init=True, learn=False)
        for i in range(priming_steps):
            self.update(init=False, learn=False)
        
    def init_metrics(self, metrics: str):
        if metrics == 'pred_success':
            self.metrics = {
                'preds': 0,
                'correct_preds': 0,
            }
        elif metrics == 'paths':
            self.metrics = {
                'step': 0,
                'active_paths_by_pred': dict[tuple, dict[UUID, int]](),
                'paths': list[tuple[int, int]](),
            }
        else:
            self.metrics = dict()

    def learn(self):
        """Perform a learning step."""
        self.update(learn=True)

    def update(self, init: bool = False, learn: bool = False):
        """Update agent state and prediction trees."""
        if init:
            action: int = None
            self.last_oa: OA = None
        else:
            action = self._get_next_action()
            self.perform(action)
            self.last_oa = OA(self.prev_obs, action)
        
        obs = self.observe()
        
        # Add new tree if observation is new
        current_tree = self.trees_by_pred.setdefault(obs, PredictionTree(obs, depth=self.depth))
        # Add new node if sequence is new
        current_tree.reinforce_correct_prediction(Context(self.last_oa, current_tree.pred_node))

        # Prepare updated active contexts
        new_active_contexts_by_pred = dict[tuple, dict[UUID, Context]]()
        if learn and 'paths' in self.metrics:
            new_active_paths_by_pred = dict[tuple, dict[UUID, int]]()
            self.metrics['step'] += 1

        for pred, tree in self.trees_by_pred.items():
            matching_nodes = tree.get_nodes_from_obs(obs).copy()
            updated_contexts = {}

            if learn and 'paths' in self.metrics:
                updated_paths = {}
            
            # Iterate over previous active contexts
            for prev_node, context in self.active_contexts_by_pred.get(pred, {}).items():
                # Get next node and action
                _, next_node, other_action = next(iter(tree.out_edges(prev_node, data='action')))
                if other_action == action:
                    if learn and next_node == tree.pred_node:
                        # Update prediction if next node is the prediction node
                        tree.update_prediction(
                            context,
                            correct_pred=(obs == pred),
                        )
                        if 'preds' in self.metrics:
                            self.metrics['preds'] += 1
                            self.metrics['correct_preds'] += int(obs == pred)
                        elif 'paths' in self.metrics:
                            start = self.metrics['active_paths_by_pred'][pred][prev_node]
                            self.metrics['paths'].append((start, self.metrics['step']))
                    
                    elif next_node in matching_nodes:
                        # Transfer context to next node if it matches previous observation and action
                        updated_contexts[next_node] = context
                        matching_nodes.remove(next_node)

                        if learn and 'paths' in self.metrics:
                            # Update active paths if applicable:
                            updated_paths[next_node] = self.metrics['active_paths_by_pred'][pred][prev_node]

            # Create new contexts for remaining candidate nodes
            for node in matching_nodes:
                if self.activation == 'all':
                    updated_contexts[node] = Context(self.last_oa, node)

                    if learn and 'paths' in self.metrics:
                        updated_paths[node] = self.metrics['step']

                elif self.activation == 'by_confidence':
                    if random.random() < tree.nodes[node]['confidence']:
                        updated_contexts[node] = Context(self.last_oa, node)

            new_active_contexts_by_pred[pred] = updated_contexts

            if learn and 'paths' in self.metrics:
                # Update active paths for the current prediction
                new_active_paths_by_pred[pred] = updated_paths

        self.active_contexts_by_pred = new_active_contexts_by_pred

        if learn and 'paths' in self.metrics:
            # Update metrics for active paths
            self.metrics['active_paths_by_pred'] = new_active_paths_by_pred
        
        self.prev_obs = obs
                    
    def _get_next_action(self) -> int:
        """Get next action."""
        if not self.next_actions:
            self._choose_actions()
        return self.next_actions.popleft()

    def _choose_actions(self):
        """Choose next actions."""
        if self.action_choice == 'from_active':
            if (active_nodes := self._get_active_nodes()):
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
    
    def load(self, dir: Path, verbose: bool = False):
        """Load agent with prediction trees from a directory."""
        if not dir.exists():
            raise FileNotFoundError(f"Directory not found: {dir}")
            
        filepaths = glob.glob(str(dir / "*.gpickle"))
        for filepath in tqdm(filepaths, desc="Loading trees", leave=False):
            tree = PredictionTree.load(Path(filepath))
            self.trees_by_pred[tree.pred_obs] = tree
            
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
    
    def perform(self, action: int):
        """Perform an action using the action dictionary."""
        self.action_dict[action](self)