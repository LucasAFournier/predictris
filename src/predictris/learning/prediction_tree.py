import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4
import networkx as nx

from .learning_utils import filename_from_obs, init_node_metrics, update_node_metrics


Observation = tuple
Action = int


class ObservationAction(NamedTuple):
    """Observation-Action pair representing a state transition."""
    observation: Observation
    action: Action


@dataclass
class Context:
    """Context state containing observation-action pair and entry node identifiers."""
    oa: ObservationAction
    node: UUID


class PredictionTreeCache(set[Observation]):
    def __init__(self):
        super().__init__()

    def get(self, obs: Observation) -> Observation:
        for existing in self:
            if existing == obs:
                del obs
                return existing

        self.add(obs)
        
        return obs


class PredictionTree(nx.DiGraph):
    _cache = PredictionTreeCache()
    
    def __init__(self, obs: Observation, depth: int, id_generator: callable = uuid4):
        super().__init__()
        self.depth = depth
        self.id_generator = id_generator
        self._nodes_by_obs = dict[Observation, set[UUID]]()
        
        self.pred_obs = self._cache.get(obs)
        self.pred_node = self.id_generator()
        self.add_node(self.pred_node, obs=self.pred_obs, level=0)

    def get_nodes_from_obs(self, obs: Observation) -> set[UUID]:
        """Find matching nodes."""
        return self._nodes_by_obs.get(self._cache.get(obs), set[UUID]())

    def update_prediction(self, context: Context, correct_pred: bool):
        """Update prediction tree based on context and correctness of prediction."""
        context_node = self.nodes[context.node]
        if not correct_pred:
            context_node["confident"] = False
        elif not context_node["confident"]:
            if context_node["level"] < self.depth:
                self.reinforce_correct_prediction(context)
            context_node["confident"] = True
            
        update_node_metrics(context_node, correct_pred)
        
    def get_actions_to_pred(self, node: UUID) -> list[Action]:
        """Get action sequence to prediction node."""
        actions = []
        while (edges := self.out_edges(node, data='action')):
            _, node, action = next(iter(edges))
            actions.append(action)
        return actions

    def reinforce_correct_prediction(self, context: Context):
        """Create new context path for correct predictions."""
        try:
            observation, action = context.oa

            # Check if the node already exists
            for existing_node, _, existing_action in self.in_edges(context.node, data='action'):
                if action == existing_action and self.nodes[existing_node]["obs"] == observation:
                    next_node = existing_node
                    raise StopIteration

            # If not, create a new node
            next_node = self._reinforce(observation, self.nodes[context.node]["level"] + 1)

            self.add_edge(next_node, context.node, action=action)

        except:
            pass


    def _reinforce(self, obs: Observation, level: int) -> UUID:
        """Add a new node to the tree and return its identifier."""
        obs = self._cache.get(obs)
        node = self.id_generator()        
        metrics = init_node_metrics()
        self.add_node(node, obs=obs, confident=True, level=level, **metrics)

        self._nodes_by_obs.setdefault(obs, set[UUID]()).add(node)        

        return node
    
    def save(self, output_dir: Path):
        """Save context trees to file."""
        filename = filename_from_obs(self.pred_obs)
        save_path = Path(f"results/{output_dir}/{filename}.gpickle")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "PredictionTree":
        """Load context trees from file."""
        if not filepath.exists():
            raise FileNotFoundError(f"No context trees file found at {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)