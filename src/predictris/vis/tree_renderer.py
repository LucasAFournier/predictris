# Set matplotlib backend before any other imports
import matplotlib

matplotlib.use("Agg")

from tqdm import tqdm
from pyvis.network import Network
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from predictris.agent import PredictionTree

from .image_handler import ImageHandler


TETRIS_ACTIONS = {
    0: "move_tetromino_up",
    1: "move_tetromino_left",
    2: "move_tetromino_right",
    3: "move_tetromino_down",
    4: "rotate_tetromino_cw",
}
TREE_CONFIG = """
{
    "physics": {
        "enabled": true,
        "stabilization": {
            "iterations": 100,
            "updateInterval": 50
        }
    },
    "layout": {
        "hierarchical": {
            "enabled": true,
            "direction": "DU",
            "sortMethod": "directed",
            "levelSeparation": 120,
            "nodeSpacing": 10,
            "parentCentralization": true,
            "edgeMinimization": true,
            "blockShifting": true
        }
    },
    "rendering": {
        "clustering": false,
        "hideEdgesOnDrag": true,
        "hideNodesOnDrag": true
    }
}
"""


class NodeProcessor:
    def __init__(self, image_handler: ImageHandler):
        self.image_handler = image_handler

    def create_node_props(self, node, data: dict) -> dict:
        base64_img = self.image_handler.get_node_image(data["obs"])

        size = max(15, 5 * (5 - data["level"]))

        props = {
            "n_id": str(node),
            "shape": "image",
            "image": f"data:image/png;base64,{base64_img}",
            "size": size,
            "font": {"size": max(1, 5 - data["level"])},
            "level": data["level"],
        }

        if data["level"] > 0:
            props.update(
                {
                    "title": f"Evaluations: {data['eval_count']}, Confidence: {round(data['confidence'], 2)}",
                    "shapeProperties": {
                        "useImageSize": False,
                        "useBorderWithImage": True,
                    },
                    "color": {
                        "border": confidence_score_to_color(
                            data["confidence"]
                        ),
                        "borderWidth": 5,
                    },
                }
            )

        return props

    def process_node_batch(self, nodes_batch) -> list[dict]:
        return [
            self.create_node_props(node, data) for node, data in nodes_batch
        ]


class PredictionTreeRenderer(Network):
    def __init__(self, tree: PredictionTree, n_workers=4, batch_size=100):
        super().__init__(
            height="600px", width="1500px", directed=True, notebook=False
        )
        self.set_options(TREE_CONFIG)

        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tree = tree

        self.image_handler = ImageHandler()
        self.node_processor = NodeProcessor(self.image_handler)

        self._render()

    def _render(self) -> None:
        try:
            nodes_data = self._process_nodes()
            self._add_nodes_to_network(nodes_data)
            self._process_edges()
        except Exception as e:
            print(f"Error rendering tree {self.tree.name}: {str(e)}")
            raise

    def _process_nodes(self) -> list[dict]:
        nodes_data = []

        queue = [(self.tree.pred_node, 0)]
        while queue:
            node, level = queue.pop(0)
            data = self.tree.nodes[node]
            data["level"] = level
            nodes_data.append((node, data))
            queue.extend(
                (predecessor, level + 1)
                for predecessor in self.tree.predecessors(node)
            )

        results = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(0, len(nodes_data), self.batch_size):
                batch = nodes_data[i : i + self.batch_size]
                futures.append(
                    executor.submit(
                        self.node_processor.process_node_batch, batch
                    )
                )

            for future in tqdm(
                futures,
                desc=f"Processing nodes for {self.tree.name}",
                leave=False,
            ):
                results.extend(future.result())

        return results

    def _process_edges(self) -> None:
        edges_data = list(self.tree.edges(data=True))
        for i in tqdm(
            range(0, len(edges_data), self.batch_size),
            desc=f"Processing edges for {self.tree.name}",
            leave=False,
        ):
            batch = edges_data[i : i + self.batch_size]
            for u, v, edge_data in batch:
                self.add_edge(
                    source=str(u),
                    to=str(v),
                    title=TETRIS_ACTIONS[edge_data["action"]],
                    label=str(edge_data["action"]),
                    font={"size": 12},
                    color={"color": "#2B7CE9"},
                )

    def _add_nodes_to_network(self, nodes_data: list[dict]) -> None:
        for node_props in tqdm(
            nodes_data, desc=f"Adding nodes for {self.tree.name}", leave=False
        ):
            self.add_node(**node_props)

    def _calculate_tree_metrics(self, tree: PredictionTree) -> dict:
        """Calculate comprehensive metrics for the prediction tree."""
        nodes_data = list(tree.nodes(data=True))
        levels = [data["level"] for _, data in nodes_data if "level" in data]
        confidences = [
            data["confidence"]
            for _, data in nodes_data
            if "confidence" in data
        ]
        level_counts = dict(Counter(levels))

        return {
            "total_nodes": len(nodes_data),
            "level_counts": level_counts,
            "confidences": confidences,
            "max_depth": max(level_counts, default=0),
        }


def confidence_score_to_color(confidence: float) -> str:
    """Convert confidence score to color hex code (red -> yellow -> green gradient)."""
    if confidence <= 0.5:
        # Red to yellow
        ratio = confidence * 2  # 0-0.5 to 0-1
        green = int(255 * ratio)
        red = 255
    else:
        # Yellow to green
        ratio = (confidence - 0.5) * 2  # 0.5-1 to 0-1
        green = 255
        red = int(255 * (1 - ratio))

    return f"#{red:02x}{green:02x}00"
