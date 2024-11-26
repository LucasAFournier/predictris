import json
import pickle
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.figure as mplfig
from io import BytesIO

from ..environment import TetrisEnv
from ..interactions import TetrisActions, TetrisVision


AUTHORIZED_ACTIONS = [
    # "move_tetromino_down",
    "move_tetromino_left",
    "move_tetromino_right",
    "rotate_tetromino_cw",
    # "rotate_tetromino_ccw",
    "move_view_down",
    "move_view_left",
    "move_view_right",
    "move_view_up",
]


class ExplorationTree:
    def __init__(
            self,
            n_steps: int,
            grid_shape: tuple[int, int],
            view_radius: int,
            authorized_actions: Optional[list[str]] = AUTHORIZED_ACTIONS,
            batch_size: Optional[int] = 0,
        ):
        """
        Initialize the exploration tree with the given grid shape and viewing radius.
        The tree is stored as a directed acyclic graph (DAG) using networkx.
        """
        self.tree = nx.DiGraph()
        self.max_depth = n_steps
        self.authorized_actions = authorized_actions
        self.n_actions = len(authorized_actions)
        self.batch_size = batch_size

        # Root node setup
        root_env = TetrisEnv(grid_shape)
        root_vision = TetrisVision(root_env, view_radius)
        root_observation = root_vision.observe().tolist()
        root_node_id = (0, 0)

        self.tree.add_node(root_node_id, value=root_observation)
        self.queue = [(root_env.get_state(), root_vision.get_state(), root_node_id)] # Compact representation of state

        # Explore the environment
        self.explore()

    def explore(self):
        """Explore all possible states by performing actions and creating new nodes."""
        node_buffer = []
        edge_buffer = []

        while self.queue:
            env_state, view_state, parent_node_id = self.queue.pop()
            
            for action_index, action_name in enumerate(self.authorized_actions):
                new_env_state, new_view_state, observation = self.perform_action(action_name, env_state, view_state)

                # Create a new node and edge
                child_node_id = self.add_node_with_edge(parent_node_id, observation, action_index, action_name, node_buffer, edge_buffer)
                
                # Add to queue if within depth limit
                current_depth, _ = parent_node_id
                if current_depth + 1 < self.max_depth:
                    self.queue.append((new_env_state, new_view_state, child_node_id))
                
                # Add nodes and edges in buffers if batch size exceeded or exploration completed
                if len(node_buffer) > self.batch_size or not self.queue:
                    self.tree.add_nodes_from(node_buffer)
                    self.tree.add_edges_from(edge_buffer)

    def perform_action(self, action_name: str, env_state: dict, view_state: dict):
        """Perform an action and return the new environment state, view state, and observation."""
        # Create simulation environment and vision from states
        env = TetrisEnv.from_state(env_state)
        vision = TetrisVision.from_state(view_state, env)
        
        # Perform action
        try:
            action = TetrisActions(env, vision).__getattribute__(action_name)
        except:
            raise Exception(f"{action_name} is not a defined action.")        
        
        try:
            action()
        except:
            raise Exception(f"The action {action_name} failed.")

        observation = vision.observe().tolist() # list representation for better readability
        
        new_env_state = env.get_state()
        new_view_state = vision.get_state()
        
        return new_env_state, new_view_state, observation

    def add_node_with_edge(self, parent_id, observation, action_index, action_name, node_buffer, edge_buffer):
        """Create a new node and an edge connecting it to its parent, and add them to the buffers."""
        # Compute child node ID
        parent_depth, parent_position = parent_id
        child_node_id = (parent_depth + 1, self.n_actions * parent_position + action_index)
        
        # Append to respective buffers
        node_buffer.append((child_node_id, {"value": observation}))
        edge_buffer.append((parent_id, child_node_id, {"label": action_name}))

        return child_node_id

    def save_graph(self, filepath: str, format: str = "json"):
        """
        Save the exploration tree to a file in the specified format.

        Args:
            filepath (str): Path to the file where the graph will be saved.
            format (str): Format to save the graph, options are 'json', 'graphml', 'gml', 'pickle'. Default is 'json'.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Save graph in JSON format
            data = nx.readwrite.json_graph.node_link_data(self.tree)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)
        elif format == "graphml":
            # Save graph in GraphML format
            nx.write_graphml(self.tree, filepath)
        elif format == "gml":
            # Save graph in GML format
            nx.write_gml(self.tree, filepath)
        elif format == "pickle":
            # Save graph in Pickle format
            with open(filepath, "wb") as f:
                pickle.dump(self.tree, f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Choose from 'json', 'graphml', 'gml', 'pickle'.")

    def visualize(self):
        # Ensure the graph is directed
        if not isinstance(self.tree, nx.DiGraph):
            self.tree = nx.DiGraph(self.tree)

        # Use graphviz_layout for a tree-like structure
        pos = graphviz_layout(self.tree, prog='dot')

        fig, ax = plt.subplots(figsize=(20, 15))

        # Draw edges
        nx.draw_networkx_edges(self.tree, pos, ax=ax, arrows=False, style='dashed')

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.tree, 'label')
        nx.draw_networkx_edge_labels(self.tree, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        # Draw nodes with grid visualization
        for node, (x, y) in pos.items():
            value = self.tree.nodes[node]['value']
            matrix = np.array(value).reshape((3, 3), order='F')  # Adjust the reshape size according to your grid
            fig_matrix = mplfig.Figure(figsize=(0.5, 0.5), dpi=100)
            ax_matrix = fig_matrix.add_subplot(111)
            ax_matrix.matshow(matrix, cmap='viridis', interpolation='nearest')
            ax_matrix.axis('off')
            buffer = BytesIO()
            fig_matrix.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            img = plt.imread(buffer)
            imagebox = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

        # Axes configuration
        ax.invert_yaxis()  # Invert y-axis to have root at the top
        ax.axis('off')
        plt.title("Tetris Exploration Tree")
        plt.tight_layout()
        plt.savefig("exploration_tree_visualization.png")
        plt.show()