import argparse
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import product
import warnings

from predictris.tetris.encoders import actions_from_nameset, perceptions_from_nameset
from predictris.utils import create_random_environment, dir_from_params


def build_transition_graph(episode: int, steps: int, view: tuple[int, int]):
    """Build transition graph from random episodes."""
    graph = nx.DiGraph()
    auth_view = (view[0] - 2, view[1] - 2)
    auth_positions = list(product(
        range(-auth_view[0], auth_view[0] + 1),
        range(-auth_view[1], auth_view[1] + 1),
    ))
    
    pbar = tqdm(range(epochs), desc=f"Building graph", leave=False)
    
    for _ in pbar:        
        env = create_random_environment(auth_view=auth_view)

        curr_obs = agent.observe()
        for _ in range(steps):
            prev_obs = curr_obs
            body.act((action := random.choice(actions)))
            curr_obs = body.perceive(perception)
                
            if env.pos not in auth_positions:
                break

            graph.add_node(prev_obs, tetromino=env.name)
            graph.add_node(curr_obs, tetromino=env.name)
            graph.add_edge(
                prev_obs,
                curr_obs,
                action=action,
            )

    return graph


def sort_nodes_by_tetromino(graph: nx.DiGraph) -> tuple[list, list]:
    """Sort nodes by tetromino name and return both nodes and their tetromino names."""
    nodes = list(graph.nodes())
    # Get tetromino names for each node
    tetromino_names = [graph.nodes[node]['tetromino'] for node in nodes]
    # Sort both lists based on tetromino names
    sorted_pairs = sorted(zip(nodes, tetromino_names), key=lambda x: x[1])
    # Unzip the sorted pairs
    sorted_nodes, sorted_names = zip(*sorted_pairs)
    return list(sorted_nodes), list(sorted_names)


def compute_distance_matrix(graph, desc="Computing distances", timeout=1.0):
    nodes, tetromino_names = sort_nodes_by_tetromino(graph)
    n = len(nodes)
    distances = np.full((n, n), np.inf)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist_dict = dict(nx.all_pairs_shortest_path_length(graph))
        
    # Convert dictionary to matrix using sorted nodes
    for i, source in tqdm(enumerate(nodes), total=n):
        if source in dist_dict:
            for target, dist in dist_dict[source].items():
                j = nodes.index(target)
                distances[i, j] = dist
    
    return distances, nodes, tetromino_names


def get_tetromino_indices(tetromino_names: list) -> dict:
    """Get start and end indices for each tetromino in the sorted list."""
    indices = {}
    current_tetromino = None
    start_idx = 0
    
    for i, tetromino in enumerate(tetromino_names):
        if tetromino != current_tetromino:
            if current_tetromino is not None:
                indices[current_tetromino] = (start_idx, i)
            current_tetromino = tetromino
            start_idx = i
    
    # Add the last tetromino
    if current_tetromino is not None:
        indices[current_tetromino] = (start_idx, len(tetromino_names))
    
    return indices


def plot_tetromino_matrix(matrix, tetromino_name, start_idx, end_idx):
    """Plot distance matrix for a specific tetromino in a new window."""
    submatrix = matrix[start_idx:end_idx, start_idx:end_idx].copy()
    submatrix[submatrix == float('inf')] = np.nan
    
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    sns.heatmap(submatrix, cmap='viridis',
                xticklabels=False, yticklabels=False,
                ax=ax, cbar_kws={'label': 'Distance (number of actions)'})
    
    plt.title(f'Distance Matrix\n(shape: {submatrix.shape})')
    fig.suptitle(f'Tetromino: {tetromino_name}', y=0.95, fontsize=14)


def visualize_matrix(matrix: np.ndarray, nodes: list, tetromino_names: list) -> None:
    """Display full distance matrix and individual tetromino matrices."""
    # Plot full matrix
    fig = plt.figure(figsize=(12, 12))
    matrix_viz = matrix.copy()
    matrix_viz[matrix == float('inf')] = np.nan
    
    ax = plt.gca()
    heatmap = sns.heatmap(matrix_viz, cmap='viridis',
                         xticklabels=False, yticklabels=False,
                         ax=ax, cbar_kws={'label': 'Distance (number of actions)'})
    
    # Add tetromino labels
    tetromino_indices = get_tetromino_indices(tetromino_names)
    for tetromino, (start, end) in tetromino_indices.items():
        # Calculate midpoint for the tetromino section
        mid = start + (end - start) // 2
        # Add text label on the left side
        ax.text(-0.1, mid, tetromino, 
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.get_yaxis_transform(),
                fontsize=10,
                rotation=0)
    
    plt.title(f'Full Distance Matrix (shape: {matrix.shape})')


def parse_args():
    parser = argparse.ArgumentParser(description='Compute distance matrix between Tetris states')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=100, help='Number of actions per episode')
    parser.add_argument('--view', type=int, nargs=2, default=[3, 3], help='View size (height width)')
    parser.add_argument('--save', action='store_true', help='Save results to disk')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize environment parameters
    view = tuple(args.view)
    actions = list(actions_from_nameset("restrained"))
    perception = next(iter(perceptions_from_nameset('vision_only')))
    
    # Build transition graph
    graph = build_transition_graph(actions, perception, args.epochs, args.steps, view)
        
    # Compute distances
    distances, nodes, tetromino_names = compute_distance_matrix(graph)
    
    # Save results
    if args.save:
        output_dir = dir_from_params(
        episodes=args.episodes,
        steps=args.steps
        )
        output_dir = f"results/distance_matrix/{output_dir}"
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/distances.npy", distances)
        nx.write_gpickle(graph, f"{output_dir}/graph.gpickle")
        np.save(f"{output_dir}/nodes.npy", nodes)
            
    # Print statistics for each tetromino
    finite_distances = distances[np.isfinite(distances)]
        
    if len(finite_distances) > 0:
        print(f"\nStatistics:")
        print(f"Number of states: {len(graph.nodes)}")
        print(f"Number of transitions: {len(graph.edges)}")
        print(f"Average distance: {np.mean(finite_distances):.2f}")
        print(f"Maximum distance: {np.max(finite_distances):.2f}")
        print(f"Proportion of reachable states: {len(finite_distances)/(len(graph.nodes)**2):.2%}")
        
    if args.visualize:
        visualize_matrix(distances, nodes, tetromino_names)
        plt.show()  # Show all windows at once


if __name__ == '__main__':
    main()