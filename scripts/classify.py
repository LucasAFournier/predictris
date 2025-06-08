import argparse
import numpy as np
from coclust.clustering import CoclustSpectral
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Perform co-clustering on distance matrix')
    parser.add_argument('--input', type=str, help='Path to distances.npy file')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--visualize', action='store_true', help='Display clustering results')
    parser.add_argument('--save', action='store_true', help='Save clustering results')
    return parser.parse_args()

def prepare_matrix(distances: dict) -> np.ndarray:
    """Convert distance dictionary to matrix suitable for clustering."""
    # Get all unique observations
    all_obs = sorted(set(list(distances.keys()) + 
                        [end for d in distances.values() for end in d.keys()]))
    n = len(all_obs)
    matrix = np.full((n, n), float('inf'))
    np.fill_diagonal(matrix, 0)
    
    # Fill known distances
    obs_to_idx = {obs: idx for idx, obs in enumerate(all_obs)}
    for start, ends in distances.items():
        i = obs_to_idx[start]
        for end, dist in ends.items():
            j = obs_to_idx[end]
            matrix[i, j] = dist
            
    # Replace infinities with large finite value
    max_finite = np.max(matrix[matrix != float('inf')])
    matrix[matrix == float('inf')] = max_finite * 2
    
    return matrix

def perform_clustering(matrix: np.ndarray, n_clusters: int):
    """Perform co-clustering on the distance matrix."""
    model = CoclustSpectral(n_clusters=n_clusters)
    model.fit(matrix)
    return model.row_labels_

def visualize_clusters(matrix: np.ndarray, labels: np.ndarray):
    """Visualize the clustered distance matrix."""
    # Sort matrix by cluster labels
    idx = np.argsort(labels)
    sorted_matrix = matrix[idx][:, idx]
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(np.log1p(sorted_matrix), cmap='viridis',
                xticklabels=False, yticklabels=False)
    
    # Add cluster boundaries
    boundaries = np.where(np.diff(labels[idx]))[0] + 0.5
    for b in boundaries:
        plt.axhline(y=b, color='r', linestyle='-', linewidth=0.5)
        plt.axvline(x=b, color='r', linestyle='-', linewidth=0.5)
    
    plt.title(f'Clustered Distance Matrix ({len(np.unique(labels))} clusters)')
    plt.show()

def main():
    args = parse_args()
    
    # Load distance matrix
    input_path = Path(args.input)
    distances = np.load(input_path, allow_pickle=True).item()
    
    # Prepare and cluster
    matrix = prepare_matrix(distances)
    labels = perform_clustering(matrix, args.n_clusters)
    
    if args.save:
        output_dir = input_path.parent
        np.save(output_dir / 'cluster_labels.npy', labels)
    
    if args.visualize:
        visualize_clusters(matrix, labels)

if __name__ == '__main__':
    main()
