import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

from predictris.agent import StandardAgent
from predictris.tetris.encoders import perceptions_from_nameset
from predictris.tetris.constants import TETROMINO_NAMES
from predictris.utils import create_random_environment
from predictris.vis import CMAP


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tetris simulation with smart agent')
    parser.add_argument('--origin', type=str, required=True, help='Directory containing trained trees')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    return parser.parse_args()


def build_confidence_matrix(agent: StandardAgent) -> tuple[np.ndarray, list]:
    """Build matrix of confidence values between all observations."""
    observations = list(agent.tree_from_pred_obs.keys())
    n = len(observations)
    obs_to_idx = {obs: i for i, obs in enumerate(observations)}
    
    confidences = np.zeros((n, n))
    for from_obs in observations:
        i = obs_to_idx[from_obs]
        for to_obs in observations:
            j = obs_to_idx[to_obs]
            confidences[i,j] = agent.get_confidence_between(from_obs, to_obs)
            
    return confidences, observations


def visualize_matrix(matrix: np.ndarray) -> None:
    """Save confidence matrix visualization to file."""
    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    heatmap = sns.heatmap(matrix, cmap='viridis',
                         xticklabels=False, yticklabels=False,
                         ax=ax, cbar_kws={'label': 'Confidence'})
    
    plt.title(f'Confidence Matrix (shape: {matrix.shape})')
    plt.savefig('confidence_matrix.png')
    plt.close()


def visualize_tsne(matrix: np.ndarray, observations: list) -> None:
    """Save t-SNE visualization to file."""
    symmetric_matrix = (matrix + matrix.transpose()) / 2
    
    tsne = TSNE(
        n_components=2,
        metric='precomputed',
        random_state=42,
        init='random'
    )
    embedding = tsne.fit_transform(1 - symmetric_matrix)
    
    # Normalize embeddings to [0,1] range
    embedding = (embedding - embedding.min(axis=0)) / (embedding.max(axis=0) - embedding.min(axis=0))
    
    # Create a grid of positions
    grid_size = int(np.sqrt(len(observations)) * 5)  # Make grid 5x larger than needed
    grid_positions = np.zeros((grid_size, grid_size), dtype=bool)
    scale = 1/grid_size  # Scale based on grid size
    
    # Create figure with white background
    fig = plt.figure(figsize=(15, 15), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Plot each observation as an image
    for obs, (x, y) in zip(observations, embedding):
        # Convert normalized coordinates to grid coordinates
        grid_x = int(x * (grid_size-1))
        grid_y = int(y * (grid_size-1))
        
        # Find closest available grid position
        radius = 0
        placed = False
        while not placed and radius < grid_size:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    new_x = grid_x + dx
                    new_y = grid_y + dy
                    if (0 <= new_x < grid_size and 
                        0 <= new_y < grid_size and 
                        not grid_positions[new_x, new_y]):
                        grid_positions[new_x, new_y] = True
                        x = new_x / (grid_size-1)
                        y = new_y / (grid_size-1)
                        placed = True
                        break
                if placed:
                    break
            radius += 1
        
        # Plot the observation
        obs_matrix = np.array(obs).reshape(3, 3)
        extent = [x-scale/2, x+scale/2, y-scale/2, y+scale/2]
        
        ax.imshow(
            obs_matrix,
            extent=extent,
            cmap=CMAP,
            vmin=-1,
            vmax=15
        )
    
    plt.margins(0.1)
    
    plt.title('t-SNE visualization of observation space')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Load trained agent
    perception = next(iter(perceptions_from_nameset('vision_only')))
    input_dir = Path('results') / args.origin
    agent = StandardAgent.load(
        dir_path=input_dir,
        perception=perception,
        verbose=args.verbose
    )
    
    # Build and analyze confidence matrix
    confidences, observations = build_confidence_matrix(agent)
    print(f"Distance matrix shape: {confidences.shape}")
    print(f"Mean confidence: {confidences.mean():.3f}")
    print(f"Max confidence: {confidences.max():.3f}")
    
    if args.visualize:
        visualize_matrix(confidences)
        visualize_tsne(confidences, observations)
        print("Visualizations saved as 'confidence_matrix.png' and 'tsne_visualization.png'")

if __name__ == '__main__':
    main()
