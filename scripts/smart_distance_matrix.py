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
import argparse
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from predictris.agent import StandardAgent
from predictris.tetris import TetrisEnvironment
from predictris.tetris.constants import TETROMINO_NAMES, BACKGROUND_VALUE
from predictris.tetris.encoders import perceptions_from_nameset, actions_from_nameset
from predictris.utils import create_random_environment, dir_from_params


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))

def compute_distance_matrix(
        states,
        n,
        agent: StandardAgent,
        max_steps: int,
):
    """Compute distance matrix between Tetris states."""    
    obs_from_state = {}
    for state in states:
        env: TetrisEnvironment = TetrisEnvironment.from_state(state)
        agent.body = env.get_body()
        obs_from_state[tuple(state)] = agent.get_obs()

    states = sorted(list(states), key=lambda x: (x['name'], obs_from_state[tuple(x)]))

    pbar = tqdm(total=n**2, desc="Computing distances", leave=False)
    distances = np.full((n, n), np.nan)
    for i, source in enumerate(states):
        for j, target in enumerate(states):
            env: TetrisEnvironment = TetrisEnvironment.from_state(source)
            agent.body = env.get_body()
            agent.init_path_finding()

            target_obs = obs_from_state[tuple(target)]
            obs_history, action_history, success = agent.find_path(target_obs, max_steps)
            
            if success:
                distances[i, j] = len(action_history)

            pbar.update(1)
    
    return distances


def visualize_full_matrix(matrix: np.ndarray, states):
    """Display individual tetromino matrix."""
    # Plot full matrix
    fig = plt.figure(figsize=(12, 12))
    
    ax = plt.gca()
    
    # start_idx = 0
    # for tetromino, (matrix, n) in matrices.items():
    #     end_idx = start_idx + n
    #     matrix_viz[start_idx:end_idx, start_idx:end_idx] = matrix
        
    #     mid = start_idx + (end_idx - start_idx) // 2
    #     ax.text(-0.1, mid, tetromino, 
    #             horizontalalignment='right',
    #             verticalalignment='center',
    #             transform=ax.get_yaxis_transform(),
    #             fontsize=10,
    #             rotation=0)
        
    #     start_idx = end_idx

    heatmap = sns.heatmap(matrix, cmap='viridis',
                         xticklabels=False, yticklabels=False,
                         ax=ax, cbar_kws={'label': 'Distance (number of actions)'})
    
    plt.title(f'Full Distance Matrix (shape: {matrix.shape})')


def parse_args():
    parser = argparse.ArgumentParser(description='Compute distance matrix between Tetris states')
    parser.add_argument('--origin', type=str, help='Origin context trees directory')
    parser.add_argument('--maxsteps', type=int, default=10, help='Maximum number of steps')
    parser.add_argument('--radius', type=int, nargs=2, default=[3, 3], help='Radius of possible positions')
    parser.add_argument('--save', action='store_true', help='Save results to disk')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    return parser.parse_args()


def main():
    args = parse_args()
        
    actions = list(actions_from_nameset("restrained"))
    perception = next(iter(perceptions_from_nameset('vision_only')))

    positions = list(product(
            range(-args.radius[0], args.radius[0] + 1),
            range(-args.radius[1], args.radius[1] + 1),
    ))    
    states = list(product_dict(name=TETROMINO_NAMES, pos=positions, orientation=range(4)))
    n = len(TETROMINO_NAMES) * len(positions) * 4

    input_dir = Path('results') / args.origin
    agent = StandardAgent.load(
        dir_path=input_dir,
        perception=perception,
        verbose=args.verbose
    )
        
    distances = compute_distance_matrix(states, n, agent, args.maxsteps)
            
    finite_distances = distances[np.isfinite(distances)]    
    if len(finite_distances) > 0:
        print(f"\nStatistics:")
        print(f"Number of states: {n}")
        print(f"Average distance: {np.mean(finite_distances):.2f}")
        print(f"Maximum distance: {np.max(finite_distances):.2f}")
        print(f"Proportion of reachable states: {len(finite_distances)/(n**2):.2%}")
        
    if args.visualize:
        visualize_full_matrix(distances, states)
        plt.show()  # Show all windows at once

    if args.save:
        output_dir = dir_from_params(
        episodes=args.episodes,
        steps=args.steps
        )
        output_dir = f"results/distance_matrix/{output_dir}"
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/distances.npy", distances)
        np.save(f"{output_dir}/states.npy", states)
    

if __name__ == '__main__':
    main()