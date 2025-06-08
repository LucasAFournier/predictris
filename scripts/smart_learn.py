import argparse
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional
import time  # Add this import at the top
import matplotlib.pyplot as plt  # Add this import

from predictris.tetris import TetrisEnvironment, Tetromino
from predictris.agent import StandardAgent
from predictris.learning import PredictionTree
from predictris.tetris.constants import TETROMINO_NAMES, BACKGROUND_VALUE
from predictris.tetris.encoders import actions_from_nameset, perceptions_from_nameset  # Add this import
from predictris.utils import create_random_environment, dir_from_params


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent using heuristic')
    parser.add_argument('--origin', type=str, required=False, help='Optional: Directory containing trees to load')
    parser.add_argument('--context', type=int, help='Context size for smart learning')
    parser.add_argument('--nodes', type=int, nargs='+', help='List of total number of nodes')
    parser.add_argument('--steps', type=int, help='Number of smart actions per episode')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def run_episode(agent: StandardAgent, smart_steps: int, context_size: int) -> tuple[str, int]:
    """Run a single training episode with optional auto-stop mechanism."""
    result = agent.init_smart_learn(context_size)
    if result == 'aborted':
        return 0
    added_nodes = 0
    for _ in range(smart_steps):
        result = agent.smart_learn()
        if result == 'aborted':
            break
        added_nodes += result
    return added_nodes


def plot_nodes(times, added_nodes):
    plt.figure(figsize=(10, 6))
    plt.plot(times, added_nodes)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Nodes')
    plt.title('Number of Nodes over Time')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def main():
    args = parse_args()
    perception = next(iter(perceptions_from_nameset('vision_only')))
    
    # Sort epoch checkpoints and get maximum
    nodes_checkpoints = sorted(args.nodes)
    max_nodes = max(nodes_checkpoints)
    
    # Initialize parameters based on mode
    if args.origin:
        input_dir = Path('results') / args.origin
        agent = StandardAgent.load(
            dir_path=input_dir,
            perception=perception,
            verbose=args.verbose
        )
    else:
        agent = StandardAgent(
            perception=perception,
            verbose=args.verbose
        )
            
    base_desc = f"Smart learning"
    pbar = tqdm(total=max_nodes, leave=False, desc=base_desc)
    
    # Initialize tracking variables 
    last_update = time.time()
    start_time = time.time()
    
    # For plotting
    nodes_history = []
    times_history = []
    
    total_nodes = 0
    recent_nodes = 0
    while total_nodes < max_nodes:
        env = create_random_environment()
        agent.body = env.get_body()
        
        added_nodes = run_episode(
            agent, 
            args.steps, 
            args.context,
        )

        total_nodes += added_nodes        
        recent_nodes += added_nodes

        now = time.time()
        if now - last_update >= 1.0:
            rate = recent_nodes / (now - last_update)            
            if args.verbose:
                nodes_history.append(total_nodes)
                times_history.append(now - start_time)
            desc = f"{base_desc} - {args.steps:>2.0f} steps - {rate:>3.0f} nodes/s"
            pbar.set_description(desc)
            
            last_update = now
            recent_nodes = 0
            
        # Save at checkpoints if --save is enabled
        if args.save and total_nodes >= nodes_checkpoints[0]:
            if args.origin:
                output_dir = dir_from_params(
                    origin=f'({args.origin})',
                    context=args.context,
                    nodes=nodes_checkpoints[0],
                    steps=args.steps,
                )
            else:
                output_dir = dir_from_params(
                    context=args.context,
                    nodes=nodes_checkpoints[0],
                    steps=args.steps,
                )

            if args.verbose:
                print(f"\nSaving trees to {output_dir}")
            for tree in agent.tree_from_pred_obs.values():
                tree.save(output_dir)

            nodes_checkpoints.pop(0)
                
        pbar.update(added_nodes)

    pbar.close()
            
    if args.verbose and nodes_history:
        plot_nodes(times_history, nodes_history)


if __name__ == '__main__':
    main()