import argparse
import time
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

from predictris.agent import Agent, PathsMetric
from predictris.tetris import TetrisEnvironment


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory for existing prediction trees')
    # parser.add_argument('--output', type=str)
    parser.add_argument('--tetrominos', type=str, required=False, nargs='+', help='List of tetrominos')
    parser.add_argument('--depth', type=int, default=3, help='Depth of prediction trees')
    parser.add_argument('--episode', type=int, help='Maximum number of actions per episode')
    parser.add_argument('--action', type=str, choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, choices=['by_confidence', 'all'], help='Activation function')
    parser.add_argument('--steps', type=int, help='Total number of steps to run')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def run_episode(env: TetrisEnvironment, agent: Agent, episode: int, action_choice: str, activation: str, tetrominos: list):
    """Run a single training episode."""
    env.random_init(tetrominos)
    # Initialize with 1 priming step
    agent.init_learn(priming_steps=1, action_choice=action_choice, activation=activation, metrics='paths')
    
    for i in range(episode):
        agent.update(learn=True)
        
    return agent.metrics[PathsMetric].paths

def calculate_path_overlap(path1, path2):
    """Calculate the length of overlap between two paths.
    Each path is [start, end] where start and end are integers representing steps.
    Returns the number of overlapping steps."""
    start1, end1 = path1
    start2, end2 = path2
    
    # Check if ranges overlap
    if max(start1, start2) <= min(end1, end2):
        # Calculate overlap length
        return min(end1, end2) - max(start1, start2) + 1
    return 0

def get_episode_max_overlap(paths):
    """Find maximum overlap between any two paths in the episode."""
    if len(paths) < 2:
        return 0
        
    max_overlap = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            overlap = calculate_path_overlap(paths[i], paths[j])
            max_overlap = max(max_overlap, overlap)
    return max_overlap

def plot_overlap_distribution(overlaps, figsize=(10, 6)):
    """Plot the normalized distribution of maximum path overlaps."""
    plt.figure(figsize=figsize)
    
    # Calculate normalized histogram
    counts, edges = np.histogram(overlaps, bins=np.arange(min(overlaps), max(overlaps) + 2) - 0.5)
    counts = counts / len(overlaps)  # Normalize
    
    # Plot bars centered on integer values
    centers = (edges[:-1] + edges[1:]) / 2
    plt.bar(centers, counts, width=1, edgecolor='none', align='center')
    
    plt.title("Distribution of Maximum Path Overlaps per Episode")
    plt.xlabel("Maximum Overlap Length")
    plt.ylabel("Fraction of Episodes")
    plt.show()

def main():
    args = parse_args()

    env = TetrisEnvironment()
    agent = env.build_agent(depth=args.depth, dir=Path("results", args.input), verbose=args.verbose)

    total_steps = 0
    episode_overlaps = []

    with tqdm(total=args.steps, desc="Steps progress", position=0, leave=False) as pbar:
        while total_steps < args.steps:
            paths = run_episode(env, agent, args.episode,
                                    args.action, args.activation, args.tetrominos)
            total_steps += args.episode
            
            # Calculate maximum overlap for this episode's completed paths
            max_overlap = get_episode_max_overlap(paths)
            episode_overlaps.append(max_overlap)

            pbar.n = total_steps
            pbar.refresh()
    
    # Plot overlap distribution
    plot_overlap_distribution(episode_overlaps)

if __name__ == '__main__':
    main()