import argparse
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional

from predictris.tetris import TetrisEnvironment, Tetromino
from predictris.agent import StandardAgent
from predictris.learning import ContextTrees
from predictris.tetris.constants import TETROMINO_NAMES, BACKGROUND_VALUE
from predictris.tetris.encoders import actions_from_nameset
from predictris.agent.body import ViewConfig
from predictris.utils import create_random_environment


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Tetris agent')
    parser.add_argument('--context-trees', type=str, required=True,
                       help='Path to saved context trees file')
    parser.add_argument('--grid-shape', type=int, nargs=2, required=True)
    parser.add_argument('--view-radius', type=int, nargs=2, default=[1, 1])
    parser.add_argument('--action-nameset', type=str, required=True)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=10)
    return parser.parse_args()


def run_episode(agent: StandardAgent, max_steps: int) -> dict:
    """Run a single evaluation episode."""
    steps = 0
    failed_predictions = 0
    
    # Initialize test episode
    agent.init_test()
    
    while steps < max_steps:
        # Run test step
        result = agent.test()
        steps += 1
        
        if result == 'end':
            break
        elif result == 'failed_prediction':
            failed_predictions += 1
        elif result == 'failed':
            break
    
    return {
        'steps': steps,
        'failed_predictions': failed_predictions,
    }


def main():
    args = parse_args()
    
    # Load context trees
    context_trees = ContextTrees.load(args.context_trees)
    
    # Run evaluation episodes
    results = []
    for episode in tqdm(range(args.episodes)):
        # Create random environment
        env, body = create_random_environment(
            random.choice(TETROMINO_NAMES),
            args.grid_shape,
            args.action_nameset,
            args.view_radius
        )
        
        # Create agent
        agent = StandardAgent(body, context_trees)
        
        # Run episode
        episode_results = run_episode(agent, args.max_steps)
        results.append(episode_results)
    
    # Print summary
    avg_steps = sum(r['steps'] for r in results) / len(results)
    avg_failures = sum(r['failed_predictions'] for r in results) / len(results)
    
    print(f"\nEvaluation Results ({args.episodes} episodes):")
    print(f"Average steps per episode: {avg_steps:.2f}")
    print(f"Average failed predictions: {avg_failures:.2f}")


if __name__ == '__main__':
    main()