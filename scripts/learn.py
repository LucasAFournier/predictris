import argparse
import time
from tqdm import tqdm

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params


REFRESH_RATE = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent')
    parser.add_argument('--dir', type=str, required=False, default=None,
                        help='Directory for existing prediction trees')
    parser.add_argument('--tetrominos', type=str, required=False, nargs='+', help='List of tetrominos')
    parser.add_argument('--episode', type=int, help='Maximum number of actions per episode')
    parser.add_argument('--action', type=str, choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, choices=['by_confidence', 'all'], help='Activation function')
    parser.add_argument('--steps', type=int, help='Total number of steps to run')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def run_episode(env: TetrisEnvironment, agent: Agent, episode: int, action_choice: str, activation: str, tetrominos: list):
    """Run a single training episode."""
    result = 'abort'
    while result == 'abort':
        env.random_init(tetrominos)
        result = agent.init_learn(action_choice, activation)

    step = 1
    while result != 'abort' and step < episode:
        result = agent.learn()
        step += 1

    return step

def main():
    args = parse_args()
    
    dir = dir_from_params(
        dir = f'({args.dir})' if args.dir else None,
        tetrominos = ''.join(args.tetrominos) if args.tetrominos else None,
        episode = args.episode,
        action = args.action.replace('_', ''),
        activation = args.activation.replace('_', ''),
    )

    env = TetrisEnvironment()
    agent = env.build_agent(dir=args.dir, verbose=args.verbose)
    
    total_steps = 0
    last_update = time.time()

    with tqdm(total=args.steps, desc="Steps progress", position=0, leave=False) as pbar:
        while total_steps < args.steps:
            total_steps += run_episode(env, agent, args.episode,
                                       args.action, args.activation, args.tetrominos)
            
            if time.time() - last_update > REFRESH_RATE:
                pbar.n = total_steps
                pbar.set_postfix({'nodes': agent.get_nodes_count()})
                pbar.refresh()
                last_update = time.time()

    if args.save:
        agent.save(dir, verbose=args.verbose)
        
            
if __name__ == '__main__':
    main()