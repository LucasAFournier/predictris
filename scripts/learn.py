import argparse
import time
from tqdm import tqdm

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params


REFRESH_RATE = 0.5
LEARN_STEPS = 10

def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent')
    parser.add_argument('--tetrominos', type=str, required=True, nargs='+', help='List of tetrominos')
    parser.add_argument('--depth', type=int, default=3, help='Depth of prediction trees')
    parser.add_argument('--total-steps', type=int, required=True, help='Total number of steps to run')
    
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--action', type=str, required=False, default='random', choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, required=False, default='all', choices=['by_confidence', 'all'], help='Activation function')
    
    return parser.parse_args()


def learn_episode(env: TetrisEnvironment, agent: Agent, tetrominos: list, action: str, activation: str):
    """Run a single training episode."""
    env.random_init(tetrominos)
    agent.init_learn_episode(action, activation)

    for _ in range(LEARN_STEPS):
        agent.learn()


def main():
    args = parse_args()

    dir_name = dir_from_params(args)

    env = TetrisEnvironment()
    agent = env.build_agent(depth=args.depth, verbose=args.verbose)
    
    total_steps = 0
    last_update = time.time()

    with tqdm(total=args.total_steps, desc=f"Training on {dir_name}", position=0, leave=True) as pbar:
        while total_steps < args.total_steps:
            learn_episode(env, agent, args.tetrominos, args.action, args.activation)
            total_steps += LEARN_STEPS
            
            if time.time() - last_update > REFRESH_RATE:
                pbar.n = total_steps
                pbar.refresh()
                last_update = time.time()
        
        pbar.n = args.total_steps
        pbar.refresh()


    if args.save:
        agent.save(dir_name, verbose=args.verbose)
        
            
if __name__ == '__main__':
    main()