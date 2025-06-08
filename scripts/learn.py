import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params


REFRESH_RATE = 0.5
RATE_THRESHOLD = 0.1
TIMEOUT = 10.0


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent')
    parser.add_argument('--context', type=int)
    parser.add_argument('--nodes', type=int, nargs='+',
                        help='List of total number of nodes (e.g. 10000 20000 50000)')
    parser.add_argument('--steps', type=str, help='Number of actions per episode or "auto"')
    parser.add_argument('--choice', type=str, choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, choices=['by_confidence', 'all'], help='Activation function')
    parser.add_argument('--dir', type=str, required=False, default=None,
                        help='Directory for existing prediction trees')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


def run_episode(env: TetrisEnvironment, agent: Agent, steps: int, context_size: int, action_choice: str, activation: str):
    """Run a single training episode."""
    env.random_init()
    result = agent.init_learn(context_size, action_choice, activation)

    step = 0
    while result != 'abort' and step < steps:
        result = agent.learn()
        step += 1


def plot_nodes_count(times_history, nodes_count_history):
    plt.figure(figsize=(10, 6))
    plt.plot(times_history, nodes_count_history)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Nodes Count')
    plt.title('Nodes Count over Time')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def main():
    args = parse_args()
    
    env = TetrisEnvironment()
    agent = env.build_agent(dir=args.dir, verbose=args.verbose)

    nodes_count = agent.get_nodes_count()
    checkpoints = sorted(args.nodes)
    total = max(checkpoints)

    start_time = time.time()
    last_update = start_time
    last_nodes_update = start_time

    steps_auto = (args.steps == "auto")
    current_steps = 2 if steps_auto else int(args.steps)
    
    base_desc = "Learning"
    pbar = tqdm(initial=nodes_count, total=total, leave=False, desc=base_desc)
    
    if args.plot:
        nodes_count_history = []
        times_history = []
        
    while nodes_count < total:
        run_episode(env, agent, current_steps, args.context, args.choice, args.activation)
        
        now = time.time()
        if now - last_update >= REFRESH_RATE:
            added = agent.get_nodes_count() - nodes_count
            nodes_count += added
            
            if added > 0:
                last_nodes_update = now
            elif now - last_nodes_update >= TIMEOUT:
                print(f"\nNo new nodes added for {TIMEOUT} seconds. Stopping...")
                if args.save and checkpoints:
                    params = {
                        'dir': f'({args.dir})' if args.dir else None,
                        'context': args.context,
                        'steps': args.steps,
                        'choice': args.choice,
                        'activation': args.activation,
                        'nodes': 'final',
                    }
                    output_dir = dir_from_params(**params)
                    agent.save(output_dir, verbose=args.verbose)
                break

            rate = added / (now - last_update)
            desc = f"{base_desc} - {rate:>3.0f} nodes/s - {current_steps} steps"
            pbar.set_description(desc)
            
            if steps_auto and rate < RATE_THRESHOLD:
                current_steps += 1
            
            if args.plot:
                nodes_count_history.append(nodes_count)
                times_history.append(now - start_time)
            
            if args.save and nodes_count >= checkpoints[0]:
                params = {
                        'dir': f'({args.dir})' if args.dir else None,
                        'context': args.context,
                        'steps': args.steps,
                        'choice': args.choice,
                        'activation': args.activation,
                        'nodes': checkpoints[0],
                    }
                output_dir = dir_from_params(**params)
                agent.save(output_dir, verbose=args.verbose)
                                
                checkpoints.pop(0)
            
            last_update = now                
            pbar.update(added)
    pbar.close()
            
    if args.plot:
        plot_nodes_count(times_history, nodes_count_history)


if __name__ == '__main__':
    main()