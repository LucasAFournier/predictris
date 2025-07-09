import argparse
import time
import csv
from tqdm import tqdm
import os

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params


NUM_MEASUREMENTS = 100
CONFIDENCE_THRESHOLD = 0.95


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent and collect nodes count data')
    parser.add_argument('--dir', type=str, required=False, default=None,
                        help='Directory for existing prediction trees')
    parser.add_argument('--tetrominos', type=str, required=False, nargs='+', help='List of tetrominos')
    parser.add_argument('--depth', type=int, default=3, help='Depth of prediction trees')
    parser.add_argument('--episode', type=int, help='Maximum number of actions per episode')
    parser.add_argument('--action', type=str, choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, choices=['by_confidence', 'all'], help='Activation function')
    parser.add_argument('--steps', type=int, help='Total number of steps to run')
    parser.add_argument('--reps', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def run_episode(env: TetrisEnvironment, agent: Agent, tetrominos: list, depth: int, episode: int, action: str, activation: str):
    """Run a single training episode."""
    env.random_init(tetrominos)
    agent.init_learn(depth, action, activation, metrics="pred_success")

    start = time.time()
    for step in range(episode):
        agent.learn()

    return time.time() - start, agent.metrics['preds'], agent.metrics['correct_preds']


def collect_data(env: TetrisEnvironment, agent: Agent, args, pbar):
    """Collect nodes count data during training."""
    total_steps = 0
    steps_since_update = 0
    time_since_update = 0
    preds_since_update = 0
    correct_preds_since_update = 0

    steps = []
    time_per_step = []
    pred_success = []
    nodes = []
    filtered = []

    update_interval = args.steps / NUM_MEASUREMENTS
    next_update = update_interval

    while total_steps < args.steps:
        learn_time, preds, correct_preds = run_episode(env, agent, args.tetrominos, args.depth, args.episode,
                                                       args.action, args.activation)
        total_steps += args.episode
        steps_since_update += args.episode
        time_since_update += learn_time
        preds_since_update += preds
        correct_preds_since_update += correct_preds

        if total_steps >= next_update:
            steps.append(total_steps)
            time_per_step.append(round(time_since_update / steps_since_update, 4))
            pred_success.append(
                round(correct_preds_since_update / preds_since_update, 4)
                if preds_since_update != 0 else 0.0
            )
            nodes.append(agent.get_nodes_count())
            filtered.append(agent.get_nodes_count(filter=CONFIDENCE_THRESHOLD))
            
            next_update += update_interval
            steps_since_update = 0
            time_since_update = 0
            preds_since_update = 0
            correct_preds_since_update = 0
            pbar.n = total_steps
            pbar.set_postfix({'nodes': nodes[-1], 'filtered': filtered[-1]})
            pbar.refresh()

    return steps, time_per_step, pred_success, nodes, filtered


def save_history(rows, rep, dir):
    """Save history data to CSV file."""    
    file = f"plots/{dir}/{rep}.csv"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps', 'time_per_step', 'pred_success', 'nodes', 'filtered_nodes'])
        writer.writerows(rows)


def main():
    args = parse_args()
    # Initialize histories as lists of lists
    steps_hist = []
    time_per_step_hist = []
    pred_success_hist = []
    nodes_hist = []
    filtered_hist = []

    dir = dir_from_params(
        dir = f'({args.dir})' if args.dir else None,
        tetrominos = ''.join(args.tetrominos) if args.tetrominos else None,
        depth = args.depth,
        episode = args.episode,
        action = args.action.replace('_', ''),
        activation = args.activation.replace('_', ''),
    )

    for rep in range(args.reps):
        env = TetrisEnvironment()
        agent = env.build_agent(depth=args.depth, dir=args.dir, verbose=args.verbose)

        with tqdm(total=args.steps, desc=f"[{rep + 1}/{args.reps}] {dir}", position=0, leave=args.verbose) as pbar:
            steps, time_per_step, pred_success, nodes, filtered = collect_data(env, agent, args, pbar)

        if args.save:
            save_history(zip(steps, time_per_step, pred_success, nodes, filtered), rep, dir)
            
        # Append each run's data to respective histories
        steps_hist.append(steps)
        time_per_step_hist.append(time_per_step)
        pred_success_hist.append(pred_success)
        nodes_hist.append(nodes)
        filtered_hist.append(filtered)

    if args.plot:
        from predictris.plot import plot_nodes_data
        histories = [(steps_hist, time_per_step_hist, pred_success_hist, nodes_hist, filtered_hist)]
        plot_nodes_data(histories, dir, 
                       [])


if __name__ == '__main__':
    main()