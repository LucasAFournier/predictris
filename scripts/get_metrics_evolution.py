import argparse
import time
import csv
from tqdm import tqdm
import os

from predictris.agent import Agent, PredMetric, BestPredErrorRate, TimePerStepMetric
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params


NUM_MEASUREMENTS = 100
CONFIDENCE_THRESHOLD = 0.95
TEST_EPISODE_LENGTH = 10


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


def train_episode(env: TetrisEnvironment, agent: Agent, tetrominos: list, episode: int, action: str, activation: str):
    """Run a single training episode."""
    env.random_init(tetrominos)
    agent.init_episode(action, activation)

    for _ in range(episode):
        agent.update(learn=True)


def test_episode(env: TetrisEnvironment, agent: Agent, tetrominos: list):
    """Run a single test episode and collect metrics by step depth."""
    env.random_init(tetrominos)
    agent.init_episode(action_choice="random", activation="all")
 
    for _ in range(TEST_EPISODE_LENGTH):
        agent.update(test=True)


def collect_data(env: TetrisEnvironment, agent: Agent, args, pbar):
    """Collect nodes count data during training."""
    total_steps = 0

    steps = []
    time_per_step = []
    nodes = []
    filtered = []

    update_interval = args.steps / NUM_MEASUREMENTS
    next_update = update_interval

    while total_steps < args.steps:
        train_episode(env, agent, args.tetrominos, args.episode,
                      args.action, args.activation)
        total_steps += args.episode

        if total_steps >= next_update:
            steps.append(total_steps)
            
            time_per_step.append(agent.metrics[TimePerStepMetric].result())
            test_episode(env, agent, args.tetrominos)
            agent.metrics[TimePerStepMetric].reset()

            nodes.append(agent.get_nodes_count())
            filtered.append(agent.get_nodes_count(filter=CONFIDENCE_THRESHOLD))
            
            next_update += update_interval
            pbar.n = total_steps
            pbar.set_postfix({'nodes': nodes[-1], 'filtered': filtered[-1]})
            pbar.refresh()

    results = {
        'steps': steps,
        'time_per_step': time_per_step,
        'best_pred_error_rate': agent.metrics[BestPredErrorRate].result(),
        'nodes': nodes,
        'filtered': filtered
    }
    agent.metrics[BestPredErrorRate].reset()

    return results


def save_history(result, rep, dir):
    """Save history data to CSV file."""    
    file = f"plots/{dir}/{rep}.csv"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result.keys())
        writer.writerows(zip(*result.values()))


def main():
    args = parse_args()
    # Initialize histories as lists of lists
    steps_hist = []
    time_per_step_hist = []
    best_pred_error_rate_hist = []
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
        agent = env.build_agent(depth=args.depth, dir=args.dir, verbose=args.verbose, metrics=["best_pred", "time_per_step"])

        with tqdm(total=args.steps, desc=f"[{rep + 1}/{args.reps}] {dir}", position=0, leave=args.verbose) as pbar:
            result = collect_data(env, agent, args, pbar)

        if args.save:
            save_history(result, rep, dir)

        # Append each run's data to respective histories
        steps_hist.append(result['steps'])
        time_per_step_hist.append(result['time_per_step'])
        best_pred_error_rate_hist.append(result['best_pred_error_rate'])
        nodes_hist.append(result['nodes'])
        filtered_hist.append(result['filtered'])

    if args.plot:
        from predictris.plot import plot_nodes_data
        histories = [(steps_hist, time_per_step_hist, best_pred_error_rate_hist, nodes_hist, filtered_hist)]
        plot_nodes_data(histories, dir,
                       [])


if __name__ == '__main__':
    main()