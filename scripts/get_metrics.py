import argparse
import time
import csv
from tqdm import tqdm
import os
from pathlib import Path

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params
from predictris.plot import PlotMetrics, format_title


ALLOWED_METRICS = ['nodes_count', 'confidences', 'time_per_step']
LEARN_STEPS = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent and collect nodes count data')
    parser.add_argument('--tetrominos', type=str, required=True, nargs='+', help='List of tetrominos')
    parser.add_argument('--depth', type=int, help='Depth of prediction trees')
    parser.add_argument('--total-steps', type=int, help='Total number of steps to run')
    
    parser.add_argument('--nodes-count', nargs='?', type=int, const=100, default=None, help='Number of measures for nodes count. Default: 100 if flag is present.')
    parser.add_argument('--confidences', nargs='?', type=int, const=4, default=None, help='Number of measures for confidences. Default: 4 if flag is present.')
    parser.add_argument('--time-per-step', nargs='?', type=int, const=100, default=None, help='Number of measures for time per step. Default: 100 if flag is present.')
    
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    parser.add_argument('--action', type=str, required=False, default='random', choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, required=False, default='all', choices=['by_confidence', 'all'], help='Activation function')
    
    return parser.parse_args()


def learn_episode(env: TetrisEnvironment, agent: Agent, tetrominos: list, action: str, activation: str):
    """Run a single learning episode."""
    env.random_init(tetrominos)
    agent.init_learn_episode(action, activation)

    for _ in range(LEARN_STEPS):
        agent.learn()


def collect_metrics(env: TetrisEnvironment, agent: Agent, interval, args, pbar):
    """Collect nodes count data during learning."""
    metrics = {
        metric : [] for metric in interval
    }
    total_steps = 0
    next_update = {
        metric: 0 for metric in interval
    }
    while total_steps < args.total_steps:
        learn_episode(env, agent, args.tetrominos, args.action, args.activation)
        total_steps += LEARN_STEPS

        for metric in metrics:
            if total_steps >= next_update[metric]:
                metrics[metric].append((total_steps, agent.metrics[metric].result()))
                next_update[metric] += interval[metric]

            pbar.n = total_steps
            pbar.refresh()

    return metrics


def save_metrics(metrics, dir):
    """Save metrics to CSV file."""
    for metric, data in metrics.items():
        # argparse converts hyphens to underscores, so we convert them back for folder names
        folder = metric.replace('_', '-')
        file_path = Path(f"plots/{dir}/metrics/{folder}/0.csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['total_steps', 'value'])
            writer.writerows(data)


def main():
    args = parse_args()
    chosen_metrics = [metric for metric in ALLOWED_METRICS if getattr(args, metric) is not None]

    interval = {
        metric : args.total_steps // getattr(args, metric)
        for metric in chosen_metrics
    }

    dir_name = dir_from_params(
        tetrominos = ''.join(sorted(args.tetrominos)) if args.tetrominos else None,
        depth = args.depth,
    )

    env = TetrisEnvironment()
    agent = env.build_agent(depth=args.depth, verbose=args.verbose, metrics=chosen_metrics)

    with tqdm(total=args.total_steps, desc=f"{dir_name}", position=0, leave=args.verbose) as pbar:
        metrics = collect_metrics(env, agent, interval, args, pbar)

    if args.save:
        save_metrics(metrics, dir_name)

    if args.plot:
        data_series = [[metrics]]
        
        legends = [dir_name]
        title = dir_name
        
        output_path = Path(f'plots/{dir_name}/metrics.png') if args.save else None

        plotter = PlotMetrics(data_series, legends, title, output_path)
        plotter.plot(chosen_metrics)
        
if __name__ == '__main__':
    main()