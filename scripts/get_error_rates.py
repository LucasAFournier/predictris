import argparse
import time
import csv
from tqdm import tqdm
import os
from pathlib import Path

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment
from predictris.utils import dir_from_params


LEARN_STEPS = 10
TEST_STEPS = 4
TEST_EPISODES = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tetris agent and collect prediction error rate data"
    )
    parser.add_argument(
        "--tetrominos",
        type=str,
        required=False,
        nargs="+",
        help="List of tetrominos",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Depth of prediction trees"
    )
    parser.add_argument(
        "--total-steps", type=int, help="Total number of steps to run"
    )

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--step", type=int, required=False, default=3)
    parser.add_argument(
        "--points",
        nargs="?",
        type=int,
        required=False,
        default=100,
        help="Number of measures for error rates. Default: 100 if flag is present.",
    )
    parser.add_argument(
        "--action-choice",
        type=str,
        required=False,
        default="random",
        choices=["from_active", "random"],
        help="Action choice method",
    )
    parser.add_argument(
        "--activation",
        type=str,
        required=False,
        default="all",
        choices=["by_confidence", "all"],
        help="Activation function",
    )
    parser.add_argument(
        "--reps",
        type=int,
        required=False,
        default=1,
        help="Number of repetitions",
    )

    return parser.parse_args()


def learn_episode(
    env: TetrisEnvironment,
    agent: Agent,
    tetrominos: list,
    action_choice: str,
    activation: str,
):
    """Run a single learning episode."""
    env.random_init(tetrominos)
    agent.init_episode(action_choice, activation, learn=True)

    for _ in range(LEARN_STEPS):
        agent.update(learn=True)


def test_episode(
    env: TetrisEnvironment,
    agent: Agent,
    tetrominos: list,
    action_choice: str,
    activation: str,
):
    """Run a single test episode and collect metrics by step depth."""
    env.random_init(tetrominos)
    agent.init_episode(action_choice, activation, test=True)

    for _ in range(TEST_STEPS):
        agent.update(test=True)


def collect_data(
    env: TetrisEnvironment, agent: Agent, interval: int, args, pbar
):
    """Collect prediction error rate data during learning."""
    total_steps = 0
    error_rates = []
    next_update = interval

    while total_steps < args.total_steps:
        learn_episode(
            env, agent, args.tetrominos, args.action_choice, args.activation
        )
        total_steps += LEARN_STEPS

        if total_steps >= next_update:
            for _ in range(TEST_EPISODES):
                test_episode(
                    env,
                    agent,
                    args.tetrominos,
                    args.action_choice,
                    args.activation,
                )

            result = agent.metrics["error_rate"].result()
            error_rates.append((total_steps, result))

            next_update += interval
            pbar.n = total_steps
            pbar.refresh()

    return error_rates


def save_error_rates(result, rep, dir):
    """Save error rate data to CSV file."""
    if not result:
        return

    file_path = Path(f"plots/{dir}/error-rates/{rep}.csv")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Get headers from the keys of the first dictionary
    first_data_dict = result[0][1]
    headers = ["total_steps"] + sorted(list(first_data_dict.keys()))

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for total_steps, rates_dict in result:
            row = [total_steps] + [rates_dict.get(h, "") for h in headers[1:]]
            writer.writerow(row)


def main():
    args = parse_args()
    if args.points is None:
        print("Please specify the --points flag to run the script.")
        return

    interval = args.total_steps // args.points
    error_rates_reps = []

    dir_name = dir_from_params(args)

    for rep in range(args.reps):
        env = TetrisEnvironment()
        agent = env.build_agent(
            depth=args.depth, verbose=args.verbose, metrics=["error_rate"]
        )

        with tqdm(
            total=args.total_steps,
            desc=f"[{rep + 1}/{args.reps}] {dir_name}",
            position=0,
            leave=args.verbose,
        ) as pbar:
            result = collect_data(env, agent, interval, args, pbar)

        if args.save:
            save_error_rates(result, rep, dir_name)

        error_rates_reps.append(result)

    if args.plot:
        from predictris.plot import plot_error_rates

        # Since total steps are the same for all reps, we can extract them from the first rep.
        total_steps_data = (
            [step for step, _ in error_rates_reps[0]]
            if error_rates_reps
            else []
        )
        # error_rates_data is now a list of lists of dictionaries.
        error_rates_data = [
            [error_dict[args.step] for _, error_dict in rep_data]
            for rep_data in error_rates_reps
        ]

        plot_error_rates(
            total_steps_data=[
                total_steps_data
            ],  # Wrap in a list for experiment grouping
            error_rates_data=[
                error_rates_data
            ],  # Wrap in a list for experiment grouping
            legends=[dir_name],
            output_path=Path(f"plots/{dir_name}/error-rates.png")
            if args.save
            else None,
        )


if __name__ == "__main__":
    main()