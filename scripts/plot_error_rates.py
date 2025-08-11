import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from predictris.plot import plot_error_rates


HLINE_Y = 0.94  # measured error rate of random predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot best prediction error rate from experiment data"
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment directories to compare",
    )
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output file path for saving the plot",
    )
    return parser.parse_args()


def extract_error_rates(
    directory: Path, step: int
) -> Tuple[List[List[float]], List[List[float]]]:
    """Extract steps and prediction error rates from CSV files for a specific step."""
    total_steps_hist = []
    error_rates_hist = []

    for file in list(directory.glob("*.csv")):
        with open(file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            total_steps_col_idx = header.index("total_steps")

            try:
                col_idx = header.index(str(step))
            except ValueError:
                # If the step is not in this file, skip it
                continue

            total_steps = []
            error_rates = []

            for row in reader:
                total_steps.append(float(row[total_steps_col_idx]))
                error_rates.append(float(row[col_idx]))

            total_steps_hist.append(total_steps)
            error_rates_hist.append(error_rates)

    return total_steps_hist, error_rates_hist


if __name__ == "__main__":
    args = parse_args()

    total_steps_data = []
    error_rates_data = []
    legends = []

    for exp_dir in args.dirs:
        total_steps_hist, error_rates_hist = extract_error_rates(
            Path("plots") / exp_dir / Path("error-rates"), args.step
        )
        # The plot function expects a single list of steps per experiment, not a list of lists.
        if total_steps_hist:
            total_steps_data.append(total_steps_hist[0])
        else:
            total_steps_data.append([])
        error_rates_data.append(error_rates_hist)
        legends.append(Path(exp_dir).name + f" step={args.step}")

    output_path = (
        Path("plots/error-rates") / args.output if args.output else None
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_error_rates(
        total_steps_data, error_rates_data, legends, HLINE_Y, output_path
    )
