import argparse
import csv
import numpy as np
from pathlib import Path
import ast

from predictris.plot import (
    PlotMetrics,
    group_parameters,
    format_legend,
    format_title,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot nodes count data from CSV files')
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                       help='List of experiment directories to compare')
    parser.add_argument('--output', type=str, required=False, help='Output file path for saving the plot')
    parser.add_argument('--metrics', nargs='+', default=['nodes_count', 'confidences', 'time_per_step'],
                        help='Metrics to plot')
    return parser.parse_args()


def load_experiment_data(args):
    """Load and preprocess experiment data from all directories."""
    experiment_names = [Path(d).name for d in args.dirs]
    common_params, varying_params = group_parameters(experiment_names)
    
    data_series = []
    labels = []
    
    for experiment_dir in args.dirs:
        data = load_csv_files(Path('plots') / experiment_dir / Path('metrics'))
        if not data:
            print(f"Warning: No data found in {experiment_dir}, skipping.")
            continue
        data_series.append(data)
        labels.append(format_legend(Path(experiment_dir).name, varying_params))
    
    return data_series, common_params, labels


def load_csv_files(directory: Path) -> dict:
    """Load all CSV files from the subdirectories of the given directory."""
    metrics = {}
    
    # Find all metric subdirectories
    metric_dirs = [d for d in directory.iterdir() if d.is_dir()]
    
    file_name = "0.csv"

    for metric_dir in metric_dirs:
        metric_name = metric_dir.name.replace('-', '_')
        file_path = metric_dir / file_name

        if file_path.exists():
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) # skip header
                metric_data = []
                for row in reader:
                    total_steps = ast.literal_eval(row[0])
                    value = ast.literal_eval(row[1])
                    metric_data.append((total_steps, value))
                metrics[metric_name] = metric_data
                
    return metrics


if __name__ == '__main__':
    args = parse_args()
    data_series, common_params, labels = load_experiment_data(args)
    
    if not data_series:
        print("No data to plot.")
    else:
        output_path = Path('plots/metrics') / args.output if args.output else None
        plotter = PlotMetrics(data_series, labels, format_title(common_params), output_path)
        plotter.plot(args.metrics)