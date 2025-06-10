import argparse
import csv
import numpy as np
from pathlib import Path

from predictris.plot import plot_nodes_data


def parse_args():
    parser = argparse.ArgumentParser(description='Plot nodes count data from CSV files')
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                       help='List of experiment directories to compare')
    parser.add_argument('--output', type=str, help='Output file path for saving the plot')
    return parser.parse_args()


def extract_parameters(config_string):
    """Extract parameters and their values from config string."""
    params = {}
    for param in config_string.split('_'):
        if '=' in param:
            key, value = param.split('=')
            params[key] = value
    return params


def group_parameters(configs):
    """Group parameters into common and varying."""
    all_params = [extract_parameters(config) for config in configs]
    
    common = {}
    varying = {}
    
    # Get all parameter keys
    all_keys = set().union(*(params.keys() for params in all_params))
    
    # Check each parameter
    for key in all_keys:
        values = [params.get(key) for params in all_params]
        values = [v for v in values if v is not None]  # Remove None values
        
        if len(set(values)) == 1:  # If all values are the same
            common[key] = values[0]
        else:
            varying[key] = True
            
    return common, varying


def format_title(common_params):
    """Format common parameters for title."""
    parts = [f"{k}={v}" for k, v in common_params.items()]
    return ', '.join(parts)


def format_legend(config, varying_params):
    """Format varying parameters for legend."""
    params = extract_parameters(config)
    parts = [f"{k}={params[k]}" for k in varying_params if k in params]
    return ', '.join(parts)


def load_experiment_data(args):
    """Load and preprocess experiment data from all directories."""
    experiment_names = [Path(d).name for d in args.dirs]
    common_params, varying_params = group_parameters(experiment_names)
    
    data_series = []
    labels = []
    
    for experiment_dir in args.dirs:
        data = load_csv_files(Path('plots') / experiment_dir)
        if not data[0]:  # Check if steps_hist is empty
            raise ValueError(f"No data found in {experiment_dir}")
        data_series.append(data)
        labels.append(format_legend(Path(experiment_dir).name, varying_params))
    
    return data_series, common_params, labels


def mask_time_data(experiment_data: list) -> list:
    """Interpolate all experiment data to common time points."""
    # Find shortest experiment duration
    min_end_time = min(
        max(max(times) for times in data[1])  # data[1] is time_hist
        for data in experiment_data
    )
    
    # Create common time points for interpolation
    common_time_points = np.linspace(0, min_end_time, 100)
    
    processed_data = []
    for steps_hist, time_hist, wrong_preds_hist, nodes_hist, filtered_hist in experiment_data:
        interpolated_nodes = []
        interpolated_filtered = []
        
        # Interpolate each run in the experiment
        for times, wrong_preds, nodes, filtered in zip(time_hist, wrong_preds_hist, nodes_hist, filtered_hist):
            times, wrong_preds, nodes, filtered = map(np.array, [times, wrong_preds, nodes, filtered])
            mask = times <= min_end_time
            
            interpolated_nodes.append(np.interp(common_time_points, times[mask], wrong_preds[mask], nodes[mask]))
            interpolated_filtered.append(np.interp(common_time_points, times[mask], wrong_preds[mask], filtered[mask]))
        
        processed_data.append((
            steps_hist,
            common_time_points,
            wrong_preds_hist,
            nodes_hist,
            filtered_hist,
            interpolated_nodes,
            interpolated_filtered
        ))
    
    return processed_data


def load_csv_files(directory: Path) -> tuple[list, list, list, list]:
    """Load all CSV files from the directory."""
    steps_hist = []
    time_hist = []
    wrong_preds_hist = []
    nodes_hist = []
    filtered_hist = []
    
    for file in list(directory.glob("*.csv")):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            steps = []
            times = []
            wrong_preds = []
            nodes = []
            filtered = []
            
            for row in reader:
                s, t, w, n, f = map(float, row)
                steps.append(s)
                times.append(t)
                wrong_preds.append(w)
                nodes.append(int(n))
                filtered.append(int(f))
            
            if steps:
                steps_hist.append(steps)
                time_hist.append(times)
                wrong_preds_hist.append(wrong_preds)
                nodes_hist.append(nodes)
                filtered_hist.append(filtered)
    
    return steps_hist, time_hist, wrong_preds_hist, nodes_hist, filtered_hist


if __name__ == '__main__':
    args = parse_args()
    data_series, common_params, labels = load_experiment_data(args)
    output_path = Path('plots') / args.output if args.output else None
    plot_nodes_data(data_series, format_title(common_params), labels, output_path)