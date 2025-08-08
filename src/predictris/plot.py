import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path

def extract_parameters(config_string: str) -> Dict[str, str]:
    """Extract parameters and their values from config string."""
    return dict(param.split('=', 1) for param in config_string.split('_') if '=' in param)

def group_parameters(configs: List[str]) -> tuple[Dict[str, str], Dict[str, bool]]:
    """Group parameters into common and varying."""
    all_params = [extract_parameters(config) for config in configs]
    all_keys = set.union(*[set(p.keys()) for p in all_params])
    
    common, varying = {}, {}
    for key in all_keys:
        values = {p.get(key) for p in all_params if key in p}
        if len(values) == 1:
            common[key] = values.pop()
        else:
            varying[key] = True
    return common, varying

def format_title(common_params: Dict[str, str]) -> str:
    """Format common parameters for title."""
    return ', '.join(f"{k}={v}" for k, v in common_params.items())

def format_legend(config: str, varying_params: Dict[str, bool]) -> str:
    """Format varying parameters for legend."""
    params = extract_parameters(config)
    return ', '.join(f"{k}={params[k]}" for k in varying_params if k in params)

class PlotMetrics:
    """Handles plotting of different metrics from experiment data."""
    def __init__(self, data_series: List[Dict], legends: List[str], title: str, output_path: Path = None):
        self.data_series = data_series
        self.legends = legends
        self.title = title
        self.output_path = output_path
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(data_series))) if data_series else []

        self.metric_plotters = {
            'nodes_count': self._plot_mean_over_steps,
            'time_per_step': self._plot_mean_over_steps,
            'confidences': self._plot_distribution_over_steps,
        }
        self.metric_configs = {
            'nodes_count': {'ylabel': 'Total Nodes Count', 'ylim_bottom': 0},
            'time_per_step': {'ylabel': 'Time per Step (s)', 'ylim_bottom': 0},
            'confidences': {'ylabel': 'Node Confidence', 'ylim': (0, 1)},
        }

    def plot(self, metrics_to_plot: List[str]):
        """Create a grid plot for the specified metrics."""
        num_metrics = len(metrics_to_plot)
            
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle(self.title)

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            if metric in self.metric_plotters:
                self.metric_plotters[metric](ax, metric, **self.metric_configs.get(metric, {}))
            if i < num_metrics - 1:
                ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel('Total Steps')
        axes[-1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        
        if self.legends:
            axes[0].legend(loc='best')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.output_path)
        plt.show()

    def _plot_mean_over_steps(self, ax: plt.Axes, metric: str, ylabel: str = '', ylim_bottom: float = None, **kwargs):
        """Plots a metric over steps for each experiment."""
        for i, run_data in enumerate(self.data_series):
            total_steps = np.array([item[0] for item in run_data[metric]])
            values = np.array([item[1] for item in run_data[metric]])

            ax.plot(total_steps, values, color=self.colors[i], label=self.legends[i])

        ax.set_ylabel(ylabel)
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)

    def _plot_distribution_over_steps(self, ax: plt.Axes, metric: str, ylabel: str = '', ylim: tuple = None, **kwargs):
        """Plots the distribution of a metric over steps for each experiment."""
        ref_data = next((run_data.get(metric) for run_data in self.data_series if run_data.get(metric)), None)
        
        total_steps = np.array([item[0] for item in ref_data])
        step_delta = np.diff(total_steps).mean() if len(total_steps) > 1 else 1.0

        num_series = len(self.data_series)
        violin_width = step_delta / (num_series + 1)
        start_offset = -violin_width * (num_series - 1) / 2.0

        for i, run_data in enumerate(self.data_series):
            data = run_data.get(metric)[1:-1]  # Skip first and last entries

            current_steps = np.array([item[0] for item in data])
            values = [item[1] for item in data]

            positions = current_steps + start_offset + i * violin_width
            parts = ax.violinplot(values, positions=positions, widths=violin_width, showmeans=True, showextrema=False, points=200)
            for pc in parts['bodies']:
                pc.set_facecolor(self.colors[i])
                pc.set_alpha(0.7)
            ax.plot([], [], color=self.colors[i], label=self.legends[i], linewidth=5)

        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)

def plot_error_rates(total_steps_data, error_rates_data, legends, hline_y: float = None, output_path: str = None):
    """Create plot of best prediction error rates vs total steps."""
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(total_steps_data)))

    max_x = 0
    for (total_steps, all_reps_error_rates), color, legend in zip(zip(total_steps_data, error_rates_data), colors, legends):
        params = extract_parameters(legend)
        label = f"depth={params.get('depth', '?')}"
        
        y = np.array(all_reps_error_rates)
        mean = np.nanmean(y, axis=0)
        std = np.nanstd(y, axis=0)

        ax.plot(total_steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(total_steps, mean - std, mean + std, color=color, alpha=0.2)
        if len(total_steps) > 0:
            max_x = max(max_x, total_steps[-1])

    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Prediction Success Rate')
    ax.set_ylim(0, 1)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    if hline_y is not None:
        ax.axhline(y=hline_y, color='gray', linewidth=3, linestyle='--')

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
    
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()