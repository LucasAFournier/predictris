import numpy as np
import matplotlib.pyplot as plt

def plot_nodes_data(histories, title, legends, output_path=None):
    """Create a 3x1 grid plot showing nodes counts and time per step vs steps."""
    fig = plt.figure(figsize=(10, 24))
    gs = fig.add_gridspec(4, 1, hspace=0.05)
    axes = [
        fig.add_subplot(gs[0]),  # nodes vs steps
        fig.add_subplot(gs[1]),  # filtered nodes vs steps
        fig.add_subplot(gs[2]),  # time per step vs steps
        fig.add_subplot(gs[3]),  # prediction success rate vs steps
    ]
    
    # Share x axis
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    axes[3].sharex(axes[0])
    
    # Hide top plots' x labels
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].tick_params(labelbottom=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    fig.suptitle(title)
    
    for i, (hist, color) in enumerate(zip(histories, colors)):
        steps, time_per_step, pred_success, nodes, filtered = hist
        legend = legends[i] if legends else None

        # Plot steps data
        max_steps = max(max(x) for x in steps)
        steps_x = np.linspace(0, max_steps, 100)
        
        plot_means(axes[0], steps_x, nodes, steps, color, legend, bounds=True)
        plot_means(axes[1], steps_x, filtered, steps, color, legend, bounds=True)
        plot_means(axes[2], steps_x, time_per_step, steps, color, legend)
        plot_means(axes[3], steps_x, pred_success, steps, color, legend)
    
    axes[3].set_xlabel('Steps')
    axes[3].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    axes[0].set_ylabel('Total Nodes Count')
    axes[1].set_ylabel('Filtered Nodes Count')
    axes[2].set_ylabel('Time per Step (s)')
    axes[3].set_ylabel('Prediction Success Rate')
    
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    axes[3].set_ylim(0, 1)

    if legends:
        axes[3].legend(loc='lower right')
    
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_means(ax, x, y_data, x_data=None, color=None, label=None, bounds=False):
    """Plot mean with min/max bounds for given data."""
    if x_data is not None:
        y_interpolated = [np.interp(x, x_src, y) for x_src, y in zip(x_data, y_data)]
    else:
        y_interpolated = y_data
        
    y_array = np.array(y_interpolated)
    mean = np.mean(y_array, axis=0)
    
    ax.plot(x, mean, '-', color=color, label=label, linewidth=2)
    
    if bounds:
        min_y = np.min(y_array, axis=0)
        max_y = np.max(y_array, axis=0)
        ax.fill_between(x, min_y, max_y, color=color, alpha=0.1, edgecolor='black')
