import os
from pathlib import Path
import glob
from matplotlib import pyplot as plt
import base64
from io import BytesIO
import numpy as np

from predictris.learning import PredictionTree
from predictris.vis import PredictionTreeRenderer, PredictionTreeSelector
from .colors import CMAP


def visualize_trees(input_dir: str, output_dir: str, verbose: bool = False) -> None:
    """Visualize all prediction trees from an input directory and generate a selector page.
    
    Args:
        input_dir: Directory containing prediction tree files
        output_dir: Directory to save visualizations
        verbose: Whether to print progress messages
    """
    input_path = Path(f'results/{input_dir}')
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    os.makedirs(f'visualizations/{output_dir}', exist_ok=True)
    
    # Load all prediction trees from directory
    tree_data = {}
    for filepath in glob.glob(str(input_path / "*.gpickle")):
        # Extract filename without extension
        input_filename = Path(filepath).stem
            
        # Load and render tree
        tree = PredictionTree.load(Path(filepath))
        renderer = PredictionTreeRenderer(tree)
        renderer.save_graph(f'visualizations/{output_dir}/{input_filename}.html')
        
        # Collect tree metrics and prediction image - use input_filename as key
        metrics = renderer._calculate_tree_metrics(tree)
        pred_img = to_base64(tree.pred_obs)
        tree_data[input_filename] = {
            'metrics': metrics,
            'pred_image': pred_img,
            'name': tree.name,
            'max_depth': metrics['max_depth']  # Add max_depth from metrics
        }
        
    # Calculate steps as maximum depth across all trees
    steps = max(data['max_depth'] for data in tree_data.values())
    
    # Generate selector page
    selector = PredictionTreeSelector()
    selector.generate_page(tree_data, steps, f'visualizations/{output_dir}')


def to_base64(vec: np.ndarray) -> str:
    """Convert a matrix to a base64 encoded PNG image for use in HTML."""
    matrix = np.reshape(vec, (3, 3), order="F")
    # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    #     print(matrix)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=72)
    ax.matshow(matrix, cmap=CMAP, vmin=0, vmax=1)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

