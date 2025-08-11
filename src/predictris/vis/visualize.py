import os
from pathlib import Path
import glob
from matplotlib import pyplot as plt
import base64
from io import BytesIO
import numpy as np

from .tree_renderer import PredictionTreeRenderer
from .selector import generate_selector
from .colors import CMAP

from predictris.learning import PredictionTree


def visualize_trees(
    input_dir: str, output_dir: str, verbose: bool = False
) -> None:
    """Visualize all prediction trees from an input directory and generate a selector page."""
    input_path = Path(f"results/{input_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    os.makedirs(f"visualizations/{output_dir}", exist_ok=True)

    tree_data = {}
    for filepath in glob.glob(str(input_path / "*.gpickle")):
        input_filename = Path(filepath).stem

        tree = PredictionTree.load(Path(filepath))
        renderer = PredictionTreeRenderer(tree)
        renderer.save_graph(
            f"visualizations/{output_dir}/{input_filename}.html"
        )

        metrics = renderer._calculate_tree_metrics(tree)
        pred_img = to_base64(tree.pred_obs)
        tree_data[input_filename] = {
            "metrics": metrics,
            "pred_image": pred_img,
            "name": tree.name,
            "max_depth": metrics["max_depth"],
        }

    steps = max(data["max_depth"] for data in tree_data.values())

    generate_selector(tree_data, steps, f"visualizations/{output_dir}")


def to_base64(vec: np.ndarray) -> str:
    """Convert a matrix to a base64 encoded PNG image for use in HTML."""
    matrix = np.reshape(vec, (3, 3), order="F")
    fig, ax = plt.subplots(figsize=(2, 2), dpi=72)
    ax.matshow(matrix, cmap=CMAP, vmin=0, vmax=1)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str
