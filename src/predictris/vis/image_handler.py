import threading
import base64
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import List, Tuple, Optional

from .colors import CMAP


class ImageHandler:
    """Handles the creation and caching of node visualization images."""

    def __init__(self) -> None:
        self.cache_lock = threading.Lock()
        self.image_cache = {}
        plt.switch_backend("Agg")

    def get_node_image(self, observation: List[float]) -> str:
        matrix_key = tuple(observation)
        cached_image = self.get_cached_image(matrix_key)
        if cached_image is None:
            matrix_array = np.array(observation).reshape(
                (int(sqrt(len(observation))), int(sqrt(len(observation)))),
                order="F",
            )
            cached_image = self.generate_image(matrix_array, CMAP)
            self.set_cached_image(matrix_key, cached_image)
        return cached_image

    def get_cached_image(self, matrix_key: Tuple[float, ...]) -> Optional[str]:
        with self.cache_lock:
            return self.image_cache.get(matrix_key)

    def set_cached_image(
        self, matrix_key: Tuple[float, ...], image: str
    ) -> None:
        with self.cache_lock:
            self.image_cache[matrix_key] = image

    def generate_image(self, matrix_array: np.ndarray, colormap: str) -> str:
        """Generate a base64 encoded PNG image from a matrix array."""
        fig = plt.figure(figsize=(2, 2), dpi=72)
        ax = fig.add_subplot(111)
        ax.imshow(matrix_array, vmin=0, vmax=1, cmap=colormap)
        ax.axis("off")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode()
