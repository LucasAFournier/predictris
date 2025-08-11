import matplotlib.pyplot as plt
from io import BytesIO
import base64


class HistogramGenerator:
    @staticmethod
    def generate_level_histogram(level_counts: dict, steps: int) -> str:
        """Generate a histogram for level distribution."""
        fig = plt.figure(figsize=(4, 2))
        plt.bar(list(level_counts.keys()), list(level_counts.values()))
        plt.xticks(range(steps + 1))
        plt.xlim(-1, steps + 1)
        hist = HistogramGenerator._fig_to_base64(fig)
        plt.close(fig)
        return hist

    @staticmethod
    def generate_confidence_histogram(confidences: list) -> str:
        """Generate a histogram for confidence score distribution."""
        fig = plt.figure(figsize=(4, 2))
        plt.hist(confidences, bins=50, range=(0, 1))
        plt.xticks([0, 0.5, 1], ["0", "0.5", "1"])
        hist = HistogramGenerator._fig_to_base64(fig)
        plt.close(fig)
        return hist

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        data = buf.getvalue()
        buf.close()
        return base64.b64encode(data).decode("utf-8")
