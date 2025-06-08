from .histogram_utils import HistogramGenerator

class HtmlGenerator:
    @staticmethod
    def generate_html(num_trees: int, total_nodes: int, total_level_hist: str,
                     total_confidence_hist: str, tree_rows: str) -> str:
        """Generate the complete HTML page."""
        with open(__file__.replace('selector.py', 'html/selector.html'), 'r') as f:
            template = f.read()
        
        return template.format(
            num_trees=num_trees,
            total_nodes=total_nodes,
            total_level_hist=total_level_hist,
            total_confidence_hist=total_confidence_hist,
            tree_rows=tree_rows
        )

    @staticmethod
    def generate_tree_row(tree_name: str, display_name: str, metrics: dict, pred_image: str) -> str:
        """Generate HTML for a single tree row."""
        with open(__file__.replace('selector.py', 'html/tree_row.html'), 'r') as f:
            template = f.read()
        
        return template.format(
            tree_name=tree_name,
            display_name=display_name,
            pred_image=pred_image,
            total_nodes=metrics['total_nodes'],
            level_hist=metrics['level_hist'],
            confidence_hist=metrics['confidence_hist']
        )

class PredictionTreeSelector:
    def generate_page(self, tree_data: dict, steps: int, output_dir: str) -> None:
        """Generate a selector page with metrics and tree previews."""
        # Aggregate metrics and prediction images for all trees
        pred_images = {}
        tree_metrics = {}
        total_level_counts = {i: 0 for i in range(steps + 1)}
        all_confidences = []
        total_nodes = 0

        # Process each tree's data
        for tree_name, data in tree_data.items():
            pred_images[tree_name] = data['pred_image']
            metrics = data['metrics']
            total_nodes += metrics['total_nodes']
            for level, count in metrics['level_counts'].items():
                total_level_counts[level] += count
            all_confidences.extend(metrics['confidences'])
            
            # Generate histograms for each tree
            metrics['level_hist'] = HistogramGenerator.generate_level_histogram(
                metrics['level_counts'], steps)
            metrics['confidence_hist'] = HistogramGenerator.generate_confidence_histogram(
                metrics['confidences'])
            tree_metrics[tree_name] = metrics

        # Generate aggregated histograms
        total_level_hist = HistogramGenerator.generate_level_histogram(total_level_counts, steps)
        total_confidence_hist = HistogramGenerator.generate_confidence_histogram(all_confidences)

        # Sort trees by total node count (descending)
        sorted_tree_names = sorted(
            tree_metrics.keys(),
            key=lambda x: tree_metrics[x]['total_nodes'],
            reverse=True
        )

        # Generate HTML content
        tree_rows = ""
        for tree_name in sorted_tree_names:
            tree_rows += HtmlGenerator.generate_tree_row(
                tree_name,
                tree_data[tree_name]['name'],
                tree_metrics[tree_name],
                pred_images[tree_name]
            )

        html_content = HtmlGenerator.generate_html(
            len(tree_data),
            total_nodes,
            total_level_hist,
            total_confidence_hist,
            tree_rows
        )

        # Write the HTML file
        selector_path = f"{output_dir}/selector.html"
        with open(selector_path, "w", encoding="utf-8") as f:
            f.write(html_content)