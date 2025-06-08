import argparse
from pathlib import Path
import numpy as np
from scipy import stats
from tqdm import tqdm

from predictris.learning.prediction_tree import PredictionTree
from predictris.utils import dir_from_params


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize context trees by pruning leaves')
    parser.add_argument('--origin', type=str, required=True,
                       help='Directory containing context trees to optimize')
    parser.add_argument('--cutoff', type=float, required=True,
                       help='Cut-off probability p for binomial test')
    parser.add_argument('--alpha', type=float, required=True,
                       help='Significance level alpha for statistical test')
    parser.add_argument('--save', action='store_true',
                       help='Save optimized trees')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress information')
    return parser.parse_args()


def should_remove_leaf(tree: PredictionTree, node_id, p: float, alpha: float) -> bool:
    """Test if leaf should be removed based on binomial distribution."""
    node = tree.nodes[node_id]
    
    # Fast path for common cases
    if node["eval_count"] == 0:
        return True
    
    # Standard path using binomial distribution
    quartile = stats.binom.ppf(1 - alpha, node["eval_count"], p)

    return node["ambig_count"] > quartile


def optimize_tree(tree: PredictionTree, p: float, alpha: float, verbose: bool = False):
    """Optimize tree by pruning leaves based on statistical criteria."""
    leaves = set(tree.get_leaves().copy())
    
    with tqdm(total=len(leaves), disable=not verbose) as pbar:
        while leaves:
            leaf = leaves.pop()
            
            if should_remove_leaf(tree, leaf, p, alpha):
                # Get successor (parent) before removing leaf
                parent_edge = tree.out_edge(leaf)
                parent = parent_edge[0] if parent_edge else None
                
                # Remove the leaf
                obs = tree.nodes[leaf]["obs"]
                tree.remove_node(leaf)
                tree.ids_from_obs[obs].remove(leaf)                
                
                # Add parent to leaves if it became a leaf
                if parent and tree.in_degree(parent) == 0:
                    leaves.add(parent)
            
            pbar.update(1)
    
    return tree


def main():
    args = parse_args()
    origin_dir = Path('results') / args.origin
    
    # Create output directory name
    output_dir = dir_from_params(
        origin=f'({args.origin})',
        cutoff=f'{args.cutoff:.2f}',
        alpha=f'{args.alpha:.2f}'
    )
    
    if args.verbose:
        print(f'Processing trees from {origin_dir}')
        print(f'Output directory: {output_dir}')
    
    # Process each tree file
    for tree_file in origin_dir.glob('*.gpickle'):  # Note: changed extension
        if args.verbose:
            print(f'Optimizing {tree_file.name}')
        
        # Load and optimize tree
        tree = PredictionTree.load(tree_file)
        optimized_tree = optimize_tree(tree, args.cutoff, args.alpha, args.verbose)
        
        # Save if requested
        if args.save:
            optimized_tree.save(output_dir)


if __name__ == '__main__':
    main()
