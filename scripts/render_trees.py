import argparse
from predictris.vis import visualize_trees


def parse_args():
    parser = argparse.ArgumentParser(description='Render context tree visualizations')
    parser.add_argument('--dir', type=str, required=True,
                       help='Directory containing context trees to visualize')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress information')
    
    parser.add_argument('--height', type=str, required=False, default='600px',
                       help='Height of visualization')
    parser.add_argument('--width', type=str, required=False, default='1500px',
                       help='Width of visualization')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Use the new visualize_trees function with origin_dir
    visualize_trees(
        input_dir=args.dir,
        output_dir=args.dir,  # Use same directory for output
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()