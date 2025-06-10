import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris agent')
    parser.add_argument('--dir', type=str, required=False, default=None,
                        help='Directory for existing prediction trees')
    parser.add_argument('--tetrominos', type=str, required=False, nargs='+', help='List of tetrominos')
    parser.add_argument('--episode', type=int, help='Maximum number of actions per episode')
    parser.add_argument('--action', type=str, choices=['from_active', 'random'], help='Action choice method')
    parser.add_argument('--activation', type=str, choices=['by_confidence', 'all'], help='Activation function')
    parser.add_argument('--steps', type=int, help='Total number of steps to run')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()
