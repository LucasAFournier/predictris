def dir_from_params(args):
    """Create a directory name from arguments."""
    sorted_tetrominos = ''.join(sorted(args.tetrominos))

    return f"tetrominos={sorted_tetrominos}_depth={args.depth}"