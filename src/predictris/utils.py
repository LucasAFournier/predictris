def dir_from_params(**attrs):
    """Create a directory name from a set of parameters."""
    return "_".join(f"{key}={value}" for key, value in attrs.items() if value is not None)
