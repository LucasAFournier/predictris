def is_valid(obs: tuple) -> bool:
    """Check if observation is valid."""
    return not all(value == 0 for value in obs)