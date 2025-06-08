def filename_from_obs(obs: tuple):
    """Create a filename from an observation."""
    return "_".join(str(x) for x in obs)

def init_node_metrics():
    return {
        "eval_count": 0,
        "correct_count": 0,
        "confidence": 0.5,
    }

def update_node_metrics(node, correct_pred: bool):
    node["eval_count"] += 1
    if correct_pred:
        node["correct_count"] += 1
    node["confidence"] = (node["correct_count"] + 1) / (node["eval_count"] + 2)