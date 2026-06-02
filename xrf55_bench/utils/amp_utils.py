import torch


def torch_load_checkpoint(path, map_location=None):
    """Load checkpoints with weights_only=False when supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
