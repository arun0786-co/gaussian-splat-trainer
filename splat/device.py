"""The few places where Apple and NVIDIA GPUs need different calls."""

import torch


def default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def empty_cache(dev):
    if dev == "mps":
        torch.mps.empty_cache()
    elif str(dev).startswith("cuda"):
        torch.cuda.empty_cache()


def memory_gb(dev):
    """Memory the process holds on the GPU: the driver's figure on Apple
    GPUs, the allocator's reserved peak on NVIDIA ones."""
    if dev == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if str(dev).startswith("cuda"):
        return torch.cuda.max_memory_reserved() / 2**30
    return 0.0
