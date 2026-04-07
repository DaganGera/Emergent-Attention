"""
FLOPs counting, parameter counting, and throughput measurement.
"""

import time
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count


def count_flops(model: nn.Module, img_size: int = 224, device: torch.device | None = None) -> dict:
    """
    Count FLOPs for a single forward pass using fvcore.

    Args:
        model:    The model to analyse
        img_size: Input image size (square)
        device:   Device to run on (defaults to model's device)

    Returns:
        {
            "total_gflops": float,
            "by_module": dict   (fvcore module breakdown)
        }
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    return {
        "total_gflops": flops.total() / 1e9,
        "by_module": dict(flops.by_module()),
    }


def count_parameters(model: nn.Module) -> dict:
    """
    Count total, trainable, and per-component parameters.

    Returns:
        {"total": int, "trainable": int, "by_module": dict}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module = {k: v.item() for k, v in parameter_count(model).items()}
    return {"total": total, "trainable": trainable, "by_module": by_module}


def measure_throughput(
    model: nn.Module,
    img_size: int = 224,
    batch_size: int = 64,
    n_warmup: int = 20,
    n_measure: int = 100,
    device: torch.device | None = None,
) -> dict:
    """
    Measure inference throughput (images / second).

    Args:
        model:      The model to benchmark
        img_size:   Input spatial size
        batch_size: Batch size for benchmarking
        n_warmup:   Number of warmup iterations (not timed)
        n_measure:  Number of timed iterations
        device:     Device (defaults to model's device)

    Returns:
        {"images_per_sec": float, "ms_per_batch": float}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    total_images = batch_size * n_measure
    images_per_sec = total_images / elapsed
    ms_per_batch = elapsed / n_measure * 1000

    return {
        "images_per_sec": images_per_sec,
        "ms_per_batch": ms_per_batch,
        "batch_size": batch_size,
    }
