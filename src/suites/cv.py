# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Computer-vision suite. CPU + CUDA + MPS where applicable.

Workloads:
  C1 resnet50_inference   -- ResNet-50 forward pass throughput (img/s)
  C2 efficientnet_b0      -- EfficientNet-B0 forward pass throughput
  C3 conv2d_micro         -- Pure Conv2D microbenchmark (single op, big tensor)
  C4 image_resize         -- 224-side image resize throughput (decode + resize)
  C5 dinov2_features      -- DINOv2-small feature extraction throughput

All work runs on synthetic random tensors so there's no image-dataset
dependency and no first-time download cost beyond the model weights
themselves (~100 MiB total when CV runs end-to-end). Models are loaded
without a checkpoint when network is unavailable -- the timing is
representative either way (we measure forward-pass cost, not accuracy).

Quick mode: 1/4 the iterations, same shapes. The forward-pass cost is
shape-bound, not iteration-bound, so 1/4 iters keeps the timing honest.
"""

from __future__ import annotations

from typing import Any

from ..config import BenchConfig
from ..runner import Bench, register

# ---------- shared cache ----------

_MODELS: dict[str, Any] = {}
_TENSORS: dict[str, Any] = {}


def _torch_device(name: str):
    import torch
    return torch.device(name)


def _iters(cfg: BenchConfig, full: int) -> int:
    return max(2, full // 4) if cfg.quick else full


def _make_input(shape: tuple[int, ...], device: str):
    """Random fp32 input tensor on `device`. Cached per (shape, device)."""
    import torch
    key = f"{shape}-{device}"
    if key in _TENSORS:
        return _TENSORS[key]
    t = torch.randn(*shape, dtype=torch.float32, device=_torch_device(device))
    _TENSORS[key] = t
    return t


def _load_torchvision(name: str, device: str):
    """Load a torchvision pretrained model. Falls back to fresh weights
    if the network is unreachable (the checkpoint download is a best-effort
    convenience -- timing is the same with or without weights)."""
    import torchvision.models as tvm

    key = f"{name}-{device}"
    if key in _MODELS:
        return _MODELS[key]

    factory = getattr(tvm, name)
    try:
        # weights="DEFAULT" pulls torchvision's recommended checkpoint
        model = factory(weights="DEFAULT")
    except Exception:
        model = factory(weights=None)
    model.eval()
    model = model.to(_torch_device(device))
    _MODELS[key] = model
    return model


def _bench_classifier(cfg: BenchConfig, device: str, model_name: str,
                      batch: int = 8, full_iters: int = 32):
    import torch
    iters = _iters(cfg, full_iters)
    model = _load_torchvision(model_name, device)
    x = _make_input((batch, 3, 224, 224), device)
    # Warm-up (1 iter)
    with torch.inference_mode():
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    import time as _time
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = _time.perf_counter() - t0

    imgs = iters * batch
    ips = imgs / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(ips, 2), "img/s",
            {"iters": iters, "batch": batch, "model": model_name})


# ---------- C1 ResNet50 ----------

def bench_resnet50_cpu(cfg: BenchConfig):
    return _bench_classifier(cfg, "cpu", "resnet50", batch=4, full_iters=12)

def bench_resnet50_cuda(cfg: BenchConfig):
    return _bench_classifier(cfg, "cuda", "resnet50", batch=32, full_iters=64)

def bench_resnet50_mps(cfg: BenchConfig):
    return _bench_classifier(cfg, "mps", "resnet50", batch=16, full_iters=32)


# ---------- C2 EfficientNet B0 ----------

def bench_effnetb0_cpu(cfg: BenchConfig):
    return _bench_classifier(cfg, "cpu", "efficientnet_b0", batch=4, full_iters=16)

def bench_effnetb0_cuda(cfg: BenchConfig):
    return _bench_classifier(cfg, "cuda", "efficientnet_b0", batch=32, full_iters=64)

def bench_effnetb0_mps(cfg: BenchConfig):
    return _bench_classifier(cfg, "mps", "efficientnet_b0", batch=16, full_iters=32)


# ---------- C3 Conv2D microbenchmark ----------

def _bench_conv2d(cfg: BenchConfig, device: str):
    import torch
    iters = _iters(cfg, 32)
    # 64x64 in_channels, 64 out, 3x3 kernel, batch 32, 56x56 spatial.
    # Decent workload but small enough to run on CPU in a few seconds.
    in_c, out_c = 64, 64
    x = _make_input((32, in_c, 56, 56), device)
    conv = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1).to(_torch_device(device))
    conv.eval()
    with torch.inference_mode():
        _ = conv(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    import time as _time
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = conv(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = _time.perf_counter() - t0
    # Throughput as TFLOPS-equivalent: 2*K*K*Cin*Cout*H*W*B per iter
    flops_per_iter = 2 * 9 * in_c * out_c * 56 * 56 * 32
    gflops = flops_per_iter * iters / elapsed / 1e9
    return ("throughput", round(gflops, 2), "GFLOPS",
            {"iters": iters, "elapsed": round(elapsed, 3)})


def bench_conv2d_cpu(cfg: BenchConfig):  return _bench_conv2d(cfg, "cpu")
def bench_conv2d_cuda(cfg: BenchConfig): return _bench_conv2d(cfg, "cuda")
def bench_conv2d_mps(cfg: BenchConfig):  return _bench_conv2d(cfg, "mps")


# ---------- C4 Image resize (CPU-only -- OpenCV / Pillow path) ----------

def bench_image_resize_cpu(cfg: BenchConfig):
    """OpenCV resize throughput. Uses opencv-python-headless; runs on the
    CPU. There's no MPS / CUDA equivalent in the headless build, so this
    bench is CPU-only and just establishes a CPU image-pipeline baseline."""
    import cv2
    import numpy as np
    n = 256 if cfg.quick else 1024
    src = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    import time as _time
    t0 = _time.perf_counter()
    for _ in range(n):
        cv2.resize(src, (224, 224), interpolation=cv2.INTER_AREA)
    elapsed = _time.perf_counter() - t0
    ips = n / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(ips, 2), "img/s",
            {"src": "1920x1080", "dst": "224x224", "n": n})


# ---------- C5 DINOv2-small feature extraction (HuggingFace) ----------

def _bench_dinov2(cfg: BenchConfig, device: str):
    import time as _time

    import torch
    from transformers import AutoModel
    model_id = "facebook/dinov2-small"
    iters = _iters(cfg, 16)
    batch = 4 if device == "cpu" else 16
    model = _MODELS.get(f"hfdino-{device}")
    if model is None:
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        model = model.to(_torch_device(device))
        _MODELS[f"hfdino-{device}"] = model
    # DINOv2 expects 224x224 fp32 inputs (mean/std normalised); for
    # timing purposes random tensors are fine.
    x = _make_input((batch, 3, 224, 224), device)
    with torch.inference_mode():
        _ = model(pixel_values=x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = model(pixel_values=x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = _time.perf_counter() - t0
    imgs = iters * batch
    ips = imgs / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(ips, 2), "img/s",
            {"iters": iters, "batch": batch, "model": model_id})


def bench_dinov2_cpu(cfg: BenchConfig):  return _bench_dinov2(cfg, "cpu")
def bench_dinov2_cuda(cfg: BenchConfig): return _bench_dinov2(cfg, "cuda")
def bench_dinov2_mps(cfg: BenchConfig):  return _bench_dinov2(cfg, "mps")


# ---------- registration ----------

register(Bench("cv.resnet50.cpu",  "cv", "ResNet-50 inference",       "cpu",  bench_resnet50_cpu, expected_seconds=45))
register(Bench("cv.resnet50.cuda", "cv", "ResNet-50 inference",       "cuda", bench_resnet50_cuda, requires=("cuda",), expected_seconds=10))
register(Bench("cv.resnet50.mps",  "cv", "ResNet-50 inference",       "mps",  bench_resnet50_mps, requires=("mps",), expected_seconds=20))

register(Bench("cv.effnet.cpu",    "cv", "EfficientNet-B0 inference", "cpu",  bench_effnetb0_cpu, expected_seconds=30))
register(Bench("cv.effnet.cuda",   "cv", "EfficientNet-B0 inference", "cuda", bench_effnetb0_cuda, requires=("cuda",), expected_seconds=8))
register(Bench("cv.effnet.mps",    "cv", "EfficientNet-B0 inference", "mps",  bench_effnetb0_mps, requires=("mps",), expected_seconds=15))

register(Bench("cv.conv2d.cpu",    "cv", "Conv2D microbenchmark",     "cpu",  bench_conv2d_cpu, expected_seconds=10))
register(Bench("cv.conv2d.cuda",   "cv", "Conv2D microbenchmark",     "cuda", bench_conv2d_cuda, requires=("cuda",), expected_seconds=5))
register(Bench("cv.conv2d.mps",    "cv", "Conv2D microbenchmark",     "mps",  bench_conv2d_mps, requires=("mps",), expected_seconds=8))

register(Bench("cv.resize.cpu",    "cv", "Image resize 1080p->224",   "cpu",  bench_image_resize_cpu, expected_seconds=15))

register(Bench("cv.dinov2.cpu",    "cv", "DINOv2-small features",     "cpu",  bench_dinov2_cpu, expected_seconds=60))
register(Bench("cv.dinov2.cuda",   "cv", "DINOv2-small features",     "cuda", bench_dinov2_cuda, requires=("cuda",), expected_seconds=15))
register(Bench("cv.dinov2.mps",    "cv", "DINOv2-small features",     "mps",  bench_dinov2_mps, requires=("mps",), expected_seconds=30))
