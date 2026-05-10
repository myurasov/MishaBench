# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Computer-vision suite. CPU + CUDA + MPS where applicable, plus
multi-GPU CUDA variants for hosts with >=2 GPUs.

Two model tiers per workload type:

  Small (everyday inference; baseline cross-host comparison):
    cv.resnet50           ResNet-50 forward pass
    cv.effnet             EfficientNet-B0 forward pass
    cv.dinov2             DINOv2-small feature extraction (~21M params)

  Large (>= 2 GB of GPU RAM in fp32; saturates an RTX 3090 / GB10):
    cv.vit_h14            ViT-H/14 (632M params, ~2.5 GB fp32)
    cv.regnet128gf        RegNet-Y-128GF (644M params, ~2.6 GB fp32)
    cv.dinov2_giant       DINOv2-Giant features (1.1B params, ~4.4 GB fp32)

  Microbench:
    cv.conv2d             Pure Conv2D 64x64 channels, 56x56 spatial
    cv.resize             OpenCV 1080p->224p resize (CPU-only)

Multi-GPU classifier variants (`cuda_multi` device) wrap the model in
`nn.DataParallel`, letting the bench scale a single batch across all
visible GPUs. Wattage attribution sums power across all GPUs (see
`power.PowerWindow.power_for_device`); points-per-watt for the
multi-GPU bench is the honest figure for "this whole box, this load".

dtype policy: fp32 on CPU (deterministic, plenty of RAM); fp16 on
CUDA + MPS for the large models (a 632M-param fp32 model is 2.5 GB
weights + activations; fp16 cuts that in half and matches typical
inference deployments).

Quick mode: 1/4 the iterations, same shapes. Forward-pass cost is
shape-bound, not iter-bound, so 1/4 iters keeps the timing honest.
"""

from __future__ import annotations

import time as _time
from typing import Any

from ..config import BenchConfig
from ..runner import Bench, register

_MODELS: dict[str, Any] = {}
_TENSORS: dict[str, Any] = {}


def _torch_device(name: str):
    import torch
    if name == "cuda_multi":
        # Multi-GPU benches do their work on cuda:0 with DataParallel
        # replicating across all visible devices.
        return torch.device("cuda:0")
    return torch.device(name)


def _is_cuda(name: str) -> bool:
    return name.startswith("cuda")


def _sync(name: str) -> None:
    import torch
    if _is_cuda(name) and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _iters(cfg: BenchConfig, full: int) -> int:
    return max(2, full // 4) if cfg.quick else full


def _make_input(shape: tuple[int, ...], device: str, dtype=None):
    """Random input tensor on `device` (cuda_multi -> cuda:0). Cached
    per (shape, device, dtype). dtype defaults to fp32; pass torch.float16
    for the large models on CUDA / MPS."""
    import torch
    if dtype is None:
        dtype = torch.float32
    key = f"{shape}-{device}-{dtype}"
    if key in _TENSORS:
        return _TENSORS[key]
    t = torch.randn(*shape, dtype=dtype, device=_torch_device(device))
    _TENSORS[key] = t
    return t


def _load_torchvision(name: str, device: str, dtype=None,
                      data_parallel: bool = False):
    """Load a torchvision pretrained model. Falls back to fresh weights
    if the network is unreachable. Optional dtype cast (typically fp16
    for the large models on CUDA / MPS). When `data_parallel=True`,
    wraps the model in `nn.DataParallel` so it spreads across all
    visible CUDA devices -- used for the cuda_multi benches."""
    import torch
    import torchvision.models as tvm

    cache_key = f"{name}-{device}-{dtype}-dp={data_parallel}"
    if cache_key in _MODELS:
        return _MODELS[cache_key]

    factory = getattr(tvm, name)
    try:
        model = factory(weights="DEFAULT")
    except Exception:
        model = factory(weights=None)
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(_torch_device(device))
    if data_parallel:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("DataParallel requires >=2 visible CUDA devices")
        model = torch.nn.DataParallel(model)
    _MODELS[cache_key] = model
    return model


def _bench_classifier(cfg: BenchConfig, device: str, model_name: str,
                      batch: int = 8, full_iters: int = 32, dtype=None,
                      input_size: int = 224):
    """Generic torchvision classifier inference bench. Reports img/s."""
    import torch
    iters = _iters(cfg, full_iters)
    data_parallel = device == "cuda_multi"
    model = _load_torchvision(model_name, device, dtype=dtype,
                              data_parallel=data_parallel)
    x = _make_input((batch, 3, input_size, input_size), device, dtype=dtype)
    with torch.inference_mode():
        _ = model(x)  # warmup
        _sync(device)
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        _sync(device)
        elapsed = _time.perf_counter() - t0
    imgs = iters * batch
    ips = imgs / elapsed if elapsed > 0 else 0.0
    notes = {"iters": iters, "batch": batch, "model": model_name,
             "input": f"{input_size}x{input_size}",
             "dtype": str(dtype) if dtype else "fp32",
             "seconds": round(elapsed, 4)}
    if data_parallel:
        notes["n_gpus"] = torch.cuda.device_count()
    return ("throughput", round(ips, 2), "img/s", notes)


# ---------- C1 ResNet-50 (small) ----------

def bench_resnet50_cpu(cfg):  return _bench_classifier(cfg, "cpu", "resnet50", batch=4, full_iters=12)
def bench_resnet50_cuda(cfg): return _bench_classifier(cfg, "cuda", "resnet50", batch=32, full_iters=64)
def bench_resnet50_cuda_multi(cfg): return _bench_classifier(cfg, "cuda_multi", "resnet50", batch=64, full_iters=64)
def bench_resnet50_mps(cfg):  return _bench_classifier(cfg, "mps", "resnet50", batch=16, full_iters=32)


# ---------- C2 EfficientNet-B0 (small) ----------

def bench_effnetb0_cpu(cfg):  return _bench_classifier(cfg, "cpu", "efficientnet_b0", batch=4, full_iters=16)
def bench_effnetb0_cuda(cfg): return _bench_classifier(cfg, "cuda", "efficientnet_b0", batch=32, full_iters=64)
def bench_effnetb0_cuda_multi(cfg): return _bench_classifier(cfg, "cuda_multi", "efficientnet_b0", batch=64, full_iters=64)
def bench_effnetb0_mps(cfg):  return _bench_classifier(cfg, "mps", "efficientnet_b0", batch=16, full_iters=32)


# ---------- C3 Conv2D microbenchmark ----------

def _bench_conv2d(cfg: BenchConfig, device: str):
    import torch
    iters = _iters(cfg, 32)
    in_c, out_c = 64, 64
    x = _make_input((32, in_c, 56, 56), device)
    conv = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1).to(_torch_device(device))
    conv.eval()
    with torch.inference_mode():
        _ = conv(x)
        _sync(device)
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = conv(x)
        _sync(device)
        elapsed = _time.perf_counter() - t0
    flops_per_iter = 2 * 9 * in_c * out_c * 56 * 56 * 32
    gflops = flops_per_iter * iters / elapsed / 1e9
    return ("throughput", round(gflops, 2), "GFLOPS",
            {"iters": iters, "elapsed": round(elapsed, 4)})


def bench_conv2d_cpu(cfg):  return _bench_conv2d(cfg, "cpu")
def bench_conv2d_cuda(cfg): return _bench_conv2d(cfg, "cuda")
def bench_conv2d_mps(cfg):  return _bench_conv2d(cfg, "mps")


# ---------- C4 Image resize (CPU-only) ----------

def bench_image_resize_cpu(cfg: BenchConfig):
    """OpenCV resize throughput. opencv-python-headless ships CPU-only,
    so this is the CV suite's CPU image-pipeline baseline."""
    import cv2
    import numpy as np
    n = 256 if cfg.quick else 1024
    src = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    t0 = _time.perf_counter()
    for _ in range(n):
        cv2.resize(src, (224, 224), interpolation=cv2.INTER_AREA)
    elapsed = _time.perf_counter() - t0
    ips = n / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(ips, 2), "img/s",
            {"src": "1920x1080", "dst": "224x224", "n": n,
             "seconds": round(elapsed, 4)})


# ---------- C5 DINOv2 features (small + giant) ----------

def _bench_dinov2(cfg: BenchConfig, device: str, model_id: str,
                  batch: int, full_iters: int, dtype=None):
    """Generic DINOv2 feature-extraction bench (HuggingFace transformers
    AutoModel). dtype=None -> fp32; torch.float16 for CUDA/MPS on the
    Giant variant (else 4.4 GB just in weights)."""
    import torch
    from transformers import AutoModel
    iters = _iters(cfg, full_iters)
    cache_key = f"hfdino-{model_id}-{device}-{dtype}"
    model = _MODELS.get(cache_key)
    if model is None:
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(_torch_device(device))
        _MODELS[cache_key] = model
    x = _make_input((batch, 3, 224, 224), device, dtype=dtype)
    with torch.inference_mode():
        _ = model(pixel_values=x)
        _sync(device)
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = model(pixel_values=x)
        _sync(device)
        elapsed = _time.perf_counter() - t0
    imgs = iters * batch
    ips = imgs / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(ips, 2), "img/s",
            {"iters": iters, "batch": batch, "model": model_id,
             "dtype": str(dtype) if dtype else "fp32",
             "seconds": round(elapsed, 4)})


def bench_dinov2_cpu(cfg):  return _bench_dinov2(cfg, "cpu",  "facebook/dinov2-small", batch=4, full_iters=16)
def bench_dinov2_cuda(cfg): return _bench_dinov2(cfg, "cuda", "facebook/dinov2-small", batch=16, full_iters=32)
def bench_dinov2_mps(cfg):  return _bench_dinov2(cfg, "mps",  "facebook/dinov2-small", batch=16, full_iters=32)


# ---------- C6 ViT-H/14 (LARGE: 632M params, 2.5 GB fp32) ----------

# torchvision's vit_h_14 IMAGENET1K_SWAG_E2E_V1 weights expect 518x518
# inputs (the SWAG fine-tuning resolution); default alias routes there.

def bench_vit_h14_cpu(cfg):
    return _bench_classifier(cfg, "cpu", "vit_h_14", batch=1, full_iters=4,
                             input_size=518)
def bench_vit_h14_cuda(cfg):
    import torch
    return _bench_classifier(cfg, "cuda", "vit_h_14", batch=8, full_iters=16,
                             dtype=torch.float16, input_size=518)
def bench_vit_h14_cuda_multi(cfg):
    import torch
    return _bench_classifier(cfg, "cuda_multi", "vit_h_14", batch=16, full_iters=16,
                             dtype=torch.float16, input_size=518)
def bench_vit_h14_mps(cfg):
    import torch
    return _bench_classifier(cfg, "mps", "vit_h_14", batch=4, full_iters=8,
                             dtype=torch.float16, input_size=518)


# ---------- C7 RegNet-Y-128GF (LARGE: 644M params, 2.6 GB fp32) ----------

def bench_regnet128_cpu(cfg):
    return _bench_classifier(cfg, "cpu", "regnet_y_128gf", batch=1, full_iters=2)
def bench_regnet128_cuda(cfg):
    import torch
    return _bench_classifier(cfg, "cuda", "regnet_y_128gf", batch=8, full_iters=16,
                             dtype=torch.float16)
def bench_regnet128_cuda_multi(cfg):
    import torch
    return _bench_classifier(cfg, "cuda_multi", "regnet_y_128gf", batch=16, full_iters=16,
                             dtype=torch.float16)
def bench_regnet128_mps(cfg):
    import torch
    return _bench_classifier(cfg, "mps", "regnet_y_128gf", batch=4, full_iters=8,
                             dtype=torch.float16)


# ---------- C8 DINOv2-Giant (LARGE: 1.1B params, 4.4 GB fp32) ----------

def bench_dinov2_giant_cpu(cfg):
    return _bench_dinov2(cfg, "cpu", "facebook/dinov2-giant", batch=1, full_iters=2)
def bench_dinov2_giant_cuda(cfg):
    import torch
    return _bench_dinov2(cfg, "cuda", "facebook/dinov2-giant", batch=8, full_iters=16,
                         dtype=torch.float16)
def bench_dinov2_giant_mps(cfg):
    import torch
    return _bench_dinov2(cfg, "mps", "facebook/dinov2-giant", batch=4, full_iters=8,
                         dtype=torch.float16)


# ---------- registration ----------

# Small models (baseline)
register(Bench("cv.resnet50.cpu",        "cv", "ResNet-50 inference",       "cpu",        bench_resnet50_cpu,         expected_seconds=45))
register(Bench("cv.resnet50.cuda",       "cv", "ResNet-50 inference",       "cuda",       bench_resnet50_cuda,        requires=("cuda",), expected_seconds=10))
register(Bench("cv.resnet50.cuda_multi", "cv", "ResNet-50 inference (multi-GPU)", "cuda_multi", bench_resnet50_cuda_multi, requires=("cuda_multi",), expected_seconds=10))
register(Bench("cv.resnet50.mps",        "cv", "ResNet-50 inference",       "mps",        bench_resnet50_mps,         requires=("mps",), expected_seconds=20))

register(Bench("cv.effnet.cpu",          "cv", "EfficientNet-B0 inference", "cpu",        bench_effnetb0_cpu,         expected_seconds=30))
register(Bench("cv.effnet.cuda",         "cv", "EfficientNet-B0 inference", "cuda",       bench_effnetb0_cuda,        requires=("cuda",), expected_seconds=8))
register(Bench("cv.effnet.cuda_multi",   "cv", "EfficientNet-B0 inference (multi-GPU)", "cuda_multi", bench_effnetb0_cuda_multi, requires=("cuda_multi",), expected_seconds=8))
register(Bench("cv.effnet.mps",          "cv", "EfficientNet-B0 inference", "mps",        bench_effnetb0_mps,         requires=("mps",), expected_seconds=15))

register(Bench("cv.conv2d.cpu",          "cv", "Conv2D microbenchmark",     "cpu",        bench_conv2d_cpu,           expected_seconds=10))
register(Bench("cv.conv2d.cuda",         "cv", "Conv2D microbenchmark",     "cuda",       bench_conv2d_cuda,          requires=("cuda",), expected_seconds=5))
register(Bench("cv.conv2d.mps",          "cv", "Conv2D microbenchmark",     "mps",        bench_conv2d_mps,           requires=("mps",), expected_seconds=8))

register(Bench("cv.resize.cpu",          "cv", "Image resize 1080p->224",   "cpu",        bench_image_resize_cpu,     expected_seconds=15))

register(Bench("cv.dinov2.cpu",          "cv", "DINOv2-small features",     "cpu",        bench_dinov2_cpu,           expected_seconds=60))
register(Bench("cv.dinov2.cuda",         "cv", "DINOv2-small features",     "cuda",       bench_dinov2_cuda,          requires=("cuda",), expected_seconds=15))
register(Bench("cv.dinov2.mps",          "cv", "DINOv2-small features",     "mps",        bench_dinov2_mps,           requires=("mps",), expected_seconds=30))

# Large models (>= 2 GB GPU RAM in fp32)
register(Bench("cv.vit_h14.cpu",         "cv", "ViT-H/14 inference (518)",  "cpu",        bench_vit_h14_cpu,          expected_seconds=240))
register(Bench("cv.vit_h14.cuda",        "cv", "ViT-H/14 inference (518)",  "cuda",       bench_vit_h14_cuda,         requires=("cuda",), expected_seconds=30))
register(Bench("cv.vit_h14.cuda_multi",  "cv", "ViT-H/14 inference (multi-GPU, 518)", "cuda_multi", bench_vit_h14_cuda_multi, requires=("cuda_multi",), expected_seconds=30))
register(Bench("cv.vit_h14.mps",         "cv", "ViT-H/14 inference (518)",  "mps",        bench_vit_h14_mps,          requires=("mps",), expected_seconds=120))

register(Bench("cv.regnet128.cpu",       "cv", "RegNet-Y-128GF inference",  "cpu",        bench_regnet128_cpu,        expected_seconds=300))
register(Bench("cv.regnet128.cuda",      "cv", "RegNet-Y-128GF inference",  "cuda",       bench_regnet128_cuda,       requires=("cuda",), expected_seconds=30))
register(Bench("cv.regnet128.cuda_multi","cv", "RegNet-Y-128GF inference (multi-GPU)", "cuda_multi", bench_regnet128_cuda_multi, requires=("cuda_multi",), expected_seconds=30))
register(Bench("cv.regnet128.mps",       "cv", "RegNet-Y-128GF inference",  "mps",        bench_regnet128_mps,        requires=("mps",), expected_seconds=120))

register(Bench("cv.dinov2_giant.cpu",    "cv", "DINOv2-Giant features",     "cpu",        bench_dinov2_giant_cpu,     expected_seconds=300))
register(Bench("cv.dinov2_giant.cuda",   "cv", "DINOv2-Giant features",     "cuda",       bench_dinov2_giant_cuda,    requires=("cuda",), expected_seconds=45))
register(Bench("cv.dinov2_giant.mps",    "cv", "DINOv2-Giant features",     "mps",        bench_dinov2_giant_mps,     requires=("mps",), expected_seconds=180))
