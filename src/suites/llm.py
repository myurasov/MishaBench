# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""LLM suite. CPU + CUDA + MPS where applicable.

Workloads:
  L1 tokenize         -- HF-Tokenizer throughput (CPU only -- the work
                         is pure-Rust token-piece scanning, no GPU op).
  L2 embed_minilm     -- sentence-transformers MiniLM-L6-v2 encode
                         throughput (sentences/s)
  L3 prefill_tinyllama -- TinyLlama-1.1B prefill latency at seq=512
                          (one big forward pass; ms)
  L4 decode_tinyllama  -- TinyLlama-1.1B autoregressive decode throughput
                          (tokens/s, generate 128 new tokens)

The default model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2.2 GiB
fp16, public on HF, no token needed). Override via `MISHABENCH_LLM_MODEL`
env var. First-run model download counts against the wall-clock budget;
subsequent runs hit the HF cache and start in seconds.

Quick mode: shorter sequences, fewer decode steps, batch 1 instead of 4.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import BenchConfig
from ..runner import Bench, register

_LLM: dict[str, Any] = {}
_TOK: dict[str, Any] = {}
_EMB: dict[str, Any] = {}


def _torch_device(name: str):
    import torch
    return torch.device(name)


def _model_id(cfg: BenchConfig) -> str:
    return os.environ.get("MISHABENCH_LLM_MODEL", cfg.llm_model)


def _embed_id(cfg: BenchConfig) -> str:
    return os.environ.get("MISHABENCH_EMBED_MODEL", cfg.embed_model)


def _load_llm(cfg: BenchConfig, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = _model_id(cfg)
    key = f"{mid}-{device}"
    if key in _LLM:
        return _LLM[key], _TOK[key]
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    # CPU fp32 keeps memory tight (~5 GiB); CUDA fp16 fits in ~2 GiB;
    # MPS prefers fp16 on Apple Silicon (much faster, plenty of memory).
    dtype = torch.float16 if device.startswith("cuda") or device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
    model.eval()
    model = model.to(_torch_device(device))
    _LLM[key] = model
    _TOK[key] = tok
    return model, tok


def _load_embed(cfg: BenchConfig, device: str):
    from sentence_transformers import SentenceTransformer
    mid = _embed_id(cfg)
    key = f"{mid}-{device}"
    if key in _EMB:
        return _EMB[key]
    model = SentenceTransformer(mid, device=device)
    _EMB[key] = model
    return model


# ---------- L1 Tokenizer throughput ----------

def bench_tokenize_cpu(cfg: BenchConfig):
    """Pure-CPU tokenizer throughput. Uses TinyLlama's tokenizer (LLaMA
    sentencepiece via HF fast Rust impl). No GPU op exists here -- this
    is a CPU baseline for the LLM suite to score against."""
    from transformers import AutoTokenizer
    mid = _model_id(cfg)
    n = 5_000 if cfg.quick else 50_000
    text = ("The quick brown fox jumps over the lazy dog. " * 8).strip()
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    import time as _time
    t0 = _time.perf_counter()
    for _ in range(n):
        _ = tok.encode(text, add_special_tokens=True)
    elapsed = _time.perf_counter() - t0
    ips = n / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(ips, 1), "tokenizations/s",
            {"n": n, "elapsed": round(elapsed, 3)})


# ---------- L2 Sentence embedding throughput ----------

def _bench_embed(cfg: BenchConfig, device: str):
    import time as _time
    n = 500 if cfg.quick else 5_000
    batch = 32 if device != "cpu" else 16
    model = _load_embed(cfg, device)
    sents = [
        f"This is a benchmark sentence number {i}, padded with extra context "
        f"so embeddings are non-trivial and have realistic length."
        for i in range(n)
    ]
    # warmup
    _ = model.encode(sents[:batch], batch_size=batch, show_progress_bar=False)
    t0 = _time.perf_counter()
    _ = model.encode(sents, batch_size=batch, show_progress_bar=False)
    elapsed = _time.perf_counter() - t0
    sps = n / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(sps, 2), "sentences/s",
            {"n": n, "batch": batch, "model": _embed_id(cfg)})


def bench_embed_cpu(cfg: BenchConfig):  return _bench_embed(cfg, "cpu")
def bench_embed_cuda(cfg: BenchConfig): return _bench_embed(cfg, "cuda")
def bench_embed_mps(cfg: BenchConfig):  return _bench_embed(cfg, "mps")


# ---------- L3 Prefill latency ----------

def _bench_prefill(cfg: BenchConfig, device: str):
    import time as _time

    import torch
    seq = 256 if cfg.quick else 512
    iters = 4 if cfg.quick else 8
    model, tok = _load_llm(cfg, device)
    # Build a fixed-length input. We tile the canned text to reach `seq`.
    text = "The capital of France is Paris. " * 100
    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq)
    input_ids = enc["input_ids"].to(_torch_device(device))
    # warmup
    with torch.inference_mode():
        _ = model(input_ids=input_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    with torch.inference_mode():
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = model(input_ids=input_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = _time.perf_counter() - t0
    # Report as tokens/s (prefill cost / second), not latency, so it
    # composes into the geomean cleanly with the decode throughput.
    toks = iters * input_ids.shape[1]
    tps = toks / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(tps, 1), "prefill tok/s",
            {"seq": seq, "iters": iters, "model": _model_id(cfg)})


def bench_prefill_cpu(cfg: BenchConfig):  return _bench_prefill(cfg, "cpu")
def bench_prefill_cuda(cfg: BenchConfig): return _bench_prefill(cfg, "cuda")
def bench_prefill_mps(cfg: BenchConfig):  return _bench_prefill(cfg, "mps")


# ---------- L4 Decode throughput ----------

def _bench_decode(cfg: BenchConfig, device: str):
    import time as _time

    import torch
    new_tokens = 64 if cfg.quick else 128
    model, tok = _load_llm(cfg, device)
    prompt = "Once upon a time, in a small village near the mountains,"
    enc = tok(prompt, return_tensors="pt").to(_torch_device(device))
    # warmup
    with torch.inference_mode():
        _ = model.generate(
            **enc, max_new_tokens=8, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    with torch.inference_mode():
        t0 = _time.perf_counter()
        out = model.generate(
            **enc, max_new_tokens=new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = _time.perf_counter() - t0
    generated = out.shape[1] - enc["input_ids"].shape[1]
    tps = generated / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(tps, 2), "decode tok/s",
            {"new_tokens": int(generated), "model": _model_id(cfg)})


def bench_decode_cpu(cfg: BenchConfig):  return _bench_decode(cfg, "cpu")
def bench_decode_cuda(cfg: BenchConfig): return _bench_decode(cfg, "cuda")
def bench_decode_mps(cfg: BenchConfig):  return _bench_decode(cfg, "mps")


# ---------- registration ----------

# CPU LLM benches are slow but worth running -- they establish the score
# baseline. The runner's per-bench budget keeps any one of them from
# blowing the whole budget.

register(Bench("llm.tok.cpu",      "llm", "Tokenizer encode",     "cpu",  bench_tokenize_cpu, expected_seconds=15))

register(Bench("llm.embed.cpu",    "llm", "MiniLM sentence embed", "cpu",  bench_embed_cpu, expected_seconds=60))
register(Bench("llm.embed.cuda",   "llm", "MiniLM sentence embed", "cuda", bench_embed_cuda, requires=("cuda",), expected_seconds=10))
register(Bench("llm.embed.mps",    "llm", "MiniLM sentence embed", "mps",  bench_embed_mps,  requires=("mps",), expected_seconds=20))

register(Bench("llm.prefill.cpu",  "llm", "TinyLlama prefill",     "cpu",  bench_prefill_cpu, expected_seconds=120))
register(Bench("llm.prefill.cuda", "llm", "TinyLlama prefill",     "cuda", bench_prefill_cuda, requires=("cuda",), expected_seconds=20))
register(Bench("llm.prefill.mps",  "llm", "TinyLlama prefill",     "mps",  bench_prefill_mps,  requires=("mps",), expected_seconds=60))

register(Bench("llm.decode.cpu",   "llm", "TinyLlama decode",      "cpu",  bench_decode_cpu, expected_seconds=180))
register(Bench("llm.decode.cuda",  "llm", "TinyLlama decode",      "cuda", bench_decode_cuda, requires=("cuda",), expected_seconds=15))
register(Bench("llm.decode.mps",   "llm", "TinyLlama decode",      "mps",  bench_decode_mps,  requires=("mps",), expected_seconds=60))
