# Third-Party Notices

MishaBench is licensed under the [Apache License 2.0](LICENSE). It depends on the third-party open-source software listed below. Each dependency is distributed under its own license; license texts are available in the linked upstream repositories and are pulled into the runtime environment via the standard Python packaging tooling (`uv` / `pip`).

## Runtime dependencies

These packages are required for MishaBench to run. They are pulled at install time (`./mishabench install` or `uv sync`); their source code is not redistributed within this repository.

| Package | License | Project URL |
|---|---|---|
| [typer](https://github.com/fastapi/typer) | MIT | https://github.com/fastapi/typer |
| [rich](https://github.com/Textualize/rich) | MIT | https://github.com/Textualize/rich |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | https://github.com/yaml/pyyaml |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | https://github.com/giampaolo/psutil |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | https://github.com/numpy/numpy |
| [pandas](https://github.com/pandas-dev/pandas) | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| [Polars](https://github.com/pola-rs/polars) | MIT | https://github.com/pola-rs/polars |
| [PyArrow (Apache Arrow)](https://github.com/apache/arrow) | Apache-2.0 | https://github.com/apache/arrow |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU | https://github.com/python-pillow/Pillow |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | Apache-2.0 | https://github.com/opencv/opencv-python |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | https://github.com/pytorch/vision |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | https://github.com/huggingface/transformers |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |
| [tiktoken](https://github.com/openai/tiktoken) | MIT | https://github.com/openai/tiktoken |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | https://github.com/huggingface/huggingface_hub |

## Optional GPU dependencies

The `data` suite's CUDA paths use NVIDIA RAPIDS. These are not in `pyproject.toml`; the `./mishabench install --gpu` wrapper installs them out-of-band against `https://pypi.nvidia.com`. The data-suite GPU benches soft-import these and skip cleanly when they are not installed.

| Package | License | Project URL |
|---|---|---|
| [cudf-cu12](https://github.com/rapidsai/cudf) | Apache-2.0 | https://github.com/rapidsai/cudf |
| [cupy-cuda12x](https://github.com/cupy/cupy) | MIT | https://github.com/cupy/cupy |

## Development dependencies

These packages are used only for the project's own development workflow (`./mishabench test`, `./mishabench lint`, `./mishabench fmt`); they are not required at runtime and are not part of the published package.

| Package | License | Project URL |
|---|---|---|
| [pytest](https://github.com/pytest-dev/pytest) | MIT | https://github.com/pytest-dev/pytest |
| [ruff](https://github.com/astral-sh/ruff) | MIT | https://github.com/astral-sh/ruff |

## Models

The CV and LLM suites download pretrained model weights at first run. Weights are not redistributed within this repository; they are fetched from the providers below and cached locally.

| Model | License | Source |
|---|---|---|
| TinyLlama 1.1B Chat v1.0 | Apache-2.0 | https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| facebook/dinov2-small | Apache-2.0 | https://huggingface.co/facebook/dinov2-small |
| torchvision ResNet-50 (`IMAGENET1K_V2` weights) | BSD-3-Clause | https://github.com/pytorch/vision |
| torchvision EfficientNet-B0 (`IMAGENET1K_V1` weights) | BSD-3-Clause | https://github.com/pytorch/vision |

The default LLM and embedding models can be overridden at run time via the `MISHABENCH_LLM_MODEL` and `MISHABENCH_EMBED_MODEL` environment variables. If you swap in a different model, ensure its license terms are compatible with your use.

## Updates

When dependencies are added, removed, or upgraded across major versions, update this file in the same commit. Run `./mishabench install --force` after edits to `pyproject.toml` to ensure the lockfile is consistent.
