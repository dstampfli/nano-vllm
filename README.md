<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

> Fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — adds a `uv`-based setup (pinned torch + a prebuilt flash-attn wheel so `uv sync` works without compiling) and a [Google Colab demo](colab_demo.ipynb) for running on a Colab GPU with no local CUDA host.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

Clone the repo and use `uv` (recommended — pulls the pinned torch and a prebuilt flash-attn wheel, so nothing compiles):
```bash
git clone https://github.com/dstampfli/nano-vllm.git
cd nano-vllm
uv sync
```

Or install directly with pip:
```bash
pip install git+https://github.com/dstampfli/nano-vllm.git
```

## Run on Google Colab

No local GPU? Open [`colab_demo.ipynb`](colab_demo.ipynb) in Google Colab and run it on an Ampere-or-newer GPU (L4/A100). FlashAttention needs compute capability ≥ 8.0, so the free-tier T4 won't work.

## Model Download

To download the model weights manually, use the following command:
```bash
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dstampfli/nano-vllm&type=Date)](https://www.star-history.com/#dstampfli/nano-vllm&Date)