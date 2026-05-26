# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Nano-vLLM is a from-scratch reimplementation of vLLM in ~1,200 lines of Python. It targets a single CUDA host (optionally multi-GPU via tensor parallelism) for offline batched LLM inference. The API mirrors vLLM's: `LLM(model_path).generate(prompts, SamplingParams)`. Currently only Qwen3 models are wired up (`nanovllm/models/qwen3.py`); adding a new model means writing a parallel module file with the same layer building blocks and registering it in `engine/model_runner.py`.

## Running

The package only runs on a CUDA host with `flash-attn` installed; there is no test suite. Validate changes by running the bundled scripts against a downloaded Qwen3 checkpoint:

```bash
python example.py   # smoke test — short generation against ~/huggingface/Qwen3-0.6B/
python bench.py     # throughput benchmark — 256 random sequences, prints tok/s
```

Both scripts hardcode `~/huggingface/Qwen3-0.6B/` as the model path. `example.py` uses `enforce_eager=True` (skips CUDA graph capture, faster startup); `bench.py` uses CUDA graphs for realistic throughput numbers. To test tensor parallelism, pass `tensor_parallel_size=N` to `LLM(...)`.

## Architecture

The data flow on every `generate()` call is: `LLMEngine` → `Scheduler` → `ModelRunner` → `Qwen3ForCausalLM` → `Sampler`, looped until all sequences finish.

**Engine layer (`nanovllm/engine/`)** — the scheduling brain.
- `LLMEngine` (`llm_engine.py`) owns the tokenizer, scheduler, and model runner(s). For `tensor_parallel_size > 1` it spawns worker processes (one `ModelRunner` per GPU rank) via `torch.multiprocessing` and drives them through shared memory + `multiprocessing.Event` IPC. Rank 0 is in-process; ranks 1..N-1 run `ModelRunner.loop()`, polling shared memory for method calls.
- `Scheduler` (`scheduler.py`) implements **chunked prefill**: each `schedule()` step returns *either* a prefill batch *or* a decode batch, never mixed. Prefill greedily fills `max_num_batched_tokens`; only the first sequence in a prefill batch may be chunked across steps. Decode preempts running sequences (returning them to the waiting queue) when KV blocks run out.
- `BlockManager` (`block_manager.py`) implements **prefix caching**. KV cache is partitioned into fixed-size blocks (default 256 tokens). Each filled block is hashed (xxhash over token_ids, chained with the previous block's hash). When a new sequence arrives, leading blocks whose hashes match cached blocks are reused — `can_allocate` returns the count of reusable blocks, and `allocate` bumps their ref counts instead of writing fresh KV.
- `Sequence` (`sequence.py`) carries token state, the block table (list of physical block IDs), and prefill/decode bookkeeping (`num_cached_tokens`, `num_scheduled_tokens`). Its `__getstate__`/`__setstate__` are tuned for cheap pickling across the TP IPC channel — only the delta needed by workers is sent.

**Model runner (`engine/model_runner.py`)** — bridges scheduler decisions to GPU work.
- On init, runs `warmup_model()` to capture peak memory, then sizes the KV cache to fill `gpu_memory_utilization` of the remaining VRAM. The KV cache is a single contiguous tensor `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`; each layer's `Attention` module gets views as `k_cache`/`v_cache`.
- `prepare_prefill` / `prepare_decode` build the per-step tensors and stash them in a module-global `Context` (`nanovllm/utils/context.py`) read by `Attention.forward`. This avoids threading context through every layer.
- CUDA graphs (`capture_cudagraph`) are pre-recorded at a fixed set of batch sizes (1, 2, 4, 8, 16, 32, …) for *decode only*; `run_model` picks the smallest captured graph ≥ current batch size and replays it. Prefill always runs eagerly. `enforce_eager=True` disables this entirely.

**Layers (`nanovllm/layers/`)** — minimal building blocks, all tensor-parallel aware.
- `linear.py` exposes `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`. Each parameter has a `weight_loader` attribute that slices the HF checkpoint shard for the current `tp_rank` — this is how TP sharding happens transparently at load time.
- `attention.py` uses `flash_attn_varlen_func` for prefill and `flash_attn_with_kvcache` for decode. A custom Triton kernel (`store_kvcache_kernel`) writes new K/V into the paged cache using the `slot_mapping` tensor.
- Other layers (`rotary_embedding`, `layernorm`, `activation`, `embed_head`, `sampler`) follow the same pattern: small, TP-aware, no abstraction beyond what Qwen3 needs.

**Weight loading (`utils/loader.py`)** — iterates safetensors files; if a weight name matches a key in the model's `packed_modules_mapping` (e.g. `q_proj` → `qkv_proj`), the matching shard ID is passed to the parameter's `weight_loader`. This is the contract a new model module must satisfy: declare `packed_modules_mapping` and attach `weight_loader` to fused parameters.

## Conventions worth knowing

- The `Context` object in `utils/context.py` is **global mutable state**. Every forward pass calls `set_context(...)` before and `reset_context()` after. Don't try to make `Attention` stateless — the CUDA graph capture path depends on this.
- `kvcache_block_size` must be a multiple of 256 (asserted in `Config`). Triton kernel and FlashAttention paged layouts assume this.
- Greedy sampling is explicitly disallowed (`SamplingParams.temperature > 1e-10`). Don't add a code path for `temperature=0`; use a small temperature instead.
- The codebase favors terse, dense Python with minimal comments. New code should match: avoid docstrings, type hints inline at signature, no defensive checks beyond the existing `assert`s.
