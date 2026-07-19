# Sequential Speculative Decoding

A Python package implementing speculative decoding methods for faster LLM inference — without changing the output distribution.

Speculative decoding uses a small, fast **draft model** to propose tokens in batches, then lets the larger **target model** verify them in a single forward pass. The result is the same text you'd get from the target model alone, just faster.

This package implements three variants:
- **Standard SD** — token-level verification (Leviathan et al., 2023)
- **Naive HSD** — sequence-level hierarchical verification (Zhou et al., 2026)
- **Capped HSD** — improved HSD with capped prefix ratios

## Installation

First, install PyTorch for your hardware (see [pytorch.org](https://pytorch.org/get-started/locally/) to pick the right CUDA build). Then:

```bash
pip install sequential-speculative-decoding
```

Or from source:

```bash
git clone https://github.com/Yijin-Zeng/Sequential-Speculative-Decoding
cd Sequential-Speculative-Decoding
pip install -e .
```

## Quick Start

```python
from seqspecdecod import load_models, speculative_generate

tokenizer, draft_model, target_model = load_models(
    draft_name="Qwen/Qwen3-0.6B",
    target_name="Qwen/Qwen3-4B",
    device="cuda",
)

result = speculative_generate(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    prompt="Explain what machine learning is in one paragraph.",
    max_new_tokens=200,
    gamma=5,
    method="standard",
)

print(result["text"])
print(f"Speed: {result['tokens_per_sec']:.1f} tok/s")
print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
```

## Benchmarking

Use `benchmark` to compare speculative decoding against the target-only baseline across multiple prompts:

```python
from seqspecdecod import load_models, benchmark

tokenizer, draft_model, target_model = load_models()

prompts = [
    "Give me a short introduction to large language models.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is the transformer architecture and why is it so popular?",
    "What is gradient descent and how does it work?",
]

baseline_results, spec_results = benchmark(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    prompts=prompts,
    max_new_tokens=200,
    gamma=5,
    method="standard",
    verbose=True,
)
```

## Benchmark Results

Measured on Qwen3-0.6B (draft) → Qwen3-4B (target), 4 prompts, 200 tokens each, γ = 5. Hardware: NVIDIA GeForce RTX 3070 (8 GB VRAM).

| Method | Speed (tok/s) | Speedup | Acceptance Rate |
|---|---|---|---|
| Target only (baseline) | 1.9 | 1.0× | — |
| Standard SD | 3.2 | **1.7×** | 43% |
| Capped HSD | 3.7 | **1.9×** | 52% |
| Naive HSD | 1.8 | 0.9× | 49% |

**Effect of γ (draft length):** Standard SD peaks around γ = 3–5 and slows down past γ = 10 as the acceptance rate drops.

![Speed vs gamma](https://raw.githubusercontent.com/Yijin-Zeng/Sequential-Speculative-Decoding/main/docs/speculative_decoding_speed-1.png)

> **Note:** Naive HSD is slower than the baseline in practice because its resampling step requires multiple additional target model calls. It is included here for research comparison purposes.

## Methods

### Standard Speculative Decoding

The draft model generates γ tokens autoregressively. The target model evaluates them all in one forward pass. Tokens are accepted left-to-right with probability min(p/q, 1); the first rejection stops the chain and resamples from a residual distribution. Always produces an extra bonus token if all γ are accepted.

### Naive Hierarchical SD (HSD)

Verifies the entire drafted sequence at once using joint probability ratios. Scans backward to find the longest acceptable prefix, then resamples the rest using branch divergence probabilities. Each resampling step calls both models again, which is what makes it slow in practice despite a higher per-token acceptance rate.

### Capped HSD

A more efficient variant of HSD that caps prefix ratios to avoid the multi-step resampling. Achieves the best speed in our experiments.

## API

```python
# Load models
tokenizer, draft_model, target_model = load_models(
    draft_name="Qwen/Qwen3-0.6B",   # any HF model
    target_name="Qwen/Qwen3-4B",
    device="cuda",                   # or "cpu", "mps"
)

# Generate
result = speculative_generate(
    draft_model, target_model, tokenizer,
    prompt="...",
    max_new_tokens=200,
    gamma=5,                         # draft tokens per round
    method="standard",               # "standard" | "naive_hsd" | "capped_hsd"  
    draft_temperature=0.6,
    target_temperature=0.6,
)
# result keys: text, tokens_generated, tokens_per_sec, acceptance_rate, average_accepted_n

# Benchmark against target-only baseline
baseline_results, spec_results = benchmark(
    draft_model, target_model, tokenizer,
    prompts=[...],
    max_new_tokens=200,
    gamma=5,
    method="standard",
    verbose=True,
)
```

See `examples/` for runnable scripts.

## Project Structure

```
src/seqspecdecod/   # the package — generation, sampling, benchmarking, model loading
notebooks/          # experiments: benchmarks for each method and gamma comparison
examples/           # runnable scripts showing basic usage and method comparison
docs/               # figures generated from the notebooks
```

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). *ICML*.
2. Zhou, Y. et al. (2026). [Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](https://arxiv.org/abs/2601.05724).

## License

MIT
