# Sequential-Speculative-Decoding

A Python package implementing standard and sequential speculative decoding methods for accelerating large language model (LLM) inference.

## Overview

Speculative decoding is a lossless acceleration technique for LLM inference that uses a smaller, faster draft model to propose candidate tokens, which are then verified in parallel by the larger target model. This approach maintains the exact output distribution of the target model while achieving significant speedup.

Standard speculative decoding verifies in token level. If one drafted token is accepted, then the next drafted token will have chance
to be accepted, otherwise all following drafted tokens will be rejected directly. Sequential speculative decoding, on the other hand, verifies in sequence level. The Hierarchical Speculative Decoding (Zhou et al. (2026)) checks if the whole drafted sequence could be accepted at once, and if not, move gradully from backwards to accept shorter sequence.


## Installation

First install PyTorch matching your hardware (CUDA version or CPU) by following the instructions at https://pytorch.org/get-started/locally/. The default PyPI `torch` wheel may not match your CUDA setup. Then:

```bash
cd Sequential-Speculative-Decoding
pip install -e .
```

## Methods Implemented

### 1. Standard Speculative Decoding (Token Verification)

**How it works**:
- A small draft model generates gamma candidate tokens autoregressively
- The large target model evaluates all gamma tokens in one single batch (single forward pass)
- Tokens are then verified sequentially from left to right
- Each token Xi is accepted with probability min{p(Xi)/q(Xi), 1} where p is the target distribution and q is the draft distribution
- Verification stops when the first token is rejected
- If a token is rejected, it's resampled from a residual distribution: max{p(x) - q(x), 0} / Z
- If all tokens are accepted, a bonus token is sampled from the target model

**Pros and cons**:
-  Lossless, meaning that it preserves exact target model distribution
-  Simple implementation
-  Works with any draft model
-  Verifies tokens independently, it is shown to be sub-optimal in terms of expected acceptance length

**Speedup**: Typically 2x depending on draft model quality and gamma

**Reference**: Leviathan, Y., Kalman, M. and Matias, Y., 2023, July. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning (pp. 19274-19286). PMLR.

### 2. Hierarchical Speculative Decoding (HSD)

**How it works**:
- Drafting phase same as standard SD
- Verification uses hierarchical branch resampling:
  - Computes joint probability ratios r(X1:gamma) = p(X1:gamma) / q(X1:gamma)
  - Scans backward from gamma to 1 to find acceptance length tau
  - Uses branch divergence to compute acceptance probability at each position:
    ```
    Delta_Branch(X1:i-1) = p(X1:i-1) - q(X1:i-1)
    acceptance_rate = Delta_Branch(p,q) / max{Delta_Branch(p,q), Delta_Branch(q,p)}
    ```
  - If rejected, resamples remaining positions using branch resampling probability
  - Each resampling step calls both draft and target models

**Pros and Cons**:
- It is still a lossless method.
- Compared with standard speculative decoding method verifying in token level, HSD considers excess/deficient probability mass across branches.
- In practice, however, it almost always slower than standard speculative decoding because it needs to call target model multiple times during resampling

**Speedup**: Variable (0.8-1.2x vs standard SD depending on acceptance rate)

**When to use**: Best for scenarios with very low draft model quality where the hierarchical resampling can recover more tokens than standard rejection.

**Reference**: Zhou et al. (2026), "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding"


### 3. Capped Hierarchical Speculative Decoding

**Reference**: Zhou et al. (2026), "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding"

**Status**: ⚠️ Implementation under development

The capped version introduces:
- Capped prefix ratios r*(X₁:ₜ) that cap joint probabilities at 1
- Capped branch divergence for more efficient acceptance probability computation
- Single resampling step (vs multiple in naive HSD) to eliminate additional model calls

Currently not recommended for use.

## Quick Start

```python
from seqspecdecod import load_models, speculative_generate

# Load models
tokenizer, draft_model, target_model = load_models(
    draft_name="Qwen/Qwen3-0.6B",
    target_name="Qwen/Qwen3-4B",
    device="cuda"
)

# Generate with standard speculative decoding
result = speculative_generate(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    prompt="Explain machine learning in simple terms.",
    max_new_tokens=200,
    gamma=5,
    method="standard"
)

print(result["text"])
print(f"Speed: {result['tokens_per_sec']:.1f} tokens/s")
print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
```

## Benchmarking Multiple Methods

```python
from seqspecdecod import load_models, benchmark

tokenizer, draft_model, target_model = load_models()

prompts = [
    "Give me a short introduction to large language models.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is the transformer architecture and why is it so popular?",
    "What is gradient descent and how does it work?"
]

# Compare standard SD vs naive HSD
for method in ["standard", "naive_hsd"]:
    print(f"\n{'='*60}")
    print(f"Method: {method}")
    print(f"{'='*60}")
    
    baseline_results, spec_results = benchmark(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=200,
        gamma=5,
        method=method,
        verbose=True
    )
```

## Method Comparison

| Method | Speedup | Lossless | Implementation | Recommendation |
|--------|---------|----------|----------------|----------------|
| Standard SD | 2-3x | ✅ | Simple | **✅ Recommended** |
| Naive HSD | Variable | ✅ | Medium | Research/special cases |

## API Reference

### Available Methods

```python
method="standard"   # Standard speculative decoding (token verification)
method="naive_hsd"  # Naive hierarchical SD
```

### Parameters

- **gamma** (int, default=5): Number of tokens drafted per round. Higher values increase potential speedup but may reduce acceptance rate.
- **draft_temperature** (float, default=0.6): Sampling temperature for draft model.
- **target_temperature** (float, default=0.6): Sampling temperature for target model.
- **max_new_tokens** (int, default=200): Maximum tokens to generate.

## Understanding the Speedup

The speedup comes from evaluating multiple draft tokens in parallel with a single target model forward pass:

**Without Speculative Decoding:**
- Generate 100 tokens = 100 target model forward passes
- Time: 100 × T_target

**With Speculative Decoding (γ=5, acceptance rate 60%):**
- Generate 100 tokens ≈ 100/(5×0.6 + 1) ≈ 29 target model passes
- Time: 29 × T_target + 29 × 5 × T_draft
- Speedup: ~2-3x (since T_draft << T_target)

## Key Insights

### Lossless Guarantee

All implemented methods are **lossless** - they produce outputs with the exact same distribution as the target model running alone. This is achieved through:
1. Acceptance probabilities that correct for draft/target distribution mismatch
2. Residual distributions that fill in rejected positions from the target distribution
3. Mathematical proofs (see papers) showing distributional equivalence

## Examples

See the `examples/` directory:
- `basic_usage.py` - Simple generation examples
- `compare_methods.py` - Comprehensive method comparison

## References

1. **Leviathan, Y., Kalman, M., & Matias, Y. (2023).** Fast Inference from Transformers via Speculative Decoding. *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2211.17192

2. **Zhou, Y., Huang, F., Li, H., Wu, F., Wang, T., Zhang, J., Lin, J., & Cheng, Z. (2026).** Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding. https://arxiv.org/abs/2601.05724

## Implementation Notes

### Indexing Convention

The papers use 1-based indexing while Python uses 0-based. Our implementations carefully handle this conversion:

- Paper: `for i = 1, ..., γ` → Python: `for i in range(gamma)` (i = 0, ..., γ-1)
- Paper: `Xᵢ` (i-th token) → Python: `draft_tokens[i-1]` (when paper index is i)
- Paper: `X^i` (prefix 1 to i) → Python: `draft_tokens[:i]`

## Contributing

This package is designed for research and experimentation. Contributions are welcome:
- Bug fixes and improvements
- Additional verification methods
- Multi-draft support
- Extended benchmarks

## License

This project implements algorithms from published academic papers. Please cite the original papers when using these methods in your research.

## Acknowledgments

We thank the authors of the original papers for their groundbreaking work on speculative decoding methods.
