"""
Sequential-Speculative-Decoding: Speculative Decoding Methods for Language Models

This package implements various speculative decoding methods including:
- Standard Speculative Decoding
- Naive Hierarchical Speculative Decoding (HSD)
- Capped Hierarchical Speculative Decoding
"""

from .load_models import load_models
from .sampling import (
    get_logits,
    apply_temperature,
    sample_from_probs,
)
from .generation import (
    baseline_generate,
    speculative_generate,
)
from .benchmark import benchmark

__version__ = "0.1.0"

__all__ = [
    "load_models",
    "get_logits",
    "apply_temperature",
    "sample_from_probs",
    "baseline_generate",
    "speculative_generate",
    "benchmark",
]
