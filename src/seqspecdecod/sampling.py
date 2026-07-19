"""Sampling Utilities for speculative decoding."""

import torch
import torch.nn.functional as F

@torch.no_grad()
def get_logits(model, input_ids):
    """
    Return logits for the next token at every position.

    Args:
        model: Language model
        input_ids: Input token IDs, shape [1, seq_len]

    Returns:
        logits: Shape [1, seq_len, vocab]
    """
    return model(input_ids).logits


def apply_temperature(logits, temperature):
    """
    Apply temperature to scale logits and return probabilities.

    Args:
        logits: Logits tensor for a single position
        temperature: Temperature parameter (0.0 for greedy, >0 for sampling)

    Returns:
        probs: Probability distribution
    """
    if temperature == 0.0:
        # return the id with maximum prob directly
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    else:
        # scale the prob by temperature
        return F.softmax(logits / temperature, dim=-1)


def sample_from_probs(probs):
    """
    Multinomial sample from a probability vector.

    Args:
        probs: Probability distribution tensor

    Returns:
        token: Sampled token id (scalar)
    """
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
