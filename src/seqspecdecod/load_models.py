"""Model loading utilities."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_models(
    draft_name="Qwen/Qwen3-0.6B",
    target_name="Qwen/Qwen3-4B",
    device="cuda",
):
    """
    Load draft and target models for speculative decoding.

    Args:
        draft_name: model name for the draft model
        target_name: model name for the target model
        device: Device to load models on ('cuda' or 'cpu')

    Returns:
        tokenizer: Tokenizer for both models
        draft_model: Loaded draft model in eval mode
        target_model: Loaded target model in eval mode
    """
    print(f"Loading tokenizer from {draft_name}")
    tokenizer = AutoTokenizer.from_pretrained(draft_name)

    print(f"Loading draft model: ({draft_name})")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_name, torch_dtype=torch.bfloat16, device_map=device
    )
    draft_model.eval()

    print(f"Loading target model: ({target_name})")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=torch.bfloat16, device_map=device
    )
    target_model.eval()

    return tokenizer, draft_model, target_model