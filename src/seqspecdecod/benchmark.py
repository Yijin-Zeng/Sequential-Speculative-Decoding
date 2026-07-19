"""Benchmarking utilities for comparing different methods."""

import torch
from .generation import baseline_generate, speculative_generate


def benchmark(
    draft_model,
    target_model,
    tokenizer,
    prompts,
    max_new_tokens=200,
    gamma=5,
    draft_temperature=0.6,
    target_temperature=0.6,
    enable_thinking=False,
    warmup=True,
    method="standard",
    verbose=True,
):
    """
    Run baseline and speculative decoding on each prompt, return both results.

    Args:
        draft_model: Draft language model
        target_model: Target language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum number of tokens to generate
        gamma: Number of tokens to draft per round
        draft_temperature: Temperature for draft model
        target_temperature: Temperature for target model
        enable_thinking: Enable thinking mode for Qwen models
        warmup: Whether to run warmup before benchmarking
        method: Speculative decoding method ('standard', 'naive_hsd', 'capped_hsd', 'block_verification')
        cap: Cap parameter for capped HSD
        verbose: Whether to print results

    Returns:
        baseline_results: List of baseline generation results
        speculative_results: List of speculative decoding results
    """
    device = target_model.device

    if warmup:
        # run target and draft model before comparison
        if verbose:
            print('Running warmup...')
        baseline_generate(
            target_model, tokenizer, "Hello.",
            max_new_tokens=10, target_temperature=target_temperature,
        )
        speculative_generate(
            draft_model, target_model, tokenizer, "Hello.",
            max_new_tokens=10, gamma=gamma,
            draft_temperature=draft_temperature,
            target_temperature=target_temperature,
            method=method,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

    baseline_results = []
    speculative_results = []

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f'\nProcessing prompt {i+1}/{len(prompts)}')

        # baseline: target model only
        b = baseline_generate(
            target_model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            target_temperature=target_temperature,
            enable_thinking=enable_thinking,
        )
        baseline_results.append(b)

        # speculative decoding
        s = speculative_generate(
            draft_model, target_model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            draft_temperature=draft_temperature,
            target_temperature=target_temperature,
            enable_thinking=enable_thinking,
            method=method,
        )
        speculative_results.append(s)

        if verbose:
            # output results
            speedup_tokens = s["tokens_per_sec"] / b["tokens_per_sec"]
            print(f"Speedup: {speedup_tokens:.2f}x")
            print(f"Baseline: {b['tokens_generated']:4d} tokens, {b['runing_time']:6.2f}s, {b['tokens_per_sec']:6.1f} tok/s")
            print(f"Speculative ({method}): {s['tokens_generated']:4d} tokens, {s['runing_time']:6.2f}s, {s['tokens_per_sec']:6.1f} tok/s, "
                  f"acceptance: {s['acceptance_rate']:.1%}, avg accepted: {s['average_accepted_n']:.2f}")

    return baseline_results, speculative_results
