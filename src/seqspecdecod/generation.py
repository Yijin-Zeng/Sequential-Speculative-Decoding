"""Text generation functions using various speculative decoding methods."""

import time
import torch
from .sampling import get_logits, apply_temperature, sample_from_probs
from .standard_sd import standard_speculative_step
from .naive_hsd import naive_hsd_step
from .capped_hsd import capped_hsd_step

@torch.no_grad()
def baseline_generate(
    target_model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    target_temperature=0.6,
    enable_thinking=False,
):
    """
    Generate tokens using only the target model (baseline).

    Args:
        target_model: Target language model
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        target_temperature: Temperature for sampling
        enable_thinking: Enable thinking mode for Qwen models

    Returns:
        dict: Generation results including text, tokens, timing
    """
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {"enable_thinking": True} if enable_thinking else {"enable_thinking": False}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    input_ids = tokenizer([text], return_tensors="pt").input_ids
    device = target_model.device
    input_ids = input_ids.to(device)

    prompt_len = input_ids.shape[1]
    eos_token_id = tokenizer.eos_token_id

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    current_ids = input_ids.clone()
    while (current_ids.shape[1] - prompt_len) < max_new_tokens:
        logits = get_logits(target_model, current_ids)
        next_logits = logits[0, -1, :]
        probs = apply_temperature(next_logits, target_temperature)
        token = sample_from_probs(probs).view(1, 1)
        current_ids = torch.cat([current_ids, token], dim=1)
        if token.item() == eos_token_id:
            current_ids = current_ids[:, :-1]
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    runing_time = time.perf_counter() - t0

    output_ids = current_ids[0, prompt_len:].tolist()

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id != tokenizer.unk_token_id and think_end_id in output_ids:
        index = len(output_ids) - output_ids[::-1].index(think_end_id)
    else:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    n_tokens = len(output_ids)

    return {
        "text": content,
        "thinking_content": thinking_content,
        "tokens_generated": n_tokens,
        "runing_time": runing_time,
        "tokens_per_sec": n_tokens / runing_time,
    }


@torch.no_grad()
def speculative_generate(
    draft_model,
    target_model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    gamma=5,
    draft_temperature=0.6,
    target_temperature=0.6,
    enable_thinking=False,
    method="standard",
):
    """
    Generate text using speculative decoding.

    Args:
        draft_model: Draft language model
        target_model: Target language model
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        gamma: Number of tokens to draft per round
        draft_temperature: Temperature for draft model
        target_temperature: Temperature for target model
        enable_thinking: Enable thinking mode for Qwen models
        method: Speculative decoding method ('standard', 'naive_hsd', 'capped_hsd')

    Returns:
        dict: Generation results including text, tokens, timing, acceptance rate
    """
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {"enable_thinking": True} if enable_thinking else {"enable_thinking": False}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    input_ids = tokenizer([text], return_tensors="pt").input_ids
    device = next(draft_model.parameters()).device
    input_ids = input_ids.to(device)

    prompt_len = input_ids.shape[1]
    total_accepted = 0
    total_drafted = 0
    n_rounds = 0
    n_accepted_list = []
    eos_token_id = tokenizer.eos_token_id

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Select the appropriate decoding step function
    if method == "standard":
        step_fn = standard_speculative_step
    elif method == "naive_hsd":
        step_fn = naive_hsd_step
    elif method == "capped_hsd":
        step_fn = capped_hsd_step
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'standard', 'naive_hsd', 'capped_hsd'")

    while (input_ids.shape[1] - prompt_len) < max_new_tokens:
        if method == "capped_hsd":
            input_ids, n_accepted, n_draft = step_fn(
                draft_model, target_model, input_ids,
                gamma=gamma,
                draft_temperature=draft_temperature,
                target_temperature=target_temperature,
            )
        else:
            input_ids, n_accepted, n_draft = step_fn(
                draft_model, target_model, input_ids,
                gamma=gamma,
                draft_temperature=draft_temperature,
                target_temperature=target_temperature,
            )
        total_accepted += n_accepted
        total_drafted += n_draft
        n_rounds += 1
        n_accepted_list.append(n_accepted)

        new_tokens = input_ids[0, prompt_len:].tolist()
        if eos_token_id in new_tokens:
            eos_pos = new_tokens.index(eos_token_id)
            input_ids = input_ids[:, : prompt_len + eos_pos]
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    runing_time = time.perf_counter() - t0

    output_ids = input_ids[0, prompt_len:].tolist()

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id != tokenizer.unk_token_id and think_end_id in output_ids:
        index = len(output_ids) - output_ids[::-1].index(think_end_id)
    else:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    n_tokens = len(output_ids)
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0

    return {
        "text": content,
        "thinking_content": thinking_content,
        "tokens_generated": n_tokens,
        "draft_tokens": total_drafted,
        "accepted_tokens": total_accepted,
        "acceptance_rate": acceptance_rate,
        "n_rounds": n_rounds,
        "runing_time": runing_time,
        "tokens_per_sec": n_tokens / runing_time,
        "n_accepted_list": n_accepted_list,
        "average_accepted_n": sum(n_accepted_list) / len(n_accepted_list) if n_accepted_list else 0.0,
    }