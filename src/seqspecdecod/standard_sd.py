"""Implement Standard Speculative Decoding Method"""

import torch
from .sampling import get_logits, apply_temperature, sample_from_probs

@torch.no_grad()
def standard_speculative_step(
    draft_model,
    target_model,
    input_ids,
    gamma=5,
    draft_temperature=0.6,
    target_temperature=0.6,
):
    """
    Run one round of standard speculative decoding.

    Args:
        draft_model: Smaller draft model
        target_model: Larger target model
        input_ids: Current sequence, shape of [1, seq_len]
        gamma: Number of tokens to draft
        draft_temperature: Temperature for draft model
        target_temperature: Temperature for target model

    Returns:
        new_input_ids: Extended sequence
        n_accepted: Number of accepted draft tokens
        n_draft: Number of drafted tokens (gamma)
    """
    seq_len = input_ids.shape[1]

    # Draft model generates gamma tokens autoregressively
    draft_tokens = []
    draft_probs  = []

    current_ids = input_ids.clone()
    for _ in range(gamma):
        logits = get_logits(draft_model, current_ids)
        next_logits = logits[0, -1, :]
        probs = apply_temperature(next_logits, draft_temperature)
        token = sample_from_probs(probs)
        draft_tokens.append(token)
        draft_probs.append(probs)
        current_ids = torch.cat([current_ids, token.view(1, 1)], dim=1)

    # Target model verifies all drafted tokens in ONE BATCH
    target_logits = get_logits(target_model, current_ids)  # [1, seq_len+gamma, vocab]
    target_probs_draft = [
        apply_temperature(target_logits[0, seq_len - 1 + t, :], target_temperature)
        for t in range(gamma)
    ]
    target_probs_bonus = apply_temperature(
        target_logits[0, seq_len - 1 + gamma, :], target_temperature
    )

    # Perform accept, or reject and resample steps
    n_accepted = 0
    for t in range(gamma):
        p = target_probs_draft[t]
        q = draft_probs[t]
        token = draft_tokens[t]
        acceptance_prob = torch.min(p[token] / (q[token]), torch.tensor(1.0, device=p.device))
        u = torch.rand(1, device=input_ids.device)
        if u < acceptance_prob:
            # accept
            n_accepted += 1
        else:
            # reject and resample from residual distribution max(0, p - q) / Z
            residual = torch.max(p - q, torch.tensor(0.0, device=p.device))
            residual_sum = residual.sum()
            draft_tokens[t] = sample_from_probs(residual / residual_sum)
            break

    # Generate output
    if n_accepted == gamma:
        # get bonus token
        accepted_ids = torch.stack(draft_tokens).view(1, -1)
        bonus_token = sample_from_probs(target_probs_bonus).view(1, 1)
        new_ids = torch.cat([accepted_ids, bonus_token], dim=1)
    elif n_accepted == 0:
        new_ids = draft_tokens[0].view(1, 1)
    else:
        accepted_ids = torch.stack(draft_tokens[:n_accepted]).view(1, -1)
        corrected = draft_tokens[n_accepted].view(1, 1)
        new_ids = torch.cat([accepted_ids, corrected], dim=1)

    return torch.cat([input_ids, new_ids], dim=1), n_accepted, gamma