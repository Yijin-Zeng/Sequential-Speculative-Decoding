"""Naive Hierarchical Speculative Decoding (HSD) implementations."""

import torch
from .sampling import get_logits, apply_temperature, sample_from_probs

def branch_divergence(probs_p, probs_q, seq_probs_p, seq_probs_q):
    """
    Compute branch divergence between two distributions.

    Args:
        probs_p: Target model probabilities
        probs_q: Draft model probabilities
        seq_probs_p: Target sequence probability
        seq_probs_q: Draft sequence probability

    Returns:
        divergence: Branch divergence value
    """
    raw_diff = (seq_probs_p * probs_p) - (seq_probs_q * probs_q)
    return torch.sum(torch.clamp(raw_diff, min=0.0))


def branch_resample_prob(target_probs, draft_probs, target_sequence_probs, draft_sequence_probs):
    """
    Compute branch resampling probability.

    Args:
        target_probs: Target model probabilities
        draft_probs: Draft model probabilities
        target_sequence_probs: Target sequence probability
        draft_sequence_probs: Draft sequence probability

    Returns:
        res_prob: Resampling probability distribution
    """
  
    raw_diff = (target_sequence_probs * target_probs) - (draft_sequence_probs * draft_probs)
    floored_diff = torch.clamp(raw_diff, min=0.0)
    sum_probs = torch.sum(floored_diff)

    if sum_probs == 0.0:
        # residual mass is zero (target and draft agree exactly on this branch);
        # fall back to the target distribution to avoid dividing by zero
        return target_probs.view(1, -1)

    res_prob = floored_diff / sum_probs

    return res_prob.view(1, -1)


@torch.no_grad()
def naive_hsd_step(
    draft_model,
    target_model,
    input_ids,
    gamma=5,
    draft_temperature=0.6,
    target_temperature=0.6,
):
    """
    Run one round of naive hierarchical speculative decoding.

    Args:
        draft_model: Smaller draft model
        target_model: Larger target model
        input_ids: Current sequence, shape [1, seq_len]
        gamma: Number of tokens to draft
        draft_temperature: Temperature for draft model
        target_temperature: Temperature for target model

    Returns:
        new_input_ids: Extended sequence
        n_accepted: Number of accepted draft tokens
        n_draft: Number of drafted tokens (gamma)
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]

    # Draft model generates gamma tokens autoregressively
    draft_tokens = []
    draft_probs  = []
    current_ids = input_ids.clone()
    draft_sequence_probs = []

    for _ in range(gamma):
        logits = get_logits(draft_model, current_ids)
        next_logits = logits[0, -1, :]
        probs = apply_temperature(next_logits, draft_temperature)
        token = sample_from_probs(probs)
        draft_tokens.append(token)
        draft_probs.append(probs)
        if draft_sequence_probs == []:
            draft_sequence_probs.append(probs[token])
        else:
            draft_sequence_probs.append(probs[token] * draft_sequence_probs[-1])
        current_ids = torch.cat([current_ids, token.view(1, 1)], dim=1)

    # Target model verifies all draft tokens in one batch
    target_logits = get_logits(target_model, current_ids)  # [1, seq_len+gamma, vocab]
    target_probs_draft = [
        apply_temperature(target_logits[0, seq_len - 1 + t, :], target_temperature)
        for t in range(gamma)
    ]
    target_probs_bonus = apply_temperature(
        target_logits[0, seq_len - 1 + gamma, :], target_temperature
    )
    target_sequence_probs = []
    for i in range(gamma):
        # compute the prob for the sequence (path)
        if target_sequence_probs == []:
            target_sequence_probs.append(target_probs_draft[i][draft_tokens[i]])
        else:
            target_sequence_probs.append(target_probs_draft[i][draft_tokens[i]] * target_sequence_probs[-1])

    # Apply naive HSD
    seq_prob_p = target_sequence_probs[-1]
    seq_prob_q = draft_sequence_probs[-1]
    accptance_rate = torch.min(seq_prob_p / seq_prob_q, torch.tensor(1.0, device=device))
    tau = 0

    for t in range(gamma-1, -1, -1):
        # scan backwards
        u = torch.rand(1, device=device)
        if accptance_rate > u:
            tau = t
            break
        else:
            tau = t-1
            # update the acceptance rate
            branch_divergence_p_q = branch_divergence(
                probs_p=target_probs_draft[tau + 1],
                probs_q=draft_probs[tau + 1],
                seq_probs_p=target_sequence_probs[tau],
                seq_probs_q=draft_sequence_probs[tau]
            )
            branch_divergence_q_p = branch_divergence(
                probs_p=draft_probs[tau + 1],
                probs_q=target_probs_draft[tau + 1],
                seq_probs_p=draft_sequence_probs[tau],
                seq_probs_q=target_sequence_probs[tau]
            )
            accptance_rate = branch_divergence_p_q / torch.max(branch_divergence_p_q, branch_divergence_q_p)
            continue

    n_accepted = tau + 1

    # Perform resample
    if tau == gamma - 1:
        # the whole sequence is accepted, get the bonus token
        accepted_ids = torch.stack(draft_tokens).view(1, -1)
        bonus_token = sample_from_probs(target_probs_bonus).view(1, 1)
        new_ids = torch.cat([accepted_ids, bonus_token], dim=1)
    else:
        if n_accepted != 0:
            accepted_ids = torch.stack(draft_tokens[:tau+1]).view(1, -1)
            current_ids = torch.cat([input_ids, accepted_ids], dim=1)
        else:
            current_ids = input_ids

        # initialize target probs and draft probs
        target_probs = target_probs_draft[tau + 1]
        draft_probs_curr = draft_probs[tau + 1]
        # initialize target_sequence_probs and draft_sequence_probs
        if tau == -1:
            target_sequence_probs_curr = torch.tensor(1.0, device=device)
            draft_sequence_probs_curr = torch.tensor(1.0, device=device)
        else:
            target_sequence_probs_curr = target_sequence_probs[tau]
            draft_sequence_probs_curr = draft_sequence_probs[tau]

        resampled_ids = []
        for t in range(tau, gamma - 1):
            # sample from the branch resampling probability
            probs = branch_resample_prob(
                target_probs, draft_probs_curr,
                target_sequence_probs_curr, draft_sequence_probs_curr
            )
            token = sample_from_probs(probs)
            resampled_ids.append(token)
            current_ids = torch.cat([current_ids, token.view(1, 1)], dim=1)

            # run the target model
            logits = get_logits(target_model, current_ids)
            next_logits = logits[0, -1, :]
            target_sequence_probs_curr = target_sequence_probs_curr * target_probs[token]
            target_probs = apply_temperature(next_logits, target_temperature)

            # run the draft model
            logits = get_logits(draft_model, current_ids)
            next_logits = logits[0, -1, :]
            draft_sequence_probs_curr = draft_sequence_probs_curr * draft_probs_curr[token]
            draft_probs_curr = apply_temperature(next_logits, draft_temperature)

        resampled_ids = torch.stack(resampled_ids).view(1, -1)

        if n_accepted != 0:
            accepted_ids = torch.stack(draft_tokens[:tau+1]).view(1, -1)
            new_ids = torch.cat([accepted_ids, resampled_ids], dim=1)
        else:
            new_ids = resampled_ids

    return torch.cat([input_ids, new_ids], dim=1), n_accepted, gamma