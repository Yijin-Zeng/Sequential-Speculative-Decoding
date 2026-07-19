import torch
from .sampling import get_logits, apply_temperature, sample_from_probs

def compute_joint_probability_ratios(draft_tokens, draft_probs, target_probs_draft):
    """
    Compute the likelihood ratios r(X1:t) = p(X1:t) / q(X1:t) for all positions t.

    Args:
        draft_tokens: List of draft tokens
        draft_probs: List of draft probability distributions
        target_probs_draft: List of target probability distributions

    Returns:
        ratios: List of joint probability ratios r(X1:t) for t=1 to gamma
    """
    ratios = []
    ratio = 1.0

    for t, token in enumerate(draft_tokens):
        p_t = target_probs_draft[t][token]
        q_t = draft_probs[t][token]
        ratio = ratio * (p_t / q_t)
        ratios.append(ratio)

    return ratios


def compute_maximum_prefix_ratio_index(ratios):
    """
    Compute m(X1:t) - the position where joint probability ratio is maximized according to Definition 4
    in the paper: m(X1:t) = arg max_{1≤i<t} r(X1:i) or 0 if max r(X1:i) ≤ 1

    Args:
        ratios: List of joint probability ratios

    Returns:
        m: Maximum prefix ratio index (0 if no ratio exceeds 1)
    """
    m = 0
    max_ratio = 0.0

    for i, ratio in enumerate(ratios):
        if ratio > 1.0 and ratio > max_ratio:
            max_ratio = ratio
            m = i + 1  # this is important, this method requires us to use 1-indexed position
                       # (because m could = 0 in the paper, and we don't want it to be -1,
                       #  so need to handle this carefully)
    return m


def compute_capped_prefix_ratio(ratios, t):
    """
    Compute the capped prefix ratio r*(X1:t) according to Definition 5.

    r*(X1:t) = min{r(X1:m), 1} * r(X_{m+1:t})

    When m > 0 and r(X1:m) > 1, this simplifies to:
    r*(X1:t) = r(X_{m+1:t}) = r(X1:t) / r(X1:m)

    Args:
        ratios: List of joint probability ratios
        t: Current position (0-indexed)

    Returns:
        r_star: Capped prefix ratio
    """
    # find m given X1:t
    m = compute_maximum_prefix_ratio_index(ratios[:t])

    if m == 0:
        # no capping needed, return r(x1:t)
        return ratios[t-1]
    else:
        # r*(X1:t) = r(X1:t) / r(X1:m)
        return ratios[t-1] / ratios[m-1]


def compute_capped_branch_divergences(
    draft_tokens,
    draft_probs,
    target_probs_draft,
    t,
    ratios
):
    """
    Compute D^cap_Branch(p,q | X1:t) and D^cap_Branch(q,p | X1:t).

    According to Definitions 5 and 6:

    D^cap_Branch(p,q | X1:t) = sum_{r*(X1:t)>1} (r*(X1:t) - 1) * q(X1:t)
    D^cap_Branch(q,p | X1:t) = sum_{r*(X1:t)≤1} (1 - r*(X1:t)) * q(X1:t)

    where the sum is over all X1:t \in Branch(X1:t)

    Args:
        draft_tokens: List of draft tokens
        draft_probs: List of draft probability distributions
        target_probs_draft: List of target probability distributions
        t: End point for the sequence
        ratios: Joint probability ratios
        m: Maximum prefix ratio index

    Returns:
        D_cap_pq: D^cap_Branch(p,q | X1:t)
        D_cap_qp: D^cap_Branch(q,p | X1:t)
    """
    device = draft_probs[0].device
    m = compute_maximum_prefix_ratio_index(ratios[:t])
    divide = 1

    # Compute joint probabilities up to tau
    q_prefix = torch.tensor(1.0, device=device)
    p_prefix = torch.tensor(1.0, device=device)
    for i in range(t):
        q_prefix = q_prefix * draft_probs[i][draft_tokens[i]]
        p_prefix = p_prefix * target_probs_draft[i][draft_tokens[i]]
        if m >= 0:
            m -= 1
        if m == 0:
            divide = p_prefix / q_prefix

    # initialize the results
    D_cap_pq = torch.tensor(0.0, device=device)
    D_cap_qp = torch.tensor(0.0, device=device)

    # Get distributions at position t+1
    q_dist = draft_probs[t]  # q(x_{t+1} | X1:t)
    p_dist = target_probs_draft[t]  # p(x_{t+1} | X1:t)

    q_joint = q_prefix * q_dist
    p_joint = p_prefix * p_dist
    ratio_branch = torch.where(
    q_joint > 0,
    p_joint / q_joint,
    torch.zeros_like(q_joint)
    )
    r_star = ratio_branch / divide
    D_cap_pq = torch.sum(torch.where(r_star > 1, r_star - 1, torch.zeros_like(r_star)) * q_joint)
    D_cap_qp = torch.sum(torch.where(r_star <= 1, 1 - r_star, torch.zeros_like(r_star)) * q_joint)
    return r_star, D_cap_pq, D_cap_qp



@torch.no_grad()
def capped_hsd_step(
    draft_model,
    target_model,
    input_ids,
    gamma=5,
    draft_temperature=0.6,
    target_temperature=0.6,
):
    """
    Run one round of capped hierarchical speculative decoding.

    This implements Algorithm 2 from the paper with Capped Branch Resampling.

    Key difference from Naive HSD: Uses capped prefix ratios and performs
    only ONE resampling step, eliminating additional draft/target model calls.

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
    draft_probs = []
    current_ids = input_ids.clone()

    for _ in range(gamma):
        logits = get_logits(draft_model, current_ids)
        next_logits = logits[0, -1, :]
        probs = apply_temperature(next_logits, draft_temperature)
        token = sample_from_probs(probs)
        draft_tokens.append(token)
        draft_probs.append(probs)
        current_ids = torch.cat([current_ids, token.view(1, 1)], dim=1)

    # Target model verifies all draft tokens in one batch
    target_logits = get_logits(target_model, current_ids)
    target_probs_draft = [
        apply_temperature(target_logits[0, seq_len - 1 + t, :], target_temperature)
        for t in range(gamma)
    ]
    target_probs_bonus = apply_temperature(
        target_logits[0, seq_len - 1 + gamma, :], target_temperature
    )

    # compute joint probability ratios r(X1:gamma)
    ratios = compute_joint_probability_ratios(draft_tokens, draft_probs, target_probs_draft)

    # compute acceptance probability using capped ratios
    # same backward scan as naive HSD but with capped divergences
    # initialize tau first
    tau = 0

    # Compute capped ratio for full sequence
    r_star_gamma = compute_capped_prefix_ratio(ratios, gamma - 1)
    acceptance_rate = torch.min(r_star_gamma, torch.tensor(1.0, device=device))
    for t in range(gamma - 1, -1, -1):
        u = torch.rand(1, device=device)
        if acceptance_rate > u:
            tau = t
            break
        else:
            tau = t - 1
            # Update acceptance rate using capped branch divergence
            if tau >= 0:
                _, D_cap_pq, D_cap_qp = compute_capped_branch_divergences(
                    draft_tokens, draft_probs, target_probs_draft,
                    tau, ratios
                )
                acceptance_rate = D_cap_pq / D_cap_qp

    n_accepted = tau + 1

    # Perform resampling (now only one resampling step)
    if tau == gamma - 1:
        # Whole sequence accepted, get bonus token
        accepted_ids = torch.stack(draft_tokens).view(1, -1)
        bonus_token = sample_from_probs(target_probs_bonus).view(1, 1)
        new_ids = torch.cat([accepted_ids, bonus_token], dim=1)
    else:
        # compute resampling probability 
        # according to Equation (20): P^cap_res(x_t | X1:tau)
        if n_accepted > 0:
            accepted_ids = torch.stack(draft_tokens[:tau + 1]).view(1, -1)
        else:
            accepted_ids = None

        r_star, _, _ = compute_capped_branch_divergences(draft_tokens, 
                                                         draft_probs, target_probs_draft,
                                                         tau, ratios)

        q_prefix = torch.tensor(1.0, device=device)
        for i in range(tau):
            q_prefix = q_prefix * draft_probs[i][draft_tokens[i]]

        q_dist = draft_probs[tau + 1]

        resample_distribution = torch.maximum(q_dist * q_prefix * (r_star - 1), torch.tensor(0.0, device=device))
        resample_distribution_sum = resample_distribution.sum()
        resampled_token = sample_from_probs(resample_distribution / resample_distribution_sum).view(1, 1)

        if accepted_ids is not None:
            new_ids = torch.cat([accepted_ids, resampled_token], dim=1)
        else:
            new_ids = resampled_token

    return torch.cat([input_ids, new_ids], dim=1), n_accepted, gamma