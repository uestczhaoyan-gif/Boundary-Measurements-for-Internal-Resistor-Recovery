from __future__ import annotations

import torch
import torch.nn.functional as F


def support_mask_from_target(target: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.abs(target) > eps


def balanced_bce_with_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support_mask = support_mask_from_target(target)
    unchanged_mask = ~support_mask

    if support_mask.any():
        pos_loss = F.binary_cross_entropy_with_logits(
            logits[support_mask],
            torch.ones_like(logits[support_mask]),
        )
    else:
        pos_loss = logits.new_tensor(0.0)

    if unchanged_mask.any():
        neg_loss = F.binary_cross_entropy_with_logits(
            logits[unchanged_mask],
            torch.zeros_like(logits[unchanged_mask]),
        )
    else:
        neg_loss = logits.new_tensor(0.0)

    total = pos_weight * pos_loss + neg_weight * neg_loss
    return total, {
        "loss_score_pos": pos_loss,
        "loss_score_neg": neg_loss,
    }


def weighted_two_part_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    changed_weight: float,
    unchanged_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support_mask = support_mask_from_target(target)
    unchanged_mask = ~support_mask

    if support_mask.any():
        changed_loss = F.smooth_l1_loss(pred[support_mask], target[support_mask], beta=beta)
    else:
        changed_loss = pred.new_tensor(0.0)

    if unchanged_mask.any():
        unchanged_loss = F.smooth_l1_loss(pred[unchanged_mask], target[unchanged_mask], beta=beta)
    else:
        unchanged_loss = pred.new_tensor(0.0)

    total = changed_weight * changed_loss + unchanged_weight * unchanged_loss
    return total, {
        "loss_changed": changed_loss,
        "loss_unchanged": unchanged_loss,
    }


def weighted_two_part_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    changed_weight: float,
    unchanged_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support_mask = support_mask_from_target(target)
    unchanged_mask = ~support_mask

    if support_mask.any():
        changed_loss = F.mse_loss(pred[support_mask], target[support_mask])
    else:
        changed_loss = pred.new_tensor(0.0)

    if unchanged_mask.any():
        unchanged_loss = F.mse_loss(pred[unchanged_mask], target[unchanged_mask])
    else:
        unchanged_loss = pred.new_tensor(0.0)

    total = changed_weight * changed_loss + unchanged_weight * unchanged_loss
    return total, {
        "loss_changed": changed_loss,
        "loss_unchanged": unchanged_loss,
    }


def pairwise_ranking_hinge_loss(
    score_logits: torch.Tensor,
    target: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    support_mask = support_mask_from_target(target)
    sample_losses: list[torch.Tensor] = []

    for sample_scores, sample_mask in zip(score_logits, support_mask):
        pos_scores = sample_scores[sample_mask]
        neg_scores = sample_scores[~sample_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        pairwise_margin = margin - (pos_scores[:, None] - neg_scores[None, :])
        sample_losses.append(F.relu(pairwise_margin).mean())

    if not sample_losses:
        return score_logits.new_tensor(0.0)
    return torch.stack(sample_losses).mean()
