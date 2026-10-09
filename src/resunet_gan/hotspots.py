from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _ellipse_kernel(radius_mm: float, spacing_x: float, spacing_y: float, device: torch.device) -> torch.Tensor:
    radius_x = max(1, int(math.ceil(radius_mm / max(spacing_x, 1.0e-6))))
    radius_y = max(1, int(math.ceil(radius_mm / max(spacing_y, 1.0e-6))))
    y = torch.arange(-radius_y, radius_y + 1, device=device, dtype=torch.float32) * spacing_y
    x = torch.arange(-radius_x, radius_x + 1, device=device, dtype=torch.float32) * spacing_x
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return ((xx / radius_mm).pow(2) + (yy / radius_mm).pow(2) <= 1.0).to(torch.float32)


def _patch_score_map(
    values: torch.Tensor,
    roi: torch.Tensor,
    kernel_2d: torch.Tensor,
    heterogeneity_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = kernel_2d[None, None]
    padding = (kernel_2d.shape[0] // 2, kernel_2d.shape[1] // 2)
    support = F.conv2d(roi[:, None].float(), kernel, padding=padding)[:, 0]
    required = float(kernel_2d.sum().item())
    valid = support >= required - 0.5
    local_sum = F.conv2d((values * roi)[:, None], kernel, padding=padding)[:, 0]
    local_sq_sum = F.conv2d((values.square() * roi)[:, None], kernel, padding=padding)[:, 0]
    mean = local_sum / support.clamp_min(1.0)
    variance = (local_sq_sum / support.clamp_min(1.0) - mean.square()).clamp_min(0.0)
    score = mean - float(heterogeneity_penalty) * variance.sqrt()
    return score.masked_fill(~valid, -torch.inf), valid


def _rank_candidate_slices(
    score: torch.Tensor,
    valid: torch.Tensor,
    score_quantile: float,
    relative_threshold: float,
    maximum_slices: int,
    minimum_enhancement: float,
) -> list[tuple[int, float]]:
    ranked: list[tuple[int, float]] = []
    for z in range(score.shape[0]):
        values = score[z][valid[z] & torch.isfinite(score[z])]
        if values.numel():
            ranked.append((z, float(torch.quantile(values, score_quantile).item())))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= minimum_enhancement:
        return []
    threshold = max(minimum_enhancement, ranked[0][1] * relative_threshold)
    return [item for item in ranked if item[1] >= threshold][:maximum_slices]


@torch.no_grad()
def select_roi_hotspots_with_metadata(
    sub1: torch.Tensor,
    roi: torch.Tensor,
    spacing_xyz: torch.Tensor,
    count: int = 3,
    radius_mm: float = 3.0,
    minimum_radius_mm: float = 1.5,
    radius_step_mm: float = 0.5,
    minimum_voxels: int = 9,
    slice_score_quantile: float = 0.90,
    slice_relative_threshold: float = 0.70,
    maximum_candidate_slices: int = 3,
    candidate_top_fraction: float = 0.25,
    minimum_mean_enhancement: float = 0.0,
    heterogeneity_penalty: float = 0.10,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[torch.Tensor, list[list[dict[str, float | int]]]]:
    """Select typical-slice, enhancing, non-overlapping 2D SUB1 hotspots.

    Selection first ranks axial slices by robust local SUB1 enhancement. It tries
    to place all requested patches on the best slice, then progressively admits
    up to ``maximum_candidate_slices`` strongly enhancing slices only when needed.
    """
    if sub1.ndim != 5 or sub1.shape[1] != 1:
        raise ValueError("sub1 must have shape [B,1,D,H,W]")
    if roi.ndim != 5 or roi.shape[1] != 1 or roi.shape[0] != sub1.shape[0]:
        raise ValueError("roi must have shape [B,1,D,H,W]")
    if spacing_xyz.ndim != 2 or spacing_xyz.shape != (sub1.shape[0], 3):
        raise ValueError("spacing_xyz must have shape [B,3]")
    if count < 1 or radius_mm <= 0 or minimum_radius_mm <= 0 or radius_step_mm <= 0:
        raise ValueError("hotspot count and radii must be positive")
    if minimum_voxels < 3:
        raise ValueError("minimum_voxels must be at least 3")
    if not 0.0 < slice_score_quantile <= 1.0 or not 0.0 < candidate_top_fraction <= 1.0:
        raise ValueError("hotspot quantiles/fractions must be in (0,1]")
    if not 0.0 < slice_relative_threshold <= 1.0 or maximum_candidate_slices < 1:
        raise ValueError("slice_relative_threshold must be in (0,1] and maximum_candidate_slices positive")
    clip_low, clip_high = clip_percentiles
    if not 0.0 <= clip_low < clip_high <= 100.0:
        raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100")

    radii: list[float] = []
    current = radius_mm
    while current >= minimum_radius_mm - 1.0e-8:
        radii.append(current)
        current -= radius_step_mm
    if not radii:
        radii = [minimum_radius_mm]

    output = torch.zeros(
        (sub1.shape[0], count, *sub1.shape[-3:]),
        dtype=sub1.dtype,
        device=sub1.device,
    )
    all_metadata: list[list[dict[str, float | int]]] = []
    for batch_index in range(sub1.shape[0]):
        lesion_roi = roi[batch_index, 0].to(dtype=torch.bool)
        finite = lesion_roi & torch.isfinite(sub1[batch_index, 0])
        if not torch.any(finite):
            all_metadata.append([])
            continue
        values = torch.nan_to_num(sub1[batch_index, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        finite_values = values[finite]
        limits = torch.quantile(
            finite_values,
            values.new_tensor([clip_low / 100.0, clip_high / 100.0]),
        )
        values = values.clamp(float(limits[0].item()), float(limits[1].item()))
        spacing_x, spacing_y = (float(value) for value in spacing_xyz[batch_index, :2].tolist())
        best_masks: list[torch.Tensor] = []
        best_metadata: list[dict[str, float | int]] = []
        for candidate_radius in radii:
            kernel_2d = _ellipse_kernel(candidate_radius, spacing_x, spacing_y, sub1.device)
            if int(kernel_2d.sum().item()) < minimum_voxels:
                continue
            score, valid_centers = _patch_score_map(values, lesion_roi, kernel_2d, heterogeneity_penalty)
            ranked_slices = _rank_candidate_slices(
                score,
                valid_centers,
                slice_score_quantile,
                slice_relative_threshold,
                maximum_candidate_slices,
                minimum_mean_enhancement,
            )
            if not ranked_slices:
                continue
            slice_rank = {z: rank + 1 for rank, (z, _) in enumerate(ranked_slices)}
            slice_scores = dict(ranked_slices)
            radius_best_masks: list[torch.Tensor] = []
            radius_best_metadata: list[dict[str, float | int]] = []
            for admitted_count in range(1, len(ranked_slices) + 1):
                admitted = {z for z, _ in ranked_slices[:admitted_count]}
                admitted_mask = torch.zeros_like(valid_centers)
                for z in admitted:
                    admitted_mask[z] = True
                candidate_values = score[valid_centers & admitted_mask & torch.isfinite(score)]
                if not candidate_values.numel():
                    continue
                preferred_floor = float(
                    torch.quantile(candidate_values, 1.0 - candidate_top_fraction).item()
                )
                occupied = torch.zeros_like(lesion_roi)
                selected_masks: list[torch.Tensor] = []
                selected_metadata: list[dict[str, float | int]] = []
                kernel = kernel_2d[None, None]
                padding = (kernel_2d.shape[0] // 2, kernel_2d.shape[1] // 2)
                required = float(kernel_2d.sum().item())
                for _ in range(count):
                    available_roi = lesion_roi & (~occupied)
                    support = F.conv2d(available_roi[:, None].float(), kernel, padding=padding)[:, 0]
                    current_valid = support >= required - 0.5
                    candidates = (
                        current_valid
                        & admitted_mask
                        & torch.isfinite(score)
                        & (score >= max(preferred_floor, minimum_mean_enhancement))
                    )
                    if not torch.any(candidates):
                        candidates = (
                            current_valid
                            & admitted_mask
                            & torch.isfinite(score)
                            & (score > minimum_mean_enhancement)
                        )
                    if not torch.any(candidates):
                        break
                    candidate_score = score.masked_fill(~candidates, -torch.inf)
                    flat_index = int(torch.argmax(candidate_score).item())
                    height, width = candidate_score.shape[-2:]
                    z = flat_index // (height * width)
                    remainder = flat_index % (height * width)
                    y, x = divmod(remainder, width)
                    ry, rx = kernel_2d.shape[0] // 2, kernel_2d.shape[1] // 2
                    mask = torch.zeros_like(lesion_roi)
                    mask[z, y - ry : y + ry + 1, x - rx : x + rx + 1] = kernel_2d.to(dtype=torch.bool)
                    mask &= lesion_roi
                    if int(mask.sum().item()) < minimum_voxels or torch.any(mask & occupied):
                        break
                    occupied |= mask
                    selected_masks.append(mask)
                    selected_metadata.append(
                        {
                            "center_z": z,
                            "center_y": y,
                            "center_x": x,
                            "radius_mm": float(candidate_radius),
                            "voxel_count": int(mask.sum().item()),
                            "selection_score": float(score[z, y, x].item()),
                            "sub1_mean": float(values[mask].mean().item()),
                            "sub1_std": float(values[mask].std(unbiased=False).item()),
                            "slice_rank": int(slice_rank[z]),
                            "slice_score": float(slice_scores[z]),
                            "admitted_slice_count": admitted_count,
                        }
                    )
                if len(selected_masks) > len(radius_best_masks):
                    radius_best_masks = selected_masks
                    radius_best_metadata = selected_metadata
                if len(selected_masks) == count:
                    radius_best_masks = selected_masks
                    radius_best_metadata = selected_metadata
                    break
            if len(radius_best_masks) > len(best_masks):
                best_masks = radius_best_masks
                best_metadata = radius_best_metadata
            if len(radius_best_masks) == count:
                best_masks = radius_best_masks
                best_metadata = radius_best_metadata
                break
        ordered = sorted(
            zip(best_masks, best_metadata),
            key=lambda pair: (int(pair[1]["slice_rank"]), -float(pair[1]["selection_score"])),
        )
        best_masks = [pair[0] for pair in ordered]
        best_metadata = [pair[1] for pair in ordered]
        for hotspot_index, mask in enumerate(best_masks[:count]):
            output[batch_index, hotspot_index] = mask.to(dtype=sub1.dtype)
        all_metadata.append(best_metadata[:count])
    return output, all_metadata


@torch.no_grad()
def select_roi_hotspots(
    sub1: torch.Tensor,
    roi: torch.Tensor,
    spacing_xyz: torch.Tensor,
    **kwargs: float | int | tuple[float, float],
) -> torch.Tensor:
    masks, _ = select_roi_hotspots_with_metadata(sub1, roi, spacing_xyz, **kwargs)
    return masks


def hotspot_means(volume: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,K,C] hotspot means and [B,K] validity."""
    if volume.ndim != 5 or masks.ndim != 5 or volume.shape[0] != masks.shape[0]:
        raise ValueError("volume and hotspot masks must have shapes [B,C,D,H,W] and [B,K,D,H,W]")
    masks = masks.to(dtype=volume.dtype)
    denominator = masks.sum(dim=(-3, -2, -1))
    numerator = torch.einsum("bcdhw,bkdhw->bkc", volume, masks)
    return numerator / denominator.unsqueeze(-1).clamp_min(1.0), denominator > 0


def kinetic_types(curves: torch.Tensor, washout_threshold: float = -0.10, persistent_threshold: float = 0.10) -> torch.Tensor:
    """Classify delayed kinetics as 0=persistent, 1=plateau, 2=washout."""
    relative_change = (curves[..., -1] - curves[..., 0]) / curves[..., 0].abs().clamp_min(1.0e-3)
    return torch.where(
        relative_change < washout_threshold,
        torch.full_like(relative_change, 2, dtype=torch.long),
        torch.where(
            relative_change > persistent_threshold,
            torch.zeros_like(relative_change, dtype=torch.long),
            torch.ones_like(relative_change, dtype=torch.long),
        ),
    )
