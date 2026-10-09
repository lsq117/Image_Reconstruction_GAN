from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .hotspots import hotspot_means


def _as_list(value: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    return value if isinstance(value, list) else [value]


def _flatten_features(value: torch.Tensor | list) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    features: list[torch.Tensor] = []
    for item in value:
        features.extend(_flatten_features(item))
    return features


def _expand_mask(mask: torch.Tensor, channels: int) -> torch.Tensor:
    if mask.shape[1] == 1 and channels != 1:
        return mask.expand(-1, channels, -1, -1, -1)
    return mask


def _masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = _expand_mask(mask.to(dtype=value.dtype), value.shape[1])
    numerator = (value * mask).flatten(start_dim=2).sum(dim=2)
    denominator = mask.flatten(start_dim=2).sum(dim=2)
    valid = denominator > 0
    if not torch.any(valid):
        return value.new_tensor(0.0)
    return (numerator / denominator.clamp_min(1.0))[valid].mean()


def _dilate_mask(mask: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    if len(kernel_size) != 3 or any(size < 1 or size % 2 == 0 for size in kernel_size):
        raise ValueError("Mask dilation kernel must contain three positive odd integers")
    padding = tuple(size // 2 for size in kernel_size)
    return F.max_pool3d(mask.to(dtype=torch.float32), kernel_size=kernel_size, stride=1, padding=padding) > 0.5


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1.0e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = torch.sqrt((pred - target).pow(2) + self.eps * self.eps)
        return _masked_mean_per_sample(loss, mask)


class BalancedROICharbonnierLoss(nn.Module):
    """Give low, middle, and high enhancement equal weight per case and phase."""

    def __init__(self, eps: float = 1.0e-3, quantiles: tuple[float, float] = (0.33, 0.67)) -> None:
        super().__init__()
        self.eps = eps
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        error = torch.sqrt((pred - target).pow(2) + self.eps * self.eps)
        roi = _expand_mask(roi.to(dtype=torch.bool), pred.shape[1])
        losses: list[torch.Tensor] = []
        for batch_index in range(pred.shape[0]):
            for channel_index in range(pred.shape[1]):
                mask = roi[batch_index, channel_index]
                values = target[batch_index, channel_index][mask]
                if values.numel() == 0:
                    continue
                q_low, q_high = torch.quantile(
                    values.detach().float(),
                    target.new_tensor(self.quantiles, dtype=torch.float32),
                ).to(dtype=target.dtype)
                target_channel = target[batch_index, channel_index]
                region_masks = (
                    mask & (target_channel <= q_low),
                    mask & (target_channel > q_low) & (target_channel < q_high),
                    mask & (target_channel >= q_high),
                )
                region_losses = [
                    error[batch_index, channel_index][region_mask].mean()
                    for region_mask in region_masks
                    if torch.any(region_mask)
                ]
                if region_losses:
                    losses.append(torch.stack(region_losses).mean())
        return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


class NearZeroBackgroundLoss(nn.Module):
    """Prevent diffuse false enhancement where the real non-lesion SUB is near zero."""

    def __init__(self, eps: float = 1.0e-3, threshold: float = 0.25) -> None:
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        foreground: torch.Tensor | None = None,
    ) -> torch.Tensor:
        roi = _expand_mask(roi.to(dtype=torch.bool), pred.shape[1])
        mask = (~roi) & (target.detach().abs() <= self.threshold)
        if foreground is not None:
            mask = mask & _expand_mask(foreground.to(dtype=torch.bool), pred.shape[1])
        error = torch.sqrt((pred - target).pow(2) + self.eps * self.eps)
        return _masked_mean_per_sample(error, mask)


class PositiveBackgroundLeakLoss(nn.Module):
    """Penalize only spurious positive SUB outside the phase-specific lesion ROI."""

    def __init__(self, threshold: float = 0.25, tolerance: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold
        self.tolerance = tolerance

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        foreground: torch.Tensor | None = None,
    ) -> torch.Tensor:
        roi = _expand_mask(roi.to(dtype=torch.bool), pred.shape[1])
        mask = (~roi) & (target.detach().abs() <= self.threshold)
        if foreground is not None:
            mask = mask & _expand_mask(foreground.to(dtype=torch.bool), pred.shape[1])
        return _masked_mean_per_sample(F.relu(pred - target - self.tolerance), mask)


class StaticAnchorLoss(nn.Module):
    """Keep truly static voxels close to the configured enhancement anchor."""

    def __init__(self, delta_threshold: float = 0.25, tolerance: float = 0.05) -> None:
        super().__init__()
        self.delta_threshold = delta_threshold
        self.tolerance = tolerance

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor | None,
        foreground: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor is None:
            return pred.new_tensor(0.0)
        anchor = _expand_mask(anchor, pred.shape[1])
        mask = (target.detach() - anchor.detach()).abs() <= self.delta_threshold
        if foreground is not None:
            mask = mask & _expand_mask(foreground.to(dtype=torch.bool), pred.shape[1])
        predicted_change = (pred - anchor).abs()
        return _masked_mean_per_sample(F.relu(predicted_change - self.tolerance), mask)


class PerilesionalPositiveLeakLoss(nn.Module):
    """Suppress positive enhancement leaking immediately beyond each phase's lesion boundary."""

    def __init__(
        self,
        kernel_size: tuple[int, int, int] = (3, 7, 7),
        target_threshold: float = 0.25,
        tolerance: float = 0.05,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.target_threshold = target_threshold
        self.tolerance = tolerance

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        foreground: torch.Tensor | None = None,
        ring_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        roi = _expand_mask(roi.to(dtype=torch.bool), pred.shape[1])
        if ring_mask is None:
            ring = _dilate_mask(roi, self.kernel_size) & (~roi)
        else:
            ring = _expand_mask(ring_mask.to(dtype=torch.bool), pred.shape[1]) & (~roi)
        mask = ring & (target.detach().abs() <= self.target_threshold)
        if foreground is not None:
            mask = mask & _expand_mask(foreground.to(dtype=torch.bool), pred.shape[1])
        return _masked_mean_per_sample(F.relu(pred - target - self.tolerance), mask)


class SoftEnhancementSupportLoss(nn.Module):
    """Class-balanced soft Tversky loss for enhancement support."""

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.25, 0.5, 1.0),
        temperature: float = 0.15,
        false_positive_weight: float = 0.7,
        false_negative_weight: float = 0.3,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if not thresholds:
            raise ValueError("At least one enhancement-support threshold is required")
        if temperature <= 0:
            raise ValueError("Enhancement-support temperature must be positive")
        if false_positive_weight < 0 or false_negative_weight < 0:
            raise ValueError("Enhancement-support Tversky weights must be non-negative")
        if false_positive_weight + false_negative_weight <= 0:
            raise ValueError("At least one enhancement-support Tversky weight must be positive")
        if smooth <= 0:
            raise ValueError("Enhancement-support smoothing must be positive")
        self.thresholds = thresholds
        self.temperature = temperature
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        foreground: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = torch.ones_like(pred, dtype=torch.bool) if foreground is None else _expand_mask(
            foreground.to(dtype=torch.bool), pred.shape[1]
        )
        mask_float = mask.to(dtype=pred.dtype)
        losses: list[torch.Tensor] = []
        for threshold in self.thresholds:
            predicted_support = torch.sigmoid((pred - threshold) / self.temperature)
            target_support = (target.detach() > threshold).to(dtype=pred.dtype)
            predicted_support = (predicted_support * mask_float).flatten(start_dim=2)
            target_support = (target_support * mask_float).flatten(start_dim=2)
            true_positive = (predicted_support * target_support).sum(dim=2)
            false_positive = (predicted_support * (1.0 - target_support)).sum(dim=2)
            false_negative = ((1.0 - predicted_support) * target_support).sum(dim=2)
            score = (true_positive + self.smooth) / (
                true_positive
                + self.false_positive_weight * false_positive
                + self.false_negative_weight * false_negative
                + self.smooth
            )
            valid = mask_float.flatten(start_dim=2).sum(dim=2) > 0
            if torch.any(valid):
                losses.append((1.0 - score)[valid].mean())
        return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


class SSIM3DLoss(nn.Module):
    def __init__(
        self,
        window_size: int | tuple[int, int, int] = (3, 7, 7),
        data_range: float = 12.0,
        clip_value: float | None = 6.0,
    ) -> None:
        super().__init__()
        self.window_size = (window_size,) * 3 if isinstance(window_size, int) else window_size
        if len(self.window_size) != 3 or any(size < 1 or size % 2 == 0 for size in self.window_size):
            raise ValueError("SSIM window dimensions must be positive odd integers")
        self.data_range = data_range
        self.clip_value = clip_value

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.clip_value is not None:
            pred = pred.clamp(-self.clip_value, self.clip_value)
            target = target.clamp(-self.clip_value, self.clip_value)
        channels = pred.shape[1]
        wd, wh, ww = self.window_size
        kernel = torch.ones((channels, 1, wd, wh, ww), device=pred.device, dtype=pred.dtype)
        kernel = kernel / float(wd * wh * ww)
        pad = (ww // 2, ww // 2, wh // 2, wh // 2, wd // 2, wd // 2)

        def local_mean(value: torch.Tensor) -> torch.Tensor:
            return F.conv3d(F.pad(value, pad, mode="replicate"), kernel, groups=channels)

        if mask is None:
            mu_x = local_mean(pred)
            mu_y = local_mean(target)
            second_x = local_mean(pred * pred)
            second_y = local_mean(target * target)
            cross_xy = local_mean(pred * target)
        else:
            expanded_mask = _expand_mask(mask.to(dtype=pred.dtype), channels)

            def masked_local_mean(value: torch.Tensor) -> torch.Tensor:
                numerator = F.conv3d(
                    F.pad(value * expanded_mask, pad, mode="constant", value=0.0),
                    kernel,
                    groups=channels,
                )
                denominator = F.conv3d(
                    F.pad(expanded_mask, pad, mode="constant", value=0.0),
                    kernel,
                    groups=channels,
                )
                return numerator / denominator.clamp_min(1.0e-6)

            mu_x = masked_local_mean(pred)
            mu_y = masked_local_mean(target)
            second_x = masked_local_mean(pred * pred)
            second_y = masked_local_mean(target * target)
            cross_xy = masked_local_mean(pred * target)
        sigma_x = (second_x - mu_x.pow(2)).clamp_min(0.0)
        sigma_y = (second_y - mu_y.pow(2)).clamp_min(0.0)
        sigma_xy = cross_xy - mu_x * mu_y
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        )
        return 1.0 - _masked_mean_per_sample(ssim.clamp(-1.0, 1.0), mask)


class GradientLoss3D(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for dim in (-3, -2, -1):
            pred_diff = pred.diff(dim=dim)
            target_diff = target.diff(dim=dim)
            losses.append(torch.mean(torch.abs(pred_diff - target_diff)))
        return sum(losses) / len(losses)


class MaskedGradientLoss3D(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _expand_mask(mask.to(dtype=pred.dtype), pred.shape[1])
        interior_losses = []
        boundary_losses = []
        for dim in (-3, -2, -1):
            pred_diff = pred.diff(dim=dim)
            target_diff = target.diff(dim=dim)
            mask_head = mask.narrow(dim, 0, mask.shape[dim] - 1)
            mask_tail = mask.narrow(dim, 1, mask.shape[dim] - 1)
            error = torch.abs(pred_diff - target_diff)
            interior_losses.append(_masked_mean_per_sample(error, mask_head * mask_tail))
            boundary_losses.append(_masked_mean_per_sample(error, torch.abs(mask_head - mask_tail)))
        return sum(interior_losses) / len(interior_losses), sum(boundary_losses) / len(boundary_losses)


class ROICurveLoss(nn.Module):
    """Preserve the phase-wise mean enhancement curve inside the lesion ROI."""

    def __init__(self, eps: float = 1.0e-3) -> None:
        super().__init__()
        self.eps = eps

    @staticmethod
    def _roi_curve(x: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        roi = roi.to(dtype=x.dtype)
        denominator = roi.sum(dim=(-3, -2, -1)).clamp_min(1.0)
        return (x * roi).sum(dim=(-3, -2, -1)) / denominator

    def forward(self, pred: torch.Tensor, target: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        difference = self._roi_curve(pred, roi) - self._roi_curve(target, roi)
        return torch.sqrt(difference.pow(2) + self.eps * self.eps).mean()


class ROIPercentEnhancementCurveLoss(nn.Module):
    """Match lesion mean SUB/precontrast curves without changing the network inputs."""

    def __init__(self, eps: float = 1.0e-3, precontrast_floor: float = 0.25) -> None:
        super().__init__()
        self.eps = eps
        self.precontrast_floor = precontrast_floor

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        precontrast_scaled: torch.Tensor | None,
    ) -> torch.Tensor:
        if precontrast_scaled is None:
            return pred.new_tensor(0.0)
        channels = pred.shape[1]
        expanded_precontrast = precontrast_scaled.expand(-1, channels, -1, -1, -1)
        denominator = ROICurveLoss._roi_curve(expanded_precontrast, roi).abs().clamp_min(self.precontrast_floor)
        pred_pe = ROICurveLoss._roi_curve(pred, roi) / denominator
        target_pe = ROICurveLoss._roi_curve(target, roi) / denominator
        difference = pred_pe - target_pe
        return torch.sqrt(difference.pow(2) + self.eps * self.eps).mean()


class TemporalDifferenceLoss(nn.Module):
    """Match enhancement changes between adjacent post-contrast phases."""

    def __init__(self, eps: float = 1.0e-3) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor is not None:
            pred = torch.cat([anchor, pred], dim=1)
            target = torch.cat([anchor, target], dim=1)
        if pred.shape[1] < 2:
            return pred.new_tensor(0.0)
        return self.charbonnier(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], roi)


class MultiHotspotKineticLoss(nn.Module):
    """Supervise fixed ROI hotspots and emphasize true negative phase changes."""

    def __init__(
        self,
        eps: float = 1.0e-3,
        washout_delta_threshold: float = 0.05,
        washout_delta_weight: float = 2.0,
        worst_hotspot_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.washout_delta_threshold = washout_delta_threshold
        self.washout_delta_weight = washout_delta_weight
        self.worst_hotspot_weight = worst_hotspot_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor | None,
        hotspot_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor is None or hotspot_masks is None:
            zero = pred.new_tensor(0.0)
            return zero, zero
        pred_means, valid = hotspot_means(pred, hotspot_masks)
        target_means, _ = hotspot_means(target, hotspot_masks)
        anchor_means, _ = hotspot_means(anchor, hotspot_masks)
        pred_curve = torch.cat([anchor_means, pred_means], dim=-1)
        target_curve = torch.cat([anchor_means, target_means], dim=-1)

        relative_late_change = (target_curve[..., -1] - target_curve[..., 0]) / target_curve[
            ..., 0
        ].abs().clamp_min(1.0e-3)
        worst_index = relative_late_change.masked_fill(~valid, torch.inf).argmin(dim=1)
        hotspot_weight = torch.ones_like(relative_late_change)
        hotspot_weight.scatter_(1, worst_index[:, None], self.worst_hotspot_weight)
        hotspot_weight = hotspot_weight * valid.to(dtype=pred.dtype)

        curve_error = torch.sqrt((pred_means - target_means).pow(2) + self.eps * self.eps).mean(dim=-1)
        curve_loss = (curve_error * hotspot_weight).sum() / hotspot_weight.sum().clamp_min(1.0)

        pred_delta = pred_curve[..., 1:] - pred_curve[..., :-1]
        target_delta = target_curve[..., 1:] - target_curve[..., :-1]
        delta_weight = torch.where(
            target_delta < -self.washout_delta_threshold,
            target_delta.new_tensor(self.washout_delta_weight),
            target_delta.new_tensor(1.0),
        )
        delta_weight = delta_weight * hotspot_weight.unsqueeze(-1)
        temporal_error = torch.sqrt((pred_delta - target_delta).pow(2) + self.eps * self.eps)
        temporal_loss = (temporal_error * delta_weight).sum() / delta_weight.sum().clamp_min(1.0)
        return curve_loss, temporal_loss


class MultiHotspotTrendLoss(nn.Module):
    """Keep each hotspot on the correct persistent/plateau/washout side."""

    def __init__(
        self,
        washout_threshold: float = -0.10,
        persistent_threshold: float = 0.10,
        margin: float = 0.02,
        worst_hotspot_weight: float = 2.0,
    ) -> None:
        super().__init__()
        if washout_threshold >= persistent_threshold:
            raise ValueError("washout_threshold must be lower than persistent_threshold")
        if margin < 0 or 2.0 * margin >= persistent_threshold - washout_threshold:
            raise ValueError("trend margin must be non-negative and leave a non-empty plateau interval")
        self.washout_threshold = washout_threshold
        self.persistent_threshold = persistent_threshold
        self.margin = margin
        self.worst_hotspot_weight = worst_hotspot_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor | None,
        hotspot_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        if anchor is None or hotspot_masks is None:
            return pred.new_tensor(0.0)
        pred_means, valid = hotspot_means(pred, hotspot_masks)
        target_means, _ = hotspot_means(target, hotspot_masks)
        anchor_means, _ = hotspot_means(anchor, hotspot_masks)
        denominator = anchor_means[..., 0].abs().clamp_min(1.0e-3)
        pred_change = (pred_means[..., -1] - anchor_means[..., 0]) / denominator
        target_change = (target_means[..., -1] - anchor_means[..., 0]) / denominator

        washout = target_change < self.washout_threshold
        persistent = target_change > self.persistent_threshold
        washout_error = F.relu(pred_change - (self.washout_threshold - self.margin))
        persistent_error = F.relu((self.persistent_threshold + self.margin) - pred_change)
        plateau_error = F.relu((self.washout_threshold + self.margin) - pred_change) + F.relu(
            pred_change - (self.persistent_threshold - self.margin)
        )
        trend_error = torch.where(washout, washout_error, torch.where(persistent, persistent_error, plateau_error))

        worst_index = target_change.masked_fill(~valid, torch.inf).argmin(dim=1)
        hotspot_weight = torch.ones_like(target_change)
        hotspot_weight.scatter_(1, worst_index[:, None], self.worst_hotspot_weight)
        hotspot_weight = hotspot_weight * valid.to(dtype=pred.dtype)
        return (trend_error * hotspot_weight).sum() / hotspot_weight.sum().clamp_min(1.0)


class MultiHotspotPercentEnhancementLoss(nn.Module):
    """Match hotspot percent-enhancement levels and phase-to-phase changes."""

    def __init__(
        self,
        eps: float = 1.0e-3,
        precontrast_floor: float = 0.25,
        washout_delta_threshold: float = 0.05,
        washout_delta_weight: float = 2.0,
        worst_hotspot_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.precontrast_floor = precontrast_floor
        self.washout_delta_threshold = washout_delta_threshold
        self.washout_delta_weight = washout_delta_weight
        self.worst_hotspot_weight = worst_hotspot_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor | None,
        precontrast_scaled: torch.Tensor | None,
        hotspot_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor is None or precontrast_scaled is None or hotspot_masks is None:
            zero = pred.new_tensor(0.0)
            return zero, zero
        pred_means, valid = hotspot_means(pred, hotspot_masks)
        target_means, _ = hotspot_means(target, hotspot_masks)
        anchor_means, _ = hotspot_means(anchor, hotspot_masks)
        precontrast_means, _ = hotspot_means(precontrast_scaled, hotspot_masks)
        denominator = precontrast_means.abs().clamp_min(self.precontrast_floor)
        pred_curve = torch.cat([anchor_means / denominator, pred_means / denominator], dim=-1)
        target_curve = torch.cat([anchor_means / denominator, target_means / denominator], dim=-1)

        relative_late_change = (target_curve[..., -1] - target_curve[..., 0]) / target_curve[
            ..., 0
        ].abs().clamp_min(1.0e-3)
        worst_index = relative_late_change.masked_fill(~valid, torch.inf).argmin(dim=1)
        hotspot_weight = torch.ones_like(relative_late_change)
        hotspot_weight.scatter_(1, worst_index[:, None], self.worst_hotspot_weight)
        hotspot_weight = hotspot_weight * valid.to(dtype=pred.dtype)

        curve_error = torch.sqrt((pred_curve[..., 1:] - target_curve[..., 1:]).pow(2) + self.eps**2).mean(dim=-1)
        curve_loss = (curve_error * hotspot_weight).sum() / hotspot_weight.sum().clamp_min(1.0)
        pred_delta = pred_curve[..., 1:] - pred_curve[..., :-1]
        target_delta = target_curve[..., 1:] - target_curve[..., :-1]
        delta_weight = torch.where(
            target_delta < -self.washout_delta_threshold,
            target_delta.new_tensor(self.washout_delta_weight),
            target_delta.new_tensor(1.0),
        )
        delta_weight = delta_weight * hotspot_weight.unsqueeze(-1)
        temporal_error = torch.sqrt((pred_delta - target_delta).pow(2) + self.eps**2)
        temporal_loss = (temporal_error * delta_weight).sum() / delta_weight.sum().clamp_min(1.0)
        return curve_loss, temporal_loss


class ROIKineticDistributionLoss(nn.Module):
    """Match whole-lesion PE-change distributions without relying on fixed points."""

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
        precontrast_floor: float = 0.25,
        minimum_anchor_pe: float = 0.05,
        washout_threshold: float = -0.10,
        persistent_threshold: float = 0.10,
        type_temperature: float = 0.03,
        statistics_weight: float = 0.50,
        type_fraction_weight: float = 0.50,
    ) -> None:
        super().__init__()
        if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
            raise ValueError("kinetic_distribution_quantiles must be in (0,1)")
        if precontrast_floor <= 0 or minimum_anchor_pe <= 0 or type_temperature <= 0:
            raise ValueError("Kinetic distribution floors and temperature must be positive")
        if washout_threshold >= persistent_threshold:
            raise ValueError("Kinetic distribution washout threshold must be below persistent threshold")
        if statistics_weight < 0 or type_fraction_weight < 0:
            raise ValueError("Kinetic distribution component weights must be non-negative")
        self.register_buffer("quantiles", torch.tensor(quantiles))
        self.precontrast_floor = precontrast_floor
        self.minimum_anchor_pe = minimum_anchor_pe
        self.washout_threshold = washout_threshold
        self.persistent_threshold = persistent_threshold
        self.type_temperature = type_temperature
        self.statistics_weight = statistics_weight
        self.type_fraction_weight = type_fraction_weight

    def _soft_type_fractions(self, relative_change: torch.Tensor) -> torch.Tensor:
        washout = torch.sigmoid((self.washout_threshold - relative_change) / self.type_temperature)
        persistent = torch.sigmoid((relative_change - self.persistent_threshold) / self.type_temperature)
        plateau = (1.0 - washout - persistent).clamp_min(0.0)
        return torch.stack((persistent.mean(), plateau.mean(), washout.mean()))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        anchor: torch.Tensor | None,
        precontrast_scaled: torch.Tensor | None,
    ) -> torch.Tensor:
        if anchor is None or precontrast_scaled is None:
            return pred.new_tensor(0.0)
        phase_roi = _expand_mask(roi.to(dtype=torch.bool), pred.shape[1])
        lesion = torch.any(phase_roi, dim=1)
        denominator = precontrast_scaled.abs().clamp_min(self.precontrast_floor)
        anchor_pe = anchor / denominator
        pred_curve = torch.cat((anchor_pe, pred / denominator), dim=1)
        target_curve = torch.cat((anchor_pe, target / denominator), dim=1)
        pred_delta = pred_curve[:, 1:] - pred_curve[:, :-1]
        target_delta = target_curve[:, 1:] - target_curve[:, :-1]
        pred_features = torch.cat((pred_delta, pred_curve[:, -1:] - pred_curve[:, :1]), dim=1)
        target_features = torch.cat((target_delta, target_curve[:, -1:] - target_curve[:, :1]), dim=1)
        quantiles = self.quantiles.to(device=pred.device, dtype=torch.float32)
        losses: list[torch.Tensor] = []
        for batch_index in range(pred.shape[0]):
            current_mask = lesion[batch_index]
            if not torch.any(current_mask):
                continue
            feature_losses: list[torch.Tensor] = []
            for channel_index in range(pred_features.shape[1]):
                pred_values = pred_features[batch_index, channel_index][current_mask].float()
                target_values = target_features[batch_index, channel_index][current_mask].float()
                quantile_loss = F.smooth_l1_loss(
                    torch.quantile(pred_values, quantiles),
                    torch.quantile(target_values, quantiles),
                )
                statistics_loss = F.smooth_l1_loss(pred_values.mean(), target_values.mean()) + F.smooth_l1_loss(
                    pred_values.std(unbiased=False), target_values.std(unbiased=False)
                )
                feature_losses.append(quantile_loss + self.statistics_weight * statistics_loss)

            anchor_values = anchor_pe[batch_index, 0][current_mask].float()
            valid_kinetics = anchor_values.abs() >= self.minimum_anchor_pe
            type_loss = pred.new_tensor(0.0)
            if torch.any(valid_kinetics):
                pred_late = pred_features[batch_index, -1][current_mask].float()[valid_kinetics]
                target_late = target_features[batch_index, -1][current_mask].float()[valid_kinetics]
                anchor_valid = anchor_values[valid_kinetics].abs().clamp_min(self.minimum_anchor_pe)
                pred_relative = pred_late / anchor_valid
                target_relative = target_late / anchor_valid
                type_loss = F.smooth_l1_loss(
                    self._soft_type_fractions(pred_relative),
                    self._soft_type_fractions(target_relative),
                )
            losses.append(torch.stack(feature_losses).mean() + self.type_fraction_weight * type_loss)
        return torch.stack(losses).mean().to(dtype=pred.dtype) if losses else pred.new_tensor(0.0)


class IntensityDistributionLoss(nn.Module):
    """Match per-phase brightness, contrast, and robust intensity tails."""

    def __init__(self, stride: tuple[int, int, int] = (2, 4, 4)) -> None:
        super().__init__()
        self.stride = stride
        self.register_buffer("quantiles", torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_float = pred.float()
        target_float = target.float()
        if mask is not None:
            mask = _expand_mask(mask.to(dtype=torch.bool), pred.shape[1])
            sd, sh, sw = self.stride
            sampled_pred = pred_float[..., ::sd, ::sh, ::sw]
            sampled_target = target_float[..., ::sd, ::sh, ::sw]
            sampled_mask = mask[..., ::sd, ::sh, ::sw]
            losses: list[torch.Tensor] = []
            quantiles = self.quantiles.to(device=pred.device, dtype=pred_float.dtype)
            for batch_index in range(pred.shape[0]):
                for channel_index in range(pred.shape[1]):
                    full_mask = mask[batch_index, channel_index]
                    small_mask = sampled_mask[batch_index, channel_index]
                    if not torch.any(full_mask) or not torch.any(small_mask):
                        continue
                    pred_values = pred_float[batch_index, channel_index][full_mask]
                    target_values = target_float[batch_index, channel_index][full_mask]
                    pred_sample = sampled_pred[batch_index, channel_index][small_mask]
                    target_sample = sampled_target[batch_index, channel_index][small_mask]
                    losses.append(
                        F.smooth_l1_loss(pred_values.mean(), target_values.mean())
                        + F.smooth_l1_loss(
                            pred_values.std(unbiased=False),
                            target_values.std(unbiased=False),
                        )
                        + F.smooth_l1_loss(
                            torch.quantile(pred_sample, quantiles),
                            torch.quantile(target_sample, quantiles),
                        )
                    )
            return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)

        dims = (-3, -2, -1)
        mean_loss = F.smooth_l1_loss(pred_float.mean(dims), target_float.mean(dims))
        std_loss = F.smooth_l1_loss(
            pred_float.std(dims, unbiased=False),
            target_float.std(dims, unbiased=False),
        )
        sd, sh, sw = self.stride
        pred_sample = pred_float[..., ::sd, ::sh, ::sw].flatten(2)
        target_sample = target_float[..., ::sd, ::sh, ::sw].flatten(2)
        quantiles = self.quantiles.to(device=pred.device, dtype=pred_float.dtype)
        quantile_loss = F.smooth_l1_loss(
            torch.quantile(pred_sample, quantiles, dim=2),
            torch.quantile(target_sample, quantiles, dim=2),
        )
        return mean_loss + std_loss + quantile_loss


class MultiScaleHighFrequencyLoss(nn.Module):
    """Match high-frequency residuals with legacy or 3D Laplacian-pyramid bands."""

    def __init__(
        self,
        scales: int = 2,
        mode: str = "legacy",
        pyramid_kernels: tuple[tuple[int, int, int], ...] | None = None,
        pyramid_weights: tuple[float, ...] | None = None,
        phase_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.scales = max(1, scales)
        self.mode = mode.casefold()
        if self.mode not in {"legacy", "laplacian_3d"}:
            raise ValueError("texture_mode must be 'legacy' or 'laplacian_3d'")
        self.pyramid_kernels = pyramid_kernels or ((3, 3, 3), (3, 5, 5), (5, 7, 7))
        if any(
            len(kernel) != 3 or any(size < 1 or size % 2 == 0 for size in kernel)
            for kernel in self.pyramid_kernels
        ):
            raise ValueError("texture_pyramid_kernels must contain positive odd 3D kernels")
        raw_weights = pyramid_weights or tuple(1.0 for _ in self.pyramid_kernels)
        if len(raw_weights) != len(self.pyramid_kernels) or any(weight < 0 for weight in raw_weights):
            raise ValueError("texture_pyramid_weights must match the kernels and be non-negative")
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raise ValueError("At least one texture pyramid weight must be positive")
        self.pyramid_weights = tuple(weight / weight_sum for weight in raw_weights)
        self.phase_weights = phase_weights
        if phase_weights is not None and (not phase_weights or any(weight < 0 for weight in phase_weights)):
            raise ValueError("texture_phase_weights must be non-empty and non-negative")

    @staticmethod
    def _high_frequency(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            local_mean = F.avg_pool3d(x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
            return x - local_mean
        expanded_mask = _expand_mask(mask.to(dtype=x.dtype), x.shape[1])
        numerator = F.avg_pool3d(
            x * expanded_mask,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )
        denominator = F.avg_pool3d(
            expanded_mask,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )
        local_mean = numerator / denominator.clamp_min(1.0e-6)
        return (x - local_mean) * expanded_mask

    @staticmethod
    def _effective_kernel(kernel: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(
            min(size, extent if extent % 2 else max(1, extent - 1))
            for size, extent in zip(kernel, shape)
        )

    @staticmethod
    def _binomial_kernel_1d(size: int, reference: torch.Tensor) -> torch.Tensor:
        coefficients = [1.0]
        for _ in range(size - 1):
            coefficients = [1.0] + [
                coefficients[index - 1] + coefficients[index]
                for index in range(1, len(coefficients))
            ] + [1.0]
        kernel = reference.new_tensor(coefficients)
        return kernel / kernel.sum()

    @classmethod
    def _smooth_3d(
        cls,
        x: torch.Tensor,
        kernel: tuple[int, int, int],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kernel = cls._effective_kernel(kernel, tuple(x.shape[-3:]))
        kd, kh, kw = kernel
        channels = x.shape[1]
        kernel_d = cls._binomial_kernel_1d(kd, x)
        kernel_h = cls._binomial_kernel_1d(kh, x)
        kernel_w = cls._binomial_kernel_1d(kw, x)

        def separable_blur(value: torch.Tensor, padding_mode: str) -> torch.Tensor:
            output = value
            axes = (
                (kernel_d, (kd, 1, 1), (0, 0, 0, 0, kd // 2, kd // 2)),
                (kernel_h, (1, kh, 1), (0, 0, kh // 2, kh // 2, 0, 0)),
                (kernel_w, (1, 1, kw), (kw // 2, kw // 2, 0, 0, 0, 0)),
            )
            for axis_kernel, weight_shape, padding in axes:
                if axis_kernel.numel() == 1:
                    continue
                weight = axis_kernel.view(1, 1, *weight_shape).expand(channels, -1, -1, -1, -1)
                output = F.conv3d(F.pad(output, padding, mode=padding_mode), weight, groups=channels)
            return output

        if mask is None:
            return separable_blur(x, "replicate")
        expanded_mask = _expand_mask(mask.to(dtype=x.dtype), channels)
        numerator = separable_blur(x * expanded_mask, "constant")
        denominator = separable_blur(expanded_mask, "constant")
        return (numerator / denominator.clamp_min(1.0e-6)) * expanded_mask

    def _phase_weighted_mean(self, error: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.phase_weights is None:
            return _masked_mean_per_sample(error, mask)
        if len(self.phase_weights) != error.shape[1]:
            raise ValueError("texture_phase_weights must match the number of predicted phases")
        weights = error.new_tensor(self.phase_weights).view(1, -1)
        expanded_mask = (
            torch.ones_like(error)
            if mask is None
            else _expand_mask(mask.to(dtype=error.dtype), error.shape[1])
        )
        numerator = (error * expanded_mask).flatten(start_dim=2).sum(dim=2)
        denominator = expanded_mask.flatten(start_dim=2).sum(dim=2)
        valid = denominator > 0
        weighted_valid = weights * valid.to(dtype=weights.dtype)
        sample_weights = weighted_valid.sum(dim=1)
        if not torch.any(sample_weights > 0):
            return error.new_tensor(0.0)
        phase_means = numerator / denominator.clamp_min(1.0)
        sample_means = (phase_means * weighted_valid).sum(dim=1) / sample_weights.clamp_min(1.0e-6)
        return sample_means[sample_weights > 0].mean()

    def _laplacian_pyramid_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Undecimated Laplacian bands keep voxel alignment for paired medical images:
        # x-G1, G1-G2, G2-G3. The omitted coarsest low-pass band is already
        # supervised by the intensity, SSIM, distribution, and TIC losses.
        previous_pred = pred
        previous_target = target
        losses: list[torch.Tensor] = []
        for kernel, weight in zip(self.pyramid_kernels, self.pyramid_weights):
            smoothed_pred = self._smooth_3d(pred, kernel, mask)
            smoothed_target = self._smooth_3d(target, kernel, mask)
            pred_band = previous_pred - smoothed_pred
            target_band = previous_target - smoothed_target
            band_error = torch.abs(pred_band - target_band)
            losses.append(weight * self._phase_weighted_mean(band_error, mask))
            previous_pred = smoothed_pred
            previous_target = smoothed_target
        return sum(losses)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "laplacian_3d":
            return self._laplacian_pyramid_loss(pred, target, mask)
        losses: list[torch.Tensor] = []
        current_pred = pred
        current_target = target
        current_mask = mask
        for scale in range(self.scales):
            error = torch.abs(
                self._high_frequency(current_pred, current_mask)
                - self._high_frequency(current_target, current_mask)
            )
            losses.append(_masked_mean_per_sample(error, current_mask))
            if scale + 1 < self.scales:
                current_pred = F.avg_pool3d(current_pred, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                current_target = F.avg_pool3d(current_target, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if current_mask is not None:
                    current_mask = F.max_pool3d(
                        current_mask.to(dtype=current_pred.dtype),
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2),
                    )
        return sum(losses) / len(losses)


def feature_matching_loss(fake_features: torch.Tensor | list, real_features: torch.Tensor | list) -> torch.Tensor:
    fake_flat = _flatten_features(fake_features)
    real_flat = _flatten_features(real_features)
    if len(fake_flat) != len(real_flat) or not fake_flat:
        raise ValueError("Discriminator feature structures do not match")
    losses = [F.l1_loss(fake, real.detach()) for fake, real in zip(fake_flat, real_flat)]
    return sum(losses) / len(losses)


def discriminator_loss(real_logits: torch.Tensor | list[torch.Tensor], fake_logits: torch.Tensor | list[torch.Tensor], mode: str) -> torch.Tensor:
    losses = []
    for real, fake in zip(_as_list(real_logits), _as_list(fake_logits)):
        if mode == "bce":
            real_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
            fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        else:
            real_loss = torch.mean((real - 1.0).pow(2))
            fake_loss = torch.mean(fake.pow(2))
        losses.append(0.5 * (real_loss + fake_loss))
    return sum(losses) / len(losses)


def generator_adv_loss(fake_logits: torch.Tensor | list[torch.Tensor], mode: str) -> torch.Tensor:
    losses = []
    for fake in _as_list(fake_logits):
        if mode == "bce":
            losses.append(F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake)))
        else:
            losses.append(torch.mean((fake - 1.0).pow(2)))
    return sum(losses) / len(losses)


class GeneratorLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg = config["loss"]
        self.weights = {
            "charbonnier": float(cfg["charbonnier"]),
            "ssim": float(cfg["ssim"]),
            "gradient": float(cfg["gradient"]),
            "roi_charbonnier": float(cfg["roi_charbonnier"]),
            "balanced_roi_charbonnier": float(cfg.get("balanced_roi_charbonnier", 0.0)),
            "perilesional_near_charbonnier": float(cfg.get("perilesional_near_charbonnier", 0.0)),
            "perilesional_far_charbonnier": float(cfg.get("perilesional_far_charbonnier", 0.0)),
            "perilesional_near_ssim": float(cfg.get("perilesional_near_ssim", 0.0)),
            "perilesional_near_intensity_distribution": float(
                cfg.get("perilesional_near_intensity_distribution", 0.0)
            ),
            "perilesional_near_texture": float(cfg.get("perilesional_near_texture", 0.0)),
            "background_charbonnier": float(cfg.get("background_charbonnier", 0.0)),
            "background_intensity_distribution": float(cfg.get("background_intensity_distribution", 0.0)),
            "background_positive_leak": float(cfg.get("background_positive_leak", 0.0)),
            "static_anchor": float(cfg.get("static_anchor", 0.0)),
            "perilesional_positive_leak": float(cfg.get("perilesional_positive_leak", 0.0)),
            "enhancement_support": float(cfg.get("enhancement_support", 0.0)),
            "lesion_enhancement_support": float(cfg.get("lesion_enhancement_support", 0.0)),
            "roi_ssim": float(cfg.get("roi_ssim", 0.0)),
            "roi_gradient": float(cfg.get("roi_gradient", 0.0)),
            "roi_gradient_interior": float(cfg.get("roi_gradient_interior", 0.0)),
            "roi_gradient_boundary": float(cfg.get("roi_gradient_boundary", 0.0)),
            "gan": float(cfg["gan"]),
            "local_gan": float(cfg.get("local_gan", 0.0)),
            "feature_matching": float(cfg.get("feature_matching", 0.0)),
            "local_feature_matching": float(cfg.get("local_feature_matching", 0.0)),
            "roi_curve": float(cfg.get("roi_curve", 0.0)),
            "roi_pe_curve": float(cfg.get("roi_pe_curve", 0.0)),
            "temporal": float(cfg.get("temporal", 0.0)),
            "hotspot_curve": float(cfg.get("hotspot_curve", 0.0)),
            "hotspot_temporal": float(cfg.get("hotspot_temporal", 0.0)),
            "hotspot_trend": float(cfg.get("hotspot_trend", 0.0)),
            "hotspot_pe_curve": float(cfg.get("hotspot_pe_curve", 0.0)),
            "hotspot_pe_temporal": float(cfg.get("hotspot_pe_temporal", 0.0)),
            "roi_kinetic_distribution": float(cfg.get("roi_kinetic_distribution", 0.0)),
            "intensity_distribution": float(cfg.get("intensity_distribution", 0.0)),
            "roi_intensity_distribution": float(cfg.get("roi_intensity_distribution", 0.0)),
            "texture": float(cfg.get("texture", 0.0)),
            "roi_texture": float(cfg.get("roi_texture", 0.0)),
            "background_texture": float(cfg.get("background_texture", 0.0)),
        }
        negative_weights = {name: value for name, value in self.weights.items() if value < 0}
        if negative_weights:
            raise ValueError(f"Loss weights must be non-negative: {negative_weights}")
        self.max_weights = dict(self.weights)
        self.gan_mode = str(cfg.get("gan_mode", "lsgan")).lower()
        eps = float(cfg.get("charbonnier_eps", 1.0e-3))
        self.charbonnier = CharbonnierLoss(eps)
        self.balanced_roi_charbonnier = BalancedROICharbonnierLoss(eps)
        self.background_charbonnier = NearZeroBackgroundLoss(
            eps,
            float(cfg.get("background_near_zero_threshold", 0.25)),
        )
        self.background_positive_leak = PositiveBackgroundLeakLoss(
            float(cfg.get("background_near_zero_threshold", 0.25)),
            float(cfg.get("background_positive_tolerance", 0.05)),
        )
        self.static_anchor = StaticAnchorLoss(
            float(cfg.get("static_anchor_delta_threshold", 0.25)),
            float(cfg.get("static_anchor_tolerance", 0.05)),
        )
        raw_ring_kernel = cfg.get("perilesional_kernel", [3, 7, 7])
        ring_kernel = tuple(int(value) for value in raw_ring_kernel)
        self.perilesional_positive_leak = PerilesionalPositiveLeakLoss(
            ring_kernel,
            float(cfg.get("perilesional_target_threshold", 0.25)),
            float(cfg.get("perilesional_tolerance", 0.05)),
        )
        raw_support_thresholds = cfg.get("enhancement_support_thresholds", [0.25, 0.5, 1.0])
        support_thresholds = tuple(float(value) for value in raw_support_thresholds)
        self.enhancement_support = SoftEnhancementSupportLoss(
            support_thresholds,
            float(cfg.get("enhancement_support_temperature", 0.15)),
            float(cfg.get("enhancement_support_false_positive_weight", 0.7)),
            float(cfg.get("enhancement_support_false_negative_weight", 0.3)),
        )
        raw_window = cfg.get("ssim_window", [3, 7, 7])
        window = int(raw_window) if isinstance(raw_window, int) else tuple(int(value) for value in raw_window)
        ssim_clip = cfg.get("ssim_clip", 6.0)
        self.ssim = SSIM3DLoss(
            window,
            float(cfg.get("ssim_data_range", 12.0)),
            None if ssim_clip is None else float(ssim_clip),
        )
        self.gradient = GradientLoss3D()
        self.roi_gradient = MaskedGradientLoss3D()
        self.roi_curve = ROICurveLoss(eps)
        precontrast_floor = float(cfg.get("pe_precontrast_floor", 0.25))
        self.roi_pe_curve = ROIPercentEnhancementCurveLoss(eps, precontrast_floor)
        self.temporal = TemporalDifferenceLoss(eps)
        self.hotspot_kinetic = MultiHotspotKineticLoss(
            eps,
            float(cfg.get("hotspot_washout_delta_threshold", 0.05)),
            float(cfg.get("hotspot_washout_delta_weight", 2.0)),
            float(cfg.get("hotspot_worst_weight", 2.0)),
        )
        self.hotspot_trend = MultiHotspotTrendLoss(
            float(cfg.get("hotspot_trend_washout_threshold", -0.10)),
            float(cfg.get("hotspot_trend_persistent_threshold", 0.10)),
            float(cfg.get("hotspot_trend_margin", 0.02)),
            float(cfg.get("hotspot_worst_weight", 2.0)),
        )
        self.hotspot_pe_kinetic = MultiHotspotPercentEnhancementLoss(
            eps,
            precontrast_floor,
            float(cfg.get("hotspot_washout_delta_threshold", 0.05)),
            float(cfg.get("hotspot_washout_delta_weight", 2.0)),
            float(cfg.get("hotspot_worst_weight", 2.0)),
        )
        self.roi_kinetic_distribution = ROIKineticDistributionLoss(
            tuple(float(value) for value in cfg.get("kinetic_distribution_quantiles", [0.10, 0.25, 0.50, 0.75, 0.90])),
            precontrast_floor,
            float(cfg.get("kinetic_distribution_minimum_anchor_pe", 0.05)),
            float(cfg.get("kinetic_distribution_washout_threshold", -0.10)),
            float(cfg.get("kinetic_distribution_persistent_threshold", 0.10)),
            float(cfg.get("kinetic_distribution_type_temperature", 0.03)),
            float(cfg.get("kinetic_distribution_statistics_weight", 0.50)),
            float(cfg.get("kinetic_distribution_type_fraction_weight", 0.50)),
        )
        anchor_key = str(cfg.get("temporal_anchor_key", ""))
        self.zero_anchor = anchor_key.casefold() in {"zero", "zeros", "baseline_zero"}
        input_keys = [str(key) for key in config.get("data", {}).get("input_keys", [])]
        self.anchor_channel = input_keys.index(anchor_key) if anchor_key in input_keys else None
        distribution_stride = tuple(int(value) for value in cfg.get("distribution_stride", [2, 4, 4]))
        if len(distribution_stride) != 3 or any(value < 1 for value in distribution_stride):
            raise ValueError("loss.distribution_stride must contain three positive integers")
        self.intensity_distribution = IntensityDistributionLoss(distribution_stride)
        pyramid_kernels = tuple(
            tuple(int(size) for size in kernel)
            for kernel in cfg.get("texture_pyramid_kernels", [[3, 3, 3], [3, 5, 5], [5, 7, 7]])
        )
        pyramid_weights = tuple(
            float(weight) for weight in cfg.get("texture_pyramid_weights", [0.5, 0.35, 0.15])
        )
        raw_phase_weights = cfg.get("texture_phase_weights")
        phase_weights = None if raw_phase_weights is None else tuple(float(weight) for weight in raw_phase_weights)
        self.texture = MultiScaleHighFrequencyLoss(
            int(cfg.get("texture_scales", 2)),
            mode=str(cfg.get("texture_mode", "legacy")),
            pyramid_kernels=pyramid_kernels,
            pyramid_weights=pyramid_weights,
            phase_weights=phase_weights,
        )

    def set_adversarial_scale(self, scale: float) -> None:
        scale = max(0.0, min(1.0, float(scale)))
        for name in ("gan", "local_gan", "feature_matching", "local_feature_matching"):
            self.weights[name] = self.max_weights[name] * scale

    def reconstruction_terms(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        condition: torch.Tensor | None = None,
        foreground: torch.Tensor | None = None,
        hotspot_masks: torch.Tensor | None = None,
        perilesional_near: torch.Tensor | None = None,
        perilesional_far: torch.Tensor | None = None,
        precontrast_scaled: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = pred.new_tensor(0.0)

        def active(name: str, compute) -> torch.Tensor:
            return compute() if self.weights[name] > 0 else zero

        if any(
            self.weights[name] > 0
            for name in ("roi_gradient", "roi_gradient_interior", "roi_gradient_boundary")
        ):
            interior_gradient, boundary_gradient = self.roi_gradient(pred, target, roi)
        else:
            interior_gradient = boundary_gradient = zero
        anchor = None
        if condition is not None:
            if self.zero_anchor:
                anchor = torch.zeros_like(pred[:, :1])
            elif self.anchor_channel is not None:
                anchor = condition[:, self.anchor_channel : self.anchor_channel + 1]
        expanded_roi = _expand_mask(roi.to(dtype=torch.bool), pred.shape[1])
        background = ~expanded_roi
        if foreground is not None:
            background = background & _expand_mask(foreground.to(dtype=torch.bool), pred.shape[1])
        if self.weights["hotspot_curve"] > 0 or self.weights["hotspot_temporal"] > 0:
            hotspot_curve, hotspot_temporal = self.hotspot_kinetic(pred, target, anchor, hotspot_masks)
        else:
            hotspot_curve = hotspot_temporal = zero
        hotspot_trend = active(
            "hotspot_trend", lambda: self.hotspot_trend(pred, target, anchor, hotspot_masks)
        )
        if self.weights["hotspot_pe_curve"] > 0 or self.weights["hotspot_pe_temporal"] > 0:
            hotspot_pe_curve, hotspot_pe_temporal = self.hotspot_pe_kinetic(
                pred,
                target,
                anchor,
                precontrast_scaled,
                hotspot_masks,
            )
        else:
            hotspot_pe_curve = hotspot_pe_temporal = zero
        perilesional_ring = None
        if perilesional_near is not None and perilesional_far is not None:
            perilesional_ring = (perilesional_near > 0) | (perilesional_far > 0)
        return {
            "charbonnier": active("charbonnier", lambda: self.charbonnier(pred, target)),
            "ssim": active("ssim", lambda: self.ssim(pred, target)),
            "gradient": active("gradient", lambda: self.gradient(pred, target)),
            "roi_charbonnier": active("roi_charbonnier", lambda: self.charbonnier(pred, target, roi)),
            "balanced_roi_charbonnier": active(
                "balanced_roi_charbonnier", lambda: self.balanced_roi_charbonnier(pred, target, roi)
            ),
            "perilesional_near_charbonnier": active(
                "perilesional_near_charbonnier", lambda: self.charbonnier(pred, target, perilesional_near)
            ),
            "perilesional_far_charbonnier": active(
                "perilesional_far_charbonnier", lambda: self.charbonnier(pred, target, perilesional_far)
            ),
            "perilesional_near_ssim": active(
                "perilesional_near_ssim", lambda: self.ssim(pred, target, perilesional_near)
            ),
            "perilesional_near_intensity_distribution": active(
                "perilesional_near_intensity_distribution",
                lambda: self.intensity_distribution(pred, target, perilesional_near),
            ),
            "perilesional_near_texture": active(
                "perilesional_near_texture", lambda: self.texture(pred, target, perilesional_near)
            ),
            "background_charbonnier": active(
                "background_charbonnier", lambda: self.background_charbonnier(pred, target, roi, foreground)
            ),
            "background_intensity_distribution": active(
                "background_intensity_distribution",
                lambda: self.intensity_distribution(pred, target, background),
            ),
            "background_positive_leak": active(
                "background_positive_leak", lambda: self.background_positive_leak(pred, target, roi, foreground)
            ),
            "static_anchor": active(
                "static_anchor", lambda: self.static_anchor(pred, target, anchor, foreground)
            ),
            "perilesional_positive_leak": active(
                "perilesional_positive_leak",
                lambda: self.perilesional_positive_leak(
                    pred,
                    target,
                    roi,
                    foreground,
                    ring_mask=perilesional_ring,
                ),
            ),
            "enhancement_support": active(
                "enhancement_support", lambda: self.enhancement_support(pred, target, foreground)
            ),
            "lesion_enhancement_support": active(
                "lesion_enhancement_support", lambda: self.enhancement_support(pred, target, roi)
            ),
            "roi_ssim": active("roi_ssim", lambda: self.ssim(pred, target, roi)),
            "roi_gradient": 0.3 * interior_gradient + 0.7 * boundary_gradient,
            "roi_gradient_interior": interior_gradient,
            "roi_gradient_boundary": boundary_gradient,
            "roi_curve": active("roi_curve", lambda: self.roi_curve(pred, target, roi)),
            "roi_pe_curve": active(
                "roi_pe_curve", lambda: self.roi_pe_curve(pred, target, roi, precontrast_scaled)
            ),
            "temporal": active("temporal", lambda: self.temporal(pred, target, roi, anchor=anchor)),
            "hotspot_curve": hotspot_curve,
            "hotspot_temporal": hotspot_temporal,
            "hotspot_trend": hotspot_trend,
            "hotspot_pe_curve": hotspot_pe_curve,
            "hotspot_pe_temporal": hotspot_pe_temporal,
            "roi_kinetic_distribution": active(
                "roi_kinetic_distribution",
                lambda: self.roi_kinetic_distribution(pred, target, roi, anchor, precontrast_scaled),
            ),
            "intensity_distribution": active(
                "intensity_distribution", lambda: self.intensity_distribution(pred, target)
            ),
            "roi_intensity_distribution": active(
                "roi_intensity_distribution", lambda: self.intensity_distribution(pred, target, roi)
            ),
            "texture": active("texture", lambda: self.texture(pred, target)),
            "roi_texture": active("roi_texture", lambda: self.texture(pred, target, roi)),
            "background_texture": active(
                "background_texture", lambda: self.texture(pred, target, background)
            ),
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
        fake_logits: torch.Tensor | list[torch.Tensor] | None = None,
        fake_features: torch.Tensor | list | None = None,
        real_features: torch.Tensor | list | None = None,
        local_fake_logits: torch.Tensor | list[torch.Tensor] | None = None,
        local_fake_features: torch.Tensor | list | None = None,
        local_real_features: torch.Tensor | list | None = None,
        condition: torch.Tensor | None = None,
        foreground: torch.Tensor | None = None,
        hotspot_masks: torch.Tensor | None = None,
        perilesional_near: torch.Tensor | None = None,
        perilesional_far: torch.Tensor | None = None,
        precontrast_scaled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms = self.reconstruction_terms(
            pred,
            target,
            roi,
            condition=condition,
            foreground=foreground,
            hotspot_masks=hotspot_masks,
            perilesional_near=perilesional_near,
            perilesional_far=perilesional_far,
            precontrast_scaled=precontrast_scaled,
        )
        if fake_logits is None:
            terms["gan"] = pred.new_tensor(0.0)
        else:
            terms["gan"] = generator_adv_loss(fake_logits, self.gan_mode)
        if local_fake_logits is None:
            terms["local_gan"] = pred.new_tensor(0.0)
        else:
            terms["local_gan"] = generator_adv_loss(local_fake_logits, self.gan_mode)
        if fake_features is None or real_features is None:
            terms["feature_matching"] = pred.new_tensor(0.0)
        else:
            terms["feature_matching"] = feature_matching_loss(fake_features, real_features)
        if local_fake_features is None or local_real_features is None:
            terms["local_feature_matching"] = pred.new_tensor(0.0)
        else:
            terms["local_feature_matching"] = feature_matching_loss(local_fake_features, local_real_features)
        total = sum(self.weights[name] * value for name, value in terms.items())
        return total, terms
