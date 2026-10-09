from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import config_snapshot, prepare_output_run, resolve_checkpoint_path, save_config
from .data import (
    LesionSubDataset,
    build_patient_normalizers,
    collect_case_record,
    configured_input_keys,
    configured_target_keys,
    discover_lesion_samples,
    load_normalized_inputs,
    load_or_create_split,
    records_for_ids,
)
from .factory import build_discriminator, build_generator
from .hotspots import select_roi_hotspots
from .io import read_nifti, write_nifti
from .losses import GeneratorLoss, discriminator_loss
from .utils import ensure_dir, resolve_device, set_seed, worker_init_fn













def hotspot_masks_from_batch(
    config: dict[str, Any],
    condition: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    cfg = config.get("hotspot", {})
    if not bool(cfg.get("enabled", False)):
        return None
    selection_key = str(cfg.get("selection_input_key", "sub1"))
    input_keys = configured_input_keys(config)
    if selection_key not in input_keys:
        raise ValueError(f"hotspot.selection_input_key={selection_key!r} is not a model input")
    hotspot_roi = batch["hotspot_roi"].to(device, non_blocking=True)
    spacing_xyz = batch["spacing_xyz"].to(device, non_blocking=True)
    anchor = condition[:, input_keys.index(selection_key) : input_keys.index(selection_key) + 1]
    clip_values = cfg.get("selection_clip_percentiles", [1.0, 99.0])
    return select_roi_hotspots(
        anchor,
        hotspot_roi,
        spacing_xyz,
        count=int(cfg.get("count", 3)),
        radius_mm=float(cfg.get("radius_mm", cfg.get("maximum_radius_mm", 3.0))),
        minimum_radius_mm=float(cfg.get("minimum_radius_mm", 1.5)),
        radius_step_mm=float(cfg.get("radius_step_mm", 0.5)),
        minimum_voxels=int(cfg.get("minimum_voxels", 9)),
        slice_score_quantile=float(cfg.get("slice_score_quantile", 0.90)),
        slice_relative_threshold=float(cfg.get("slice_relative_threshold", 0.70)),
        maximum_candidate_slices=int(cfg.get("maximum_candidate_slices", 3)),
        candidate_top_fraction=float(cfg.get("candidate_top_fraction", 0.25)),
        minimum_mean_enhancement=float(cfg.get("minimum_mean_enhancement", 0.0)),
        heterogeneity_penalty=float(cfg.get("heterogeneity_penalty", 0.10)),
        clip_percentiles=(float(clip_values[0]), float(clip_values[1])),
    )


def context_masks_from_batch(
    config: dict[str, Any],
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not bool(config.get("data", {}).get("context_roi", {}).get("enabled", False)):
        return None, None
    return (
        batch["perilesional_near"].to(device, non_blocking=True),
        batch["perilesional_far"].to(device, non_blocking=True),
    )


def local_context_roi(
    roi: torch.Tensor,
    perilesional_near: torch.Tensor | None,
    perilesional_far: torch.Tensor | None,
) -> torch.Tensor:
    if perilesional_near is None or perilesional_far is None:
        return roi
    return ((roi > 0) | (perilesional_near > 0) | (perilesional_far > 0)).to(dtype=roi.dtype)





def tracked_checkpoint_metrics(config):
    return {}, {}


def make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    records, warnings = discover_lesion_samples(config)
    if not records:
        raise RuntimeError(f"No valid lesion samples found. Warnings: {warnings[:10]}")
    split = load_or_create_split(records, config)
    normalizers = build_patient_normalizers(records, config)
    train_records = records_for_ids(records, split["train"])
    val_records = records_for_ids(records, split["val"])

    if not train_records or not val_records:
        raise ValueError("Training and validation splits must both be nonempty")
    data_cfg = config["data"]
    train_cfg = config["train"]
    val_cfg = config.get("validation", {})
    common = {
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(
        LesionSubDataset(train_records, normalizers, config, training=True),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        LesionSubDataset(val_records, normalizers, config, training=False),
        batch_size=int(val_cfg.get("batch_size", 1)),
        shuffle=False,
        drop_last=False,
        **common,
    )
    info = {"records": records, "warnings": warnings, "split": split}
    return train_loader, val_loader, info








def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def discriminator_features(discriminator: torch.nn.Module, x: torch.Tensor) -> tuple[Any, Any]:
    forward_features = getattr(discriminator, "forward_features", None)
    if forward_features is None:
        raise TypeError(f"{type(discriminator).__name__} does not expose discriminator features")
    return forward_features(x)


def crop_around_roi(
    x: torch.Tensor,
    roi: torch.Tensor,
    crop_size: tuple[int, int, int],
    margin: tuple[int, int, int] = (0, 0, 0),
    resize_to_crop_size: bool = False,
) -> torch.Tensor:
    """Differentiably crop every lesion ROI, optionally resizing complete crops to one shape."""
    if len(crop_size) != 3 or any(size < 1 for size in crop_size):
        raise ValueError("local_adversarial.crop_size must contain three positive integers")
    if len(margin) != 3 or any(size < 0 for size in margin):
        raise ValueError("local_adversarial.margin must contain three non-negative integers")
    spatial_shape = x.shape[-3:]
    if any(size > available for size, available in zip(crop_size, spatial_shape)):
        raise ValueError(f"Local crop {crop_size} exceeds input shape {tuple(spatial_shape)}")
    union_roi = torch.any(roi > 0.5, dim=1)
    boxes: list[tuple[list[int], list[int]] | None] = []
    required_sizes: list[list[int]] = []
    for batch_index in range(x.shape[0]):
        coordinates = torch.nonzero(union_roi[batch_index], as_tuple=False)
        if coordinates.numel() == 0:
            boxes.append(None)
            required_sizes.append(list(crop_size))
            continue
        lower_tensor = coordinates.min(dim=0).values
        upper_tensor = coordinates.max(dim=0).values
        lower = [int(value.item()) for value in lower_tensor]
        upper = [int(value.item()) for value in upper_tensor]
        boxes.append((lower, upper))
        required_size = list(crop_size)
        for dim in range(3):
            extent = upper[dim] - lower[dim] + 1 + 2 * margin[dim]
            required_size[dim] = max(required_size[dim], min(extent, spatial_shape[dim]))
        required_sizes.append(required_size)

    if not resize_to_crop_size:
        batch_size = [max(sizes[dim] for sizes in required_sizes) for dim in range(3)]
        required_sizes = [batch_size for _ in required_sizes]

    crops: list[torch.Tensor] = []
    for batch_index in range(x.shape[0]):
        box = boxes[batch_index]
        required_size = required_sizes[batch_index]
        if box is None:
            center = [size // 2 for size in spatial_shape]
            starts = [
                max(0, min(center[dim] - required_size[dim] // 2, spatial_shape[dim] - required_size[dim]))
                for dim in range(3)
            ]
        else:
            lower, upper = box
            starts = []
            for dim in range(3):
                center_start = (lower[dim] + upper[dim] + 1 - required_size[dim]) // 2
                minimum_start = max(0, upper[dim] - required_size[dim] + 1)
                maximum_start = min(lower[dim], spatial_shape[dim] - required_size[dim])
                starts.append(max(minimum_start, min(center_start, maximum_start)))
        d, h, w = starts
        cd, ch, cw = required_size
        crop = x[batch_index : batch_index + 1, :, d : d + cd, h : h + ch, w : w + cw]
        if resize_to_crop_size and tuple(crop.shape[-3:]) != crop_size:
            crop = torch.nn.functional.interpolate(crop, size=crop_size, mode="trilinear", align_corners=False)
        crops.append(crop)
    return torch.cat(crops, dim=0)


def adversarial_scale(epoch: int, config: dict[str, Any]) -> float:
    start = int(config["loss"].get("gan_start_epoch", 1))
    ramp = max(0, int(config["loss"].get("gan_ramp_epochs", 0)))
    if epoch < start:
        return 0.0
    if ramp == 0:
        return 1.0
    return min(1.0, (epoch - start + 1) / ramp)


def build_training_discriminator(
    config: dict[str, Any],
    criterion: GeneratorLoss,
    device: torch.device,
) -> torch.nn.Module | None:
    adversarial_weights = ("gan", "local_gan", "feature_matching", "local_feature_matching")
    if not any(criterion.max_weights[name] > 0 for name in adversarial_weights):
        return None
    return build_discriminator(config).to(device)


def save_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer | None,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    best_metric_name: str | None = None,
    best_metrics: dict[str, float] | None = None,
    scheduler_g: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scaler_g: GradScaler | None = None,
    scaler_d: GradScaler | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict() if discriminator is not None else None,
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict() if opt_d is not None else None,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "best_metrics": best_metrics or {},
            "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
            "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
            "scaler_g": scaler_g.state_dict() if scaler_g is not None else None,
            "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
            "config": config_snapshot(config),
        },
        path,
    )


def load_generator_checkpoint(generator: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint["generator"] if "generator" in checkpoint else checkpoint
    generator.load_state_dict(state)
    return checkpoint


def load_training_checkpoint(
    checkpoint_path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer | None,
    device: torch.device,
    scheduler_g: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scaler_g: GradScaler | None = None,
    scaler_d: GradScaler | None = None,
    load_optimizer_state: bool = True,
) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    if discriminator is not None:
        discriminator_state = checkpoint.get("discriminator")
        if discriminator_state is None:
            raise ValueError("Cannot resume adversarial training: checkpoint has no discriminator state")
        discriminator.load_state_dict(discriminator_state)
    if load_optimizer_state:
        opt_g.load_state_dict(checkpoint["optimizer_g"])
        if opt_d is not None:
            optimizer_d_state = checkpoint.get("optimizer_d")
            if optimizer_d_state is None:
                raise ValueError("Cannot resume adversarial training: checkpoint has no discriminator optimizer state")
            opt_d.load_state_dict(optimizer_d_state)
        if scheduler_g is not None and checkpoint.get("scheduler_g") is not None:
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        if scheduler_d is not None and checkpoint.get("scheduler_d") is not None:
            scheduler_d.load_state_dict(checkpoint["scheduler_d"])
    if load_optimizer_state:
        for scaler, key in ((scaler_g, "scaler_g"), (scaler_d, "scaler_d")):
            if scaler is not None and checkpoint.get(key):
                scaler.load_state_dict(checkpoint[key])
    best_metrics = {str(name): float(value) for name, value in checkpoint.get("best_metrics", {}).items()}
    legacy_metric_name = checkpoint.get("best_metric_name")
    if not legacy_metric_name:
        legacy_metric_name = checkpoint.get("config", {}).get("train", {}).get("metric_for_best")
    if legacy_metric_name and "best_metric" in checkpoint:
        best_metrics.setdefault(str(legacy_metric_name), float(checkpoint["best_metric"]))
    return int(checkpoint.get("epoch", 0)), best_metrics


def metric_is_better(current: float, best: float, lower_is_better: bool) -> bool:
    return current < best if lower_is_better else current > best


def metric_improves_by(
    current: float,
    best: float,
    lower_is_better: bool,
    min_delta: float,
    relative: bool,
) -> bool:
    if not math.isfinite(best):
        return True
    required_delta = min_delta * max(abs(best), 1.0e-8) if relative else min_delta
    return current < best - required_delta if lower_is_better else current > best + required_delta


def best_history_metric(
    history_rows: list[dict[str, float]],
    metric_name: str,
    lower_is_better: bool,
) -> float | None:
    values = [float(row[metric_name]) for row in history_rows if metric_name in row and pd.notna(row[metric_name])]
    if not values:
        return None
    return min(values) if lower_is_better else max(values)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict[str, Any] | None,
    lower_is_better: bool,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if not scheduler_cfg or str(scheduler_cfg.get("type", "none")).lower() in {"none", "off", "false"}:
        return None
    scheduler_type = str(scheduler_cfg["type"]).lower()
    if scheduler_type in {"plateau", "reduce_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if lower_is_better else "max",
            factor=float(scheduler_cfg.get("factor", 0.3)),
            patience=int(scheduler_cfg.get("patience", 8)),
            threshold=float(scheduler_cfg.get("threshold", 1.0e-3)),
            threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
            cooldown=int(scheduler_cfg.get("cooldown", 0)),
            min_lr=float(scheduler_cfg.get("min_lr", 1.0e-6)),
            verbose=bool(scheduler_cfg.get("verbose", True)),
        )
    if scheduler_type in {"step", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 30)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported train.scheduler.type: {scheduler_cfg['type']}")


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    metric: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


def logged_loss_terms(prefix: str, terms: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, float]:
    raw = {f"{prefix}_{name}_loss": float(value.item()) for name, value in terms.items()}
    weighted = {
        f"{prefix}_weighted_{name}_loss": float(weights[name] * value.item())
        for name, value in terms.items()
    }
    return {**raw, **weighted}


@torch.no_grad()
def validate(generator, loader, criterion, device, config, max_batches=None):
    """Select checkpoints using held-out supervised reconstruction loss."""
    generator.eval()
    totals = {}
    count = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        roi = batch["roi"].to(device)
        pred = generator(image)
        near, far = context_masks_from_batch(config, batch, device)
        loss, _ = criterion(
            pred, target, roi, fake_logits=None, condition=image,
            foreground=batch["foreground"].to(device),
            hotspot_masks=hotspot_masks_from_batch(config, image, batch, device),
            perilesional_near=near, perilesional_far=far,
            precontrast_scaled=batch["precontrast_scaled"].to(device),
        )
        weight = image.shape[0]
        row = {"val_loss": float(loss), "val_mae": float((pred - target).abs().mean())}
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + weight * value
        count += weight
    if count == 0:
        raise ValueError("Validation set is empty; check the patient split and max_val_batches")
    return {key: value / count for key, value in totals.items()}


def average_metric_rows(rows):
    if not rows:
        raise ValueError("No training batches were processed")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}



def train(config: dict[str, Any]) -> dict[str, float]:
    set_seed(int(config["project"]["seed"]))
    output_dir = prepare_output_run(config, "train")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "logs")
    save_config(config, output_dir / "run_config.yaml")
    import platform
    from .utils import write_json
    write_json({"python": platform.python_version(), "pytorch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda, "requested_device": config["train"]["device"],
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
               output_dir / "environment.json")

    train_loader, val_loader, info = make_loaders(config)
    for warning in info["warnings"][:20]:
        print(f"[data warning] {warning}")
    print(f"Samples: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    device = resolve_device(str(config["train"]["device"]))
    generator = build_generator(config).to(device)
    criterion = GeneratorLoss(config).to(device)
    discriminator = build_training_discriminator(config, criterion, device)
    adversarial_enabled = discriminator is not None
    if discriminator is None:
        print("Adversarial training disabled: discriminator and its optimizer were not created")
    tracked_checkpoints, tracked_directions = tracked_checkpoint_metrics(config)
    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=float(config["train"]["lr_g"]),
        betas=tuple(float(v) for v in config["train"]["betas"]),
        weight_decay=float(config["train"].get("weight_decay_g", config["train"].get("weight_decay", 0.0))),
    )
    opt_d = None
    if discriminator is not None:
        opt_d = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(config["train"]["lr_d"]),
            betas=tuple(float(v) for v in config["train"]["betas"]),
            weight_decay=float(config["train"].get("weight_decay_d", config["train"].get("weight_decay", 0.0))),
        )

    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler_g = GradScaler(enabled=amp_enabled)
    scaler_d = GradScaler(enabled=amp_enabled) if discriminator is not None else None
    lower_is_better = bool(config["train"].get("lower_is_better", True))
    metric_name = str(config["train"]["metric_for_best"])
    metric_directions = dict(tracked_directions)
    metric_directions[metric_name] = lower_is_better
    legacy_scheduler = config["train"].get("scheduler")
    scheduler_g = build_scheduler(opt_g, config["train"].get("scheduler_g", legacy_scheduler), metric_directions[metric_name])
    scheduler_d = (
        build_scheduler(opt_d, config["train"].get("scheduler_d", legacy_scheduler), metric_directions[metric_name])
        if opt_d is not None
        else None
    )
    best_metrics = {
        name: float("inf") if direction else -float("inf")
        for name, direction in metric_directions.items()
    }
    start_epoch = 0
    resume_path = config["train"].get("resume")
    init_checkpoint = config["train"].get("init_checkpoint")
    if resume_path and init_checkpoint:
        raise ValueError("train.resume and train.init_checkpoint are mutually exclusive")
    if init_checkpoint:
        checkpoint = load_generator_checkpoint(generator, init_checkpoint, device)
        source_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        print(
            f"Initialized generator weights from {init_checkpoint}"
            + (f" (source epoch {source_epoch})" if source_epoch is not None else "")
        )
        print(
            "Started a new run at epoch 1 with fresh optimizer, scheduler, and best metrics"
            + ("; discriminator initialized" if discriminator is not None else "; discriminator disabled")
        )
    if resume_path:
        resume_optimizer_state = bool(config["train"].get("resume_optimizer_state", True))
        resume_best_metrics = bool(config["train"].get("resume_best_metrics", resume_optimizer_state))
        start_epoch, checkpoint_best_metrics = load_training_checkpoint(
            resume_path,
            generator,
            discriminator,
            opt_g,
            opt_d,
            device,
            scheduler_g,
            scheduler_d,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            load_optimizer_state=resume_optimizer_state,
        )
        if resume_best_metrics:
            for name, value in checkpoint_best_metrics.items():
                if name in best_metrics and metric_is_better(value, best_metrics[name], metric_directions[name]):
                    best_metrics[name] = value
        resume_mode = "model plus optimizer" if resume_optimizer_state else "model weights with fresh optimizer"
        print(f"Resumed {resume_mode} from {resume_path} at epoch {start_epoch}")
        if not resume_best_metrics:
            print("Reset checkpoint-selection metrics for this new fine-tuning run")
        for group in opt_g.param_groups:
            group["lr"] = float(config["train"]["lr_g"])
        if opt_d is not None:
            for group in opt_d.param_groups:
                group["lr"] = float(config["train"]["lr_d"])
        lr_message = f"Applied configured learning rate after resume: G={config['train']['lr_g']}"
        if opt_d is not None:
            lr_message += f" D={config['train']['lr_d']}"
        print(lr_message)
    history_path = output_dir / "metrics" / "train_history.csv"
    history_rows: list[dict[str, float]] = []
    if start_epoch > 0 and history_path.exists():
        history_rows = pd.read_csv(history_path).to_dict("records")
        history_rows = [row for row in history_rows if float(row.get("epoch", 0)) <= start_epoch]
        for name, direction in metric_directions.items():
            history_best = best_history_metric(history_rows, name, direction)
            if history_best is not None and metric_is_better(history_best, best_metrics[name], direction):
                best_metrics[name] = history_best
        print(f"Best validation metrics through epoch {start_epoch}: {best_metrics}")
    writer = None
    if bool(config.get("logging", {}).get("tensorboard", True)):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(output_dir / "logs"))
        except Exception as exc:  # pragma: no cover
            print(f"TensorBoard disabled: {exc}")

    epochs = int(config["train"]["epochs"])
    grad_accum = max(1, int(config["train"].get("gradient_accumulation", 1)))
    max_train_batches = config["train"].get("max_train_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_val_batches = config["train"].get("max_val_batches")
    max_val_batches = None if max_val_batches is None else int(max_val_batches)
    gan_mode = str(config["loss"].get("gan_mode", "lsgan")).lower()
    d_update_interval = max(1, int(config["train"].get("d_update_interval", 1)))
    grad_clip_g = float(config["train"].get("grad_clip_g", 0.0) or 0.0)
    grad_clip_d = float(config["train"].get("grad_clip_d", 0.0) or 0.0)
    local_cfg = config["train"].get("local_adversarial", {})
    local_enabled = bool(local_cfg.get("enabled", False))
    local_crop_size = tuple(int(value) for value in local_cfg.get("crop_size", [15, 64, 64]))
    local_margin = tuple(int(value) for value in local_cfg.get("margin", [0, 0, 0]))
    local_resize = bool(local_cfg.get("resize", False))
    local_d_weight = float(local_cfg.get("discriminator_weight", 1.0))
    if local_d_weight < 0:
        raise ValueError("train.local_adversarial.discriminator_weight must be non-negative")
    early_cfg = config["train"].get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 20))
    early_min_epochs = int(early_cfg.get("min_epochs", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_relative = str(early_cfg.get("min_delta_mode", "abs")).lower() == "rel"
    early_reset_on_lr_drop = bool(early_cfg.get("reset_on_lr_drop", False))
    early_lr_grace_epochs = int(early_cfg.get("lr_drop_grace_epochs", 0))
    if early_patience < 1 or early_min_epochs < 0 or early_min_delta < 0 or early_lr_grace_epochs < 0:
        raise ValueError("early_stopping patience must be positive and min_epochs/min_delta non-negative")
    early_best = best_metrics[metric_name]
    early_stale_epochs = 0
    early_grace_until_epoch = 0

    for epoch in range(start_epoch + 1, epochs + 1):
        current_adversarial_scale = adversarial_scale(epoch, config) if adversarial_enabled else 0.0
        criterion.set_adversarial_scale(current_adversarial_scale)
        generator.train()
        if discriminator is not None:
            discriminator.train()
        epoch_rows: list[dict[str, float]] = []
        opt_g.zero_grad(set_to_none=True)
        if opt_d is not None:
            opt_d.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        effective_batches = len(train_loader)
        if max_train_batches is not None:
            effective_batches = min(effective_batches, max_train_batches)
        for batch_index, batch in enumerate(progress, start=1):
            if max_train_batches is not None and batch_index > max_train_batches:
                break
            accum_scale = grad_accum
            if batch_index > effective_batches - (effective_batches % grad_accum or grad_accum):
                accum_scale = effective_batches % grad_accum or grad_accum
            should_step = batch_index % grad_accum == 0 or batch_index == effective_batches
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            roi = batch["roi"].to(device, non_blocking=True)
            foreground = batch["foreground"].to(device, non_blocking=True)
            precontrast_scaled = batch["precontrast_scaled"].to(device, non_blocking=True)
            hotspot_masks = hotspot_masks_from_batch(config, image, batch, device)
            perilesional_near, perilesional_far = context_masks_from_batch(config, batch, device)
            local_roi = local_context_roi(roi, perilesional_near, perilesional_far)

            with autocast(enabled=amp_enabled):
                fake = generator(image)
            optimizer_group = (batch_index - 1) // grad_accum
            update_discriminator = (
                discriminator is not None
                and current_adversarial_scale > 0
                and optimizer_group % d_update_interval == 0
            )
            d_loss = fake.new_tensor(0.0)
            global_d_loss = fake.new_tensor(0.0)
            local_d_loss = fake.new_tensor(0.0)
            real_pair = None
            if discriminator is not None:
                set_requires_grad(discriminator, update_discriminator)
                real_pair = torch.cat([image, target], dim=1)
                with autocast(enabled=amp_enabled):
                    fake_pair = torch.cat([image, fake.detach()], dim=1)
                    if update_discriminator:
                        global_d_loss = discriminator_loss(discriminator(real_pair), discriminator(fake_pair), gan_mode)
                        if local_enabled:
                            local_real_pair = crop_around_roi(real_pair, local_roi, local_crop_size, local_margin, local_resize)
                            local_fake_pair = crop_around_roi(fake_pair, local_roi, local_crop_size, local_margin, local_resize)
                            local_d_loss = discriminator_loss(
                                discriminator(local_real_pair),
                                discriminator(local_fake_pair),
                                gan_mode,
                            )
                            combined_d_loss = (global_d_loss + local_d_weight * local_d_loss) / (1.0 + local_d_weight)
                        else:
                            combined_d_loss = global_d_loss
                        d_loss = combined_d_loss / accum_scale
            if update_discriminator:
                assert scaler_d is not None
                scaler_d.scale(d_loss).backward()
            if should_step and update_discriminator:
                assert discriminator is not None and opt_d is not None and scaler_d is not None
                if grad_clip_d > 0:
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_d)
                scaler_d.step(opt_d)
                scaler_d.update()
                opt_d.zero_grad(set_to_none=True)

            if discriminator is not None:
                set_requires_grad(discriminator, False)
            with autocast(enabled=amp_enabled):
                if current_adversarial_scale > 0:
                    if discriminator is None or real_pair is None:
                        raise RuntimeError("Adversarial loss is active but the discriminator is unavailable")
                    generator_pair = torch.cat([image, fake], dim=1)
                    fake_logits, fake_features = discriminator_features(discriminator, generator_pair)
                    _, real_features = discriminator_features(discriminator, real_pair)
                    if local_enabled:
                        local_generator_pair = crop_around_roi(
                            generator_pair, local_roi, local_crop_size, local_margin, local_resize
                        )
                        local_real_pair = crop_around_roi(real_pair, local_roi, local_crop_size, local_margin, local_resize)
                        local_fake_logits, local_fake_features = discriminator_features(discriminator, local_generator_pair)
                        _, local_real_features = discriminator_features(discriminator, local_real_pair)
                    else:
                        local_fake_logits = local_fake_features = local_real_features = None
                else:
                    fake_logits = fake_features = real_features = None
                    local_fake_logits = local_fake_features = local_real_features = None
                g_loss, terms = criterion(
                    fake,
                    target,
                    roi,
                    fake_logits=fake_logits,
                    fake_features=fake_features,
                    real_features=real_features,
                    local_fake_logits=local_fake_logits,
                    local_fake_features=local_fake_features,
                    local_real_features=local_real_features,
                    condition=image,
                    foreground=foreground,
                    hotspot_masks=hotspot_masks,
                    perilesional_near=perilesional_near,
                    perilesional_far=perilesional_far,
                    precontrast_scaled=precontrast_scaled,
                )
                g_loss = g_loss / accum_scale
            scaler_g.scale(g_loss).backward()
            if should_step:
                if grad_clip_g > 0:
                    scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_g)
                scaler_g.step(opt_g)
                scaler_g.update()
                opt_g.zero_grad(set_to_none=True)

            row = {
                "train_g_loss": float(g_loss.item() * accum_scale),
                "train_d_loss": float(d_loss.item() * accum_scale),
                "train_d_global_loss": float(global_d_loss.item()),
                "train_d_local_loss": float(local_d_loss.item()),
                "train_d_updated": float(update_discriminator),
                **logged_loss_terms("train", terms, criterion.weights),
            }
            epoch_rows.append(row)
            if batch_index % int(config["train"]["log_every"]) == 0:
                progress.set_postfix(g=row["train_g_loss"], d=row["train_d_loss"])

        train_summary = average_metric_rows(epoch_rows)
        val_summary = validate(generator, val_loader, criterion, device, config, max_batches=max_val_batches)
        summary = {"epoch": float(epoch), **train_summary, **val_summary}
        summary["adversarial_scale"] = float(current_adversarial_scale)
        summary["gan_weight"] = float(criterion.weights["gan"])
        summary["local_gan_weight"] = float(criterion.weights["local_gan"])
        summary["lr_g"] = float(opt_g.param_groups[0]["lr"])
        summary["lr_d"] = float(opt_d.param_groups[0]["lr"]) if opt_d is not None else 0.0
        early_improved = metric_improves_by(
            float(summary[metric_name]),
            early_best,
            lower_is_better,
            early_min_delta,
            early_relative,
        )
        if early_improved:
            early_best = float(summary[metric_name])
            early_stale_epochs = 0
        else:
            early_stale_epochs += 1

        missing_metrics = [name for name in metric_directions if name not in summary]
        if missing_metrics:
            raise KeyError(f"Validation summary is missing checkpoint metrics: {missing_metrics}")
        improved: dict[str, bool] = {}
        for name, direction in metric_directions.items():
            current = float(summary[name])
            improved[name] = metric_is_better(current, best_metrics[name], direction)
            if improved[name]:
                best_metrics[name] = current

        scheduler_metric = float(summary[metric_name])
        step_scheduler(scheduler_g, scheduler_metric)
        step_scheduler(scheduler_d, scheduler_metric)
        next_lr_g = float(opt_g.param_groups[0]["lr"])
        lr_reduced = next_lr_g < float(summary["lr_g"]) - 1.0e-15
        if lr_reduced:
            early_grace_until_epoch = max(early_grace_until_epoch, epoch + early_lr_grace_epochs)
            if early_reset_on_lr_drop:
                early_stale_epochs = 0
        summary["lr_g_next"] = next_lr_g
        summary["lr_g_reduced"] = float(lr_reduced)
        summary["early_stopping_stale_epochs"] = float(early_stale_epochs)
        summary["early_stopping_grace_remaining"] = float(max(0, early_grace_until_epoch - epoch))
        history_rows.append(summary)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        if writer is not None:
            for key, value in summary.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        console_metrics = config.get("logging", {}).get("console_metrics")
        if console_metrics:
            displayed = []
            for name in console_metrics:
                if name in summary:
                    displayed.append(f"{name}={float(summary[name]):.6g}")
            print("Epoch summary: " + ", ".join(displayed))
        else:
            print(summary)

        for name, filename in tracked_checkpoints.items():
            if improved[name]:
                save_checkpoint(
                    output_dir / "checkpoints" / filename,
                    generator,
                    discriminator,
                    opt_g,
                    opt_d,
                    epoch,
                    best_metrics[name],
                    config,
                    best_metric_name=name,
                    best_metrics=best_metrics,
                    scheduler_g=scheduler_g,
                    scheduler_d=scheduler_d,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                )
        if improved[metric_name]:
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch,
                best_metrics[metric_name],
                config,
                best_metric_name=metric_name,
                best_metrics=best_metrics,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                scaler_g=scaler_g,
                scaler_d=scaler_d,
            )
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            generator,
            discriminator,
            opt_g,
            opt_d,
            epoch,
            best_metrics[metric_name],
            config,
            best_metric_name=metric_name,
            best_metrics=best_metrics,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )
        if epoch % int(config["train"]["save_every"]) == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch,
                best_metrics[metric_name],
                config,
                best_metric_name=metric_name,
                best_metrics=best_metrics,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                scaler_g=scaler_g,
                scaler_d=scaler_d,
            )
        if (
            early_enabled
            and epoch >= early_min_epochs
            and epoch >= early_grace_until_epoch
            and early_stale_epochs >= early_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: {metric_name} did not improve by "
                f"{early_min_delta:g} ({'relative' if early_relative else 'absolute'}) for "
                f"{early_stale_epochs} epochs. Best={early_best:.6g}."
            )
            break

    if writer is not None:
        writer.close()
    return history_rows[-1] if history_rows else {}





@torch.no_grad()
def inference(config: dict[str, Any], case_root: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    record = collect_case_record(case_root, require_targets=False, input_keys=configured_input_keys(config))
    records = [record]
    normalizers = build_patient_normalizers(records, config)
    image, reference_path, target_scale = load_normalized_inputs(record, normalizers[record.patient_id], config)
    device = resolve_device(str(config["train"]["device"]))
    generator = build_generator(config).to(device)
    checkpoint_path = resolve_checkpoint_path(config, "inference")
    config["inference"]["checkpoint"] = str(checkpoint_path)
    if output_dir is None:
        run_dir = prepare_output_run(config, "inference")
        save_config(config, run_dir / "run_config.yaml")
        out_dir = ensure_dir(run_dir / "predictions" / "inference" / record.patient_id)
    else:
        out_dir = ensure_dir(output_dir)
    print(f"Checkpoint: {checkpoint_path}")
    load_generator_checkpoint(generator, checkpoint_path, device)
    generator.eval()
    image_tensor = torch.from_numpy(image[None]).to(device)
    pred = generator(image_tensor)[0].detach().cpu().numpy() * float(target_scale)
    _, reference = read_nifti(reference_path)
    paths: list[Path] = []
    for phase, target_key in enumerate(configured_target_keys(config)):
        path = out_dir / f"{record.patient_id}_pred_{target_key.upper()}.nii.gz"
        write_nifti(pred[phase], reference, path)
        paths.append(path)
    return paths
