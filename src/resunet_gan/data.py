from __future__ import annotations

import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .io import read_nifti
from .spatial_masks import physical_perilesional_bands
from .utils import read_json, write_json, write_json_atomic


IMAGE_PATTERN = re.compile(
    r"^(?P<patient>.+)-(?P<modality>T1|T2|DWI|SUB)_(?P<series>[^_]+)_(?P<phase>[0-5])\.nii(?:\.gz)?$",
    re.IGNORECASE,
)
ROI_PREFIX = "roi_"
DEFAULT_INPUT_KEYS = ("t1_pre", "t2", "dwi", "sub1")
DEFAULT_TARGET_KEYS = ("sub2", "sub3", "sub4", "sub5")
REQUIRED_IMAGE_KEYS = (*DEFAULT_INPUT_KEYS, *DEFAULT_TARGET_KEYS)

SPLIT_NAMES = ("train", "val", "test")




@dataclass(frozen=True)
class SampleRecord:
    patient_id: str
    root: Path
    image_paths: dict[str, Path]
    roi_paths: dict[str, Path]


@dataclass(frozen=True)
class PatientNorm:
    t1_center: float
    t1_scale: float
    t2_center: float
    t2_scale: float
    dwi_center: float
    dwi_scale: float


def parse_image_name(path: Path) -> tuple[str, str] | None:
    filename = path.name
    is_roi = filename.casefold().startswith(ROI_PREFIX)
    stem_name = filename[len(ROI_PREFIX) :] if is_roi else filename
    match = IMAGE_PATTERN.match(stem_name)
    if match is None:
        return None

    modality = match.group("modality").upper()
    phase = int(match.group("phase"))
    if modality == "T1" and phase == 0:
        key = "t1_pre"
    elif modality == "T1":
        key = f"t1_post{phase}"
    elif modality == "T2" and phase == 0:
        key = "t2"
    elif modality == "DWI" and phase == 0:
        key = "dwi"
    elif modality == "SUB" and 1 <= phase <= 5:
        key = f"sub{phase}"
    else:
        return None
    return match.group("patient"), key


def _collect_patient_dir(patient_dir: Path) -> tuple[str | None, dict[str, Path], dict[str, Path]]:
    patient_id: str | None = None
    images: dict[str, Path] = {}
    rois: dict[str, Path] = {}
    for path in sorted(patient_dir.glob("*.nii*")):
        parsed = parse_image_name(path)
        if parsed is None:
            continue
        current_patient, key = parsed
        patient_id = patient_id or current_patient
        if current_patient != patient_id:
            raise ValueError(f"Multiple patient IDs found in one directory: {patient_dir}")
        if path.name.casefold().startswith(ROI_PREFIX):
            rois[key] = path
        else:
            images[key] = path
    return patient_id, images, rois


def discover_lesion_samples(config: dict[str, Any]) -> tuple[list[SampleRecord], list[str]]:
    lesion_root = Path(str(config["data"]["lesion_root"]))

    records: list[SampleRecord] = []
    warnings: list[str] = []
    if not lesion_root.exists():
        return [], [f"Lesion root does not exist: {lesion_root}"]

    for patient_dir in sorted(p for p in lesion_root.iterdir() if p.is_dir()):
        patient_id, images, rois = _collect_patient_dir(patient_dir)
        if patient_id is None:
            warnings.append(f"No parseable NIfTI files in {patient_dir}")
            continue
        required = set(configured_input_keys(config)) | set(configured_target_keys(config))
        missing = sorted(required - set(images))
        if missing:
            warnings.append(f"{patient_id} missing required images: {missing}")
            continue
        records.append(SampleRecord(patient_id=patient_id, root=patient_dir, image_paths=images, roi_paths=rois))
    return records, warnings


def collect_case_record(
    case_root: str | Path,
    require_targets: bool = False,
    input_keys: tuple[str, ...] = DEFAULT_INPUT_KEYS,
    target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
) -> SampleRecord:
    case_root = Path(case_root)
    patient_id, images, rois = _collect_patient_dir(case_root)
    if patient_id is None:
        raise RuntimeError(f"No parseable NIfTI files in {case_root}")
    required = set(input_keys)
    if require_targets:
        required.update(target_keys)
    missing = sorted(required - set(images))
    if missing:
        raise RuntimeError(f"{patient_id} missing required files for inference: {missing}")
    return SampleRecord(patient_id=patient_id, root=case_root, image_paths=images, roi_paths=rois)





def configured_input_keys(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(key) for key in config["data"].get("input_keys", DEFAULT_INPUT_KEYS))


def configured_target_keys(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(key) for key in config["data"].get("target_keys", DEFAULT_TARGET_KEYS))


def _read_required_image(record: SampleRecord, key: str) -> np.ndarray:
    image, _ = read_nifti(record.image_paths[key])
    return image


def _norm_params(norm: PatientNorm, key: str) -> tuple[float, float]:
    if key == "t1_pre":
        return norm.t1_center, norm.t1_scale
    if key == "t2":
        return norm.t2_center, norm.t2_scale
    if key == "dwi":
        return norm.dwi_center, norm.dwi_scale
    raise ValueError(f"No normalizer is defined for input key: {key}")


def _scale_for_key(norm: PatientNorm, key: str) -> float:
    if key == "t1_pre":
        return norm.t1_scale
    if key == "t2":
        return norm.t2_scale
    if key == "dwi":
        return norm.dwi_scale
    if key in {"none", "raw"}:
        return 1.0
    raise ValueError(f"Unsupported normalization.target_scale: {key}")


def _clip_optional(array: np.ndarray, clip_z: float | None) -> np.ndarray:
    if clip_z is None:
        return array.astype(np.float32, copy=False)
    return np.clip(array, -clip_z, clip_z).astype(np.float32, copy=False)


def _normalization_clips(config: dict[str, Any]) -> tuple[float | None, float | None]:
    cfg = config["normalization"]
    legacy_clip = cfg.get("clip_z", 6.0)
    input_value = cfg.get("input_clip_z", legacy_clip)
    sub_value = cfg.get("sub_clip_z", legacy_clip)
    input_clip = None if input_value is None else float(input_value)
    sub_clip = None if sub_value is None else float(sub_value)
    return input_clip, sub_clip


def _normalize_input(
    record: SampleRecord,
    norm: PatientNorm,
    key: str,
    input_clip_z: float | None,
    sub_clip_z: float | None,
) -> np.ndarray:
    array = _read_required_image(record, key)
    if key.startswith("sub"):
        return _clip_optional(array / norm.t1_scale, sub_clip_z)
    normalized = (array - _norm_params(norm, key)[0]) / _norm_params(norm, key)[1]
    return _clip_optional(normalized, input_clip_z)


def load_normalized_inputs(record: SampleRecord, norm: PatientNorm, config: dict[str, Any]) -> tuple[np.ndarray, Path, float]:
    input_clip_z, sub_clip_z = _normalization_clips(config)
    input_keys = configured_input_keys(config)
    image = np.stack(
        [_normalize_input(record, norm, key, input_clip_z, sub_clip_z) for key in input_keys],
        axis=0,
    )
    target_scale_key = str(config["normalization"].get("target_scale", "t1_pre"))
    return image, record.image_paths[input_keys[0]], _scale_for_key(norm, target_scale_key)


def patient_split_stratum(patient_id: str, config: dict[str, Any]) -> str:
    stratify_cfg = config["split"].get("stratify", {})
    if not bool(stratify_cfg.get("enabled", False)):
        return "all"

    prefixes = [str(value) for value in stratify_cfg.get("prefixes", [])]
    if not prefixes:
        raise ValueError("split.stratify.prefixes must contain at least one prefix when stratification is enabled")
    case_sensitive = bool(stratify_cfg.get("case_sensitive", False))
    candidate = patient_id if case_sensitive else patient_id.casefold()
    normalized_prefixes = prefixes if case_sensitive else [prefix.casefold() for prefix in prefixes]
    prefix_label = str(stratify_cfg.get("prefix_label", "prefix"))
    other_label = str(stratify_cfg.get("other_label", "other"))
    return prefix_label if candidate.startswith(tuple(normalized_prefixes)) else other_label


def _split_ratios(config: dict[str, Any]) -> list[float]:
    ratios = [float(config["split"][f"{name}_ratio"]) for name in SPLIT_NAMES]
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"split ratios must be positive and sum to 1.0, got {ratios}")
    return ratios


def _apportion_split_counts(n_total: int, ratios: list[float]) -> list[int]:
    exact_counts = [n_total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact_counts]
    remainder_order = sorted(
        range(3),
        key=lambda index: (exact_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[: n_total - sum(counts)]:
        counts[index] += 1
    return counts


def _partition_patient_ids(patient_ids: list[str], ratios: list[float], rng: random.Random) -> dict[str, list[str]]:
    shuffled_ids = sorted(patient_ids)
    rng.shuffle(shuffled_ids)
    counts = _apportion_split_counts(len(shuffled_ids), ratios)
    n_train, n_val, _ = counts
    return {
        "train": shuffled_ids[:n_train],
        "val": shuffled_ids[n_train : n_train + n_val],
        "test": shuffled_ids[n_train + n_val :],
    }


def create_patient_split(patient_ids: list[str], config: dict[str, Any]) -> dict[str, list[str]]:
    unique_ids = sorted(set(patient_ids))
    if len(unique_ids) < 3:
        raise ValueError("At least three patients are required for train/val/test splitting")
    ratios = _split_ratios(config)
    rng = random.Random(int(config["project"]["seed"]))
    strata: dict[str, list[str]] = {}
    for patient_id in unique_ids:
        strata.setdefault(patient_split_stratum(patient_id, config), []).append(patient_id)

    split = {name: [] for name in SPLIT_NAMES}
    for stratum, stratum_ids in sorted(strata.items()):
        counts = _apportion_split_counts(len(stratum_ids), ratios)
        if any(count == 0 for count in counts):
            raise ValueError(
                f"Stratum {stratum!r} has only {len(stratum_ids)} patients and cannot be represented "
                f"in every positive-ratio split (allocated counts={counts})"
            )
        stratum_split = _partition_patient_ids(stratum_ids, ratios, rng)
        for name in SPLIT_NAMES:
            split[name].extend(stratum_split[name])
    return {name: sorted(split[name]) for name in SPLIT_NAMES}


def load_or_create_split(records: list[SampleRecord], config: dict[str, Any], overwrite: bool | None = None) -> dict[str, list[str]]:
    split_path = Path(str(config["split"]["path"]))
    overwrite = bool(config["split"].get("overwrite", False)) if overwrite is None else overwrite
    if split_path.exists() and not overwrite:
        split = read_json(split_path)
    else:
        split = create_patient_split([record.patient_id for record in records], config)
        write_json(split, split_path)
    assert_no_split_leakage(split)
    expected_ids = {record.patient_id for record in records}
    split_ids = set().union(*(set(split.get(name, [])) for name in ("train", "val", "test")))
    if split_ids != expected_ids:
        missing = sorted(expected_ids - split_ids)
        unexpected = sorted(split_ids - expected_ids)
        raise ValueError(
            f"Split does not match the discovered patient cohort (missing={missing[:10]}, unexpected={unexpected[:10]}). "
            "Use a new split.path for a new cohort, or set split.overwrite=true to regenerate the split."
        )
    return split


def assert_no_split_leakage(split: dict[str, list[str]]) -> None:
    train, val, test = (set(split.get(name, [])) for name in ("train", "val", "test"))
    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    bad = {key: value for key, value in overlaps.items() if value}
    if bad:
        raise ValueError(f"Patient leakage across splits: {bad}")


def records_for_ids(records: list[SampleRecord], patient_ids: list[str]) -> list[SampleRecord]:
    selected = set(patient_ids)
    return [record for record in records if record.patient_id in selected]


def _foreground_values(array: np.ndarray, percentile: float) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    threshold = np.percentile(values, percentile)
    foreground = values[values > threshold]
    if foreground.size < 16:
        foreground = values
    return foreground.astype(np.float32, copy=False)


def _center_scale(array: np.ndarray, percentile: float, min_scale: float) -> tuple[float, float]:
    values = _foreground_values(array, percentile)
    center = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < min_scale:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < min_scale:
        scale = 1.0
    return center, scale


def _normalizer_cache_path(config: dict[str, Any]) -> Path:
    norm_cfg = config["normalization"]
    configured_path = norm_cfg.get("cache_path")
    if configured_path:
        return Path(str(configured_path))
    project_cfg = config["project"]
    output_root = Path(str(project_cfg.get("output_root", project_cfg["output_dir"])))
    return output_root / "cache" / "patient_normalizers.json"


def _normalizer_source_fingerprint(record: SampleRecord) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for key in ("t1_pre", "t2", "dwi"):
        path = record.image_paths[key]
        stat = path.stat()
        sources[key] = {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return sources


def _patient_norm_from_cache(value: Any) -> PatientNorm | None:
    if not isinstance(value, dict):
        return None
    try:
        norm = PatientNorm(
            t1_center=float(value["t1_center"]),
            t1_scale=float(value["t1_scale"]),
            t2_center=float(value["t2_center"]),
            t2_scale=float(value["t2_scale"]),
            dwi_center=float(value["dwi_center"]),
            dwi_scale=float(value["dwi_scale"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    values = (
        norm.t1_center,
        norm.t1_scale,
        norm.t2_center,
        norm.t2_scale,
        norm.dwi_center,
        norm.dwi_scale,
    )
    if not all(np.isfinite(value) for value in values):
        return None
    if norm.t1_scale <= 0 or norm.t2_scale <= 0 or norm.dwi_scale <= 0:
        return None
    return norm


def _patient_norm_to_cache(norm: PatientNorm) -> dict[str, float]:
    return {
        "t1_center": norm.t1_center,
        "t1_scale": norm.t1_scale,
        "t2_center": norm.t2_center,
        "t2_scale": norm.t2_scale,
        "dwi_center": norm.dwi_center,
        "dwi_scale": norm.dwi_scale,
    }


def _compute_patient_norm(record: SampleRecord, percentile: float, min_scale: float) -> PatientNorm:
    t1, _ = read_nifti(record.image_paths["t1_pre"])
    t2, _ = read_nifti(record.image_paths["t2"])
    dwi, _ = read_nifti(record.image_paths["dwi"])
    t1_center, t1_scale = _center_scale(t1, percentile, min_scale)
    t2_center, t2_scale = _center_scale(t2, percentile, min_scale)
    dwi_center, dwi_scale = _center_scale(dwi, percentile, min_scale)
    return PatientNorm(t1_center, t1_scale, t2_center, t2_scale, dwi_center, dwi_scale)


def build_patient_normalizers(records: list[SampleRecord], config: dict[str, Any]) -> dict[str, PatientNorm]:
    norm_cfg = config["normalization"]
    percentile = float(norm_cfg["foreground_percentile"])
    min_scale = float(norm_cfg["min_scale"])
    cache_enabled = bool(norm_cfg.get("cache_enabled", True))
    workers = max(1, int(norm_cfg.get("workers", 4)))
    cache_path = _normalizer_cache_path(config)
    signature = {
        "foreground_percentile": percentile,
        "min_scale": min_scale,
    }
    cached_records: dict[str, Any] = {}
    if cache_enabled and cache_path.is_file():
        try:
            cached = read_json(cache_path)
            if cached.get("schema_version") == 1 and cached.get("normalization") == signature:
                cached_records = dict(cached.get("records", {}))
        except Exception as exc:
            print(f"Patient normalizer cache ignored ({cache_path}): {exc}")

    cache_label = str(cache_path) if cache_enabled else "disabled"
    print(f"Patient normalizers: checking {len(records)} cases (cache={cache_label}, workers={workers})")
    norms: dict[str, PatientNorm] = {}
    updated_cache_records: dict[str, Any] = {}
    reused = 0
    pending: list[tuple[SampleRecord, dict[str, dict[str, Any]]]] = []
    for record in tqdm(records, desc="checking normalizer cache", unit="case", dynamic_ncols=True):
        sources = _normalizer_source_fingerprint(record) if cache_enabled else {}
        cached_record = cached_records.get(record.patient_id, {}) if cache_enabled else {}
        norm = None
        if isinstance(cached_record, dict) and cached_record.get("sources") == sources:
            norm = _patient_norm_from_cache(cached_record.get("norm"))
        if norm is None:
            pending.append((record, sources))
            continue
        reused += 1
        norms[record.patient_id] = norm
        updated_cache_records[record.patient_id] = {"sources": sources, "norm": _patient_norm_to_cache(norm)}

    if pending:
        if workers == 1:
            for record, sources in tqdm(
                pending,
                desc="computing patient normalizers",
                unit="case",
                dynamic_ncols=True,
            ):
                norm = _compute_patient_norm(record, percentile, min_scale)
                norms[record.patient_id] = norm
                updated_cache_records[record.patient_id] = {
                    "sources": sources,
                    "norm": _patient_norm_to_cache(norm),
                }
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_compute_patient_norm, record, percentile, min_scale): (record, sources)
                    for record, sources in pending
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="computing patient normalizers",
                    unit="case",
                    dynamic_ncols=True,
                ):
                    record, sources = futures[future]
                    norm = future.result()
                    norms[record.patient_id] = norm
                    updated_cache_records[record.patient_id] = {
                        "sources": sources,
                        "norm": _patient_norm_to_cache(norm),
                    }

    if cache_enabled:
        try:
            write_json_atomic(
                {
                    "schema_version": 1,
                    "normalization": signature,
                    "records": updated_cache_records,
                },
                cache_path,
            )
        except Exception as exc:
            print(f"Patient normalizer cache could not be saved ({cache_path}): {exc}")
    print(f"Patient normalizers ready: reused={reused}, recomputed={len(pending)}")
    return norms


def _norm_clip(array: np.ndarray, center: float, scale: float, clip_z: float) -> np.ndarray:
    return np.clip((array - center) / scale, -clip_z, clip_z).astype(np.float32, copy=False)


def _load_roi(record: SampleRecord, source: str) -> np.ndarray:
    if source != "union" and source in record.roi_paths:
        roi, _ = read_nifti(record.roi_paths[source])
        return (roi > 0).astype(np.float32, copy=False)
    if record.roi_paths:
        union: np.ndarray | None = None
        for path in record.roi_paths.values():
            roi, _ = read_nifti(path)
            current = roi > 0
            union = current if union is None else (union | current)
        return union.astype(np.float32, copy=False)
    ref, _ = read_nifti(record.image_paths["t1_pre"])
    return np.zeros_like(ref, dtype=np.float32)


class LesionSubDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        normalizers: dict[str, PatientNorm],
        config: dict[str, Any],
        training: bool,
    ) -> None:
        self.records = records
        self.normalizers = normalizers
        self.config = config
        self.training = training
        self.expected_shape = tuple(int(v) for v in config["data"]["expected_shape"])
        self.input_clip_z, self.sub_clip_z = _normalization_clips(config)
        self.foreground_percentile = float(config["normalization"]["foreground_percentile"])
        self.roi_source = str(config["data"].get("roi_source", "t1_post3"))
        self.flip_prob = float(config["augmentation"].get("flip_prob", 0.0))
        self.augment = bool(config["augmentation"].get("enable", False)) and training
        self.input_keys = configured_input_keys(config)
        self.target_keys = configured_target_keys(config)
        configured_roi_sources = config["data"].get("roi_sources")
        if configured_roi_sources is None:
            self.roi_sources: tuple[str, ...] | None = None
        else:
            missing_sources = [key for key in self.target_keys if key not in configured_roi_sources]
            if missing_sources:
                raise ValueError(f"data.roi_sources is missing target keys: {missing_sources}")
            self.roi_sources = tuple(str(configured_roi_sources[key]) for key in self.target_keys)
        self.target_scale_key = str(config["normalization"].get("target_scale", "t1_pre"))
        hotspot_cfg = config.get("hotspot", {})
        self.hotspot_enabled = bool(hotspot_cfg.get("enabled", False))
        self.hotspot_roi_source = str(hotspot_cfg.get("roi_source", "t1_post1"))
        context_cfg = config["data"].get("context_roi", {})
        self.context_enabled = bool(context_cfg.get("enabled", False))
        single_radius = context_cfg.get("radius_mm")
        if single_radius is not None:
            self.context_inner_radius_mm = float(single_radius)
            self.context_outer_radius_mm: float | None = None
            if self.context_enabled and self.context_inner_radius_mm <= 0:
                raise ValueError("data.context_roi.radius_mm must be positive")
        else:
            self.context_inner_radius_mm = float(context_cfg.get("inner_radius_mm", 5.0))
            self.context_outer_radius_mm = float(context_cfg.get("outer_radius_mm", 10.0))
            if self.context_enabled and not (
                0.0 < self.context_inner_radius_mm < self.context_outer_radius_mm
            ):
                raise ValueError("data.context_roi radii must satisfy 0 < inner_radius_mm < outer_radius_mm")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        norm = self.normalizers[record.patient_id]
        _, ref = read_nifti(record.image_paths[self.input_keys[0]])

        image = np.stack(
            [
                _normalize_input(record, norm, key, self.input_clip_z, self.sub_clip_z)
                for key in self.input_keys
            ],
            axis=0,
        )
        target_scale = _scale_for_key(norm, self.target_scale_key)
        target = np.stack(
            [
                _clip_optional(_read_required_image(record, key) / target_scale, self.sub_clip_z)
                for key in self.target_keys
            ],
            axis=0,
        )
        t1_pre = _read_required_image(record, "t1_pre")
        precontrast_scaled = (t1_pre / target_scale)[None].astype(np.float32, copy=False)
        if self.roi_sources is None:
            roi = _load_roi(record, self.roi_source)[None]
        else:
            missing_roi_paths = [source for source in self.roi_sources if source not in record.roi_paths]
            if missing_roi_paths:
                raise RuntimeError(f"{record.patient_id} missing configured ROI sources: {missing_roi_paths}")
            roi = np.stack([_load_roi(record, source) for source in self.roi_sources], axis=0)
        if self.hotspot_enabled:
            if self.hotspot_roi_source not in record.roi_paths:
                raise RuntimeError(f"{record.patient_id} missing hotspot ROI source: {self.hotspot_roi_source}")
            hotspot_roi = _load_roi(record, self.hotspot_roi_source)[None]
        else:
            hotspot_roi = np.zeros((1, *self.expected_shape), dtype=np.float32)
        if self.context_enabled:
            perilesional_near, perilesional_far = physical_perilesional_bands(
                roi,
                tuple(float(value) for value in ref.GetSpacing()),
                self.context_inner_radius_mm,
                self.context_outer_radius_mm,
            )
        else:
            perilesional_near = np.zeros_like(roi, dtype=np.float32)
            perilesional_far = np.zeros_like(roi, dtype=np.float32)
        finite_t1 = t1_pre[np.isfinite(t1_pre)]
        foreground_threshold = np.percentile(finite_t1, self.foreground_percentile) if finite_t1.size else 0.0
        foreground = (np.isfinite(t1_pre) & (t1_pre > foreground_threshold))[None].astype(np.float32, copy=False)

        if image.shape[1:] != self.expected_shape:
            raise ValueError(f"{record.root} image shape {image.shape[1:]} != {self.expected_shape}")
        if target.shape[1:] != self.expected_shape:
            raise ValueError(f"{record.root} target shape {target.shape[1:]} != {self.expected_shape}")
        if roi.shape[1:] != self.expected_shape:
            raise ValueError(f"{record.root} roi shape {roi.shape[1:]} != {self.expected_shape}")

        if self.augment:
            (
                image,
                target,
                roi,
                hotspot_roi,
                foreground,
                perilesional_near,
                perilesional_far,
                precontrast_scaled,
            ) = self._augment(
                image,
                target,
                roi,
                hotspot_roi,
                foreground,
                perilesional_near,
                perilesional_far,
                precontrast_scaled,
            )

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "roi": torch.from_numpy(np.ascontiguousarray(roi)),
            "hotspot_roi": torch.from_numpy(np.ascontiguousarray(hotspot_roi)),
            "perilesional_near": torch.from_numpy(np.ascontiguousarray(perilesional_near)),
            "perilesional_far": torch.from_numpy(np.ascontiguousarray(perilesional_far)),
            "foreground": torch.from_numpy(np.ascontiguousarray(foreground)),
            "precontrast_scaled": torch.from_numpy(np.ascontiguousarray(precontrast_scaled)),
            "patient_id": record.patient_id,
            "root": str(record.root),
            "reference_path": str(record.image_paths[self.input_keys[0]]),
            "target_scale": torch.tensor(target_scale, dtype=torch.float32),
            "spacing_xyz": torch.tensor(ref.GetSpacing(), dtype=torch.float32),
        }

    def _augment(
        self,
        image: np.ndarray,
        target: np.ndarray,
        roi: np.ndarray,
        hotspot_roi: np.ndarray,
        foreground: np.ndarray,
        perilesional_near: np.ndarray,
        perilesional_far: np.ndarray,
        precontrast_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=2)
            target = np.flip(target, axis=2)
            roi = np.flip(roi, axis=2)
            hotspot_roi = np.flip(hotspot_roi, axis=2)
            foreground = np.flip(foreground, axis=2)
            perilesional_near = np.flip(perilesional_near, axis=2)
            perilesional_far = np.flip(perilesional_far, axis=2)
            precontrast_scaled = np.flip(precontrast_scaled, axis=2)
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=3)
            target = np.flip(target, axis=3)
            roi = np.flip(roi, axis=3)
            hotspot_roi = np.flip(hotspot_roi, axis=3)
            foreground = np.flip(foreground, axis=3)
            perilesional_near = np.flip(perilesional_near, axis=3)
            perilesional_far = np.flip(perilesional_far, axis=3)
            precontrast_scaled = np.flip(precontrast_scaled, axis=3)
        return image, target, roi, hotspot_roi, foreground, perilesional_near, perilesional_far, precontrast_scaled
