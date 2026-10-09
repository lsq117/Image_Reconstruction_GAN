from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "extends" in config:
        raise ValueError("Use the standalone configuration; inheritance is not supported")
    return config


def config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = copy.deepcopy(config)
    snapshot.pop("_runtime", None)
    return snapshot


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config_snapshot(config), stream, sort_keys=False)


def deep_update(config: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key, value in updates.items():
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def output_root(config: dict[str, Any]) -> Path:
    project = config["project"]
    return Path(str(project.get("output_root", project["output_dir"])))


def _run_index(path: Path) -> int:
    suffix = path.name[3:]
    return 1 if path.name == "exp" else int(suffix) if path.name.startswith("exp") and suffix.isdigit() else 0


def _allocate(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = parent / (prefix if index == 1 else f"{prefix}{index}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def resolve_checkpoint_path(config: dict[str, Any], section: str) -> Path:
    path = Path(str(config[section]["checkpoint"]))
    if path.is_file():
        return path.absolute()
    root = output_root(config)
    # Only the default root/checkpoints layout may auto-select a recent run.
    # An explicit missing checkpoint must never silently load another model.
    if path.parent.absolute() == (root / "checkpoints").absolute() and root.is_dir():
        runs = sorted((p for p in root.iterdir() if p.is_dir() and _run_index(p)), key=_run_index, reverse=True)
        for run in runs:
            candidate = run / "checkpoints" / path.name
            if candidate.is_file():
                return candidate.absolute()
    raise FileNotFoundError(f"Checkpoint does not exist: {path}. Train a model or pass --checkpoint.")


def prepare_output_run(config: dict[str, Any], mode: str) -> Path:
    runtime = config.setdefault("_runtime", {})
    if runtime.get("output_run_prepared"):
        return Path(config["project"]["output_dir"])
    root = output_root(config)
    if mode == "train":
        run = _allocate(root, "exp")
        training_run = run
    elif mode == "inference":
        checkpoint = resolve_checkpoint_path(config, "inference")
        training_run = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else root
        run = _allocate(training_run, "inference")
        config["inference"]["checkpoint"] = str(checkpoint)
    else:
        raise ValueError(f"Unsupported run mode: {mode}")
    config["project"].update(output_root=str(root), output_dir=str(run), run_name=run.name,
                             run_mode=mode, training_run_dir=str(training_run))
    if mode == "train":
        default = Path(config["inference"]["checkpoint"])
        if default.parent.absolute() == (root / "checkpoints").absolute():
            config["inference"]["checkpoint"] = str(run / "checkpoints" / default.name)
    runtime["output_run_prepared"] = True
    print(f"Output run: {run}")
    return run
