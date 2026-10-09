from __future__ import annotations

from torch import nn

from .models import MultiScalePatchDiscriminator3D, PatchDiscriminator3D, ResidualUNet3D


def build_generator(config: dict) -> nn.Module:
    cfg = config["model"]["generator"]
    input_keys = config["data"].get("input_keys", [])
    target_keys = config["data"].get("target_keys", [])
    if input_keys and int(cfg["in_channels"]) != len(input_keys):
        raise ValueError(f"generator.in_channels={cfg['in_channels']} but data.input_keys has {len(input_keys)} entries")
    if target_keys and int(cfg["out_channels"]) != len(target_keys):
        raise ValueError(f"generator.out_channels={cfg['out_channels']} but data.target_keys has {len(target_keys)} entries")
    phase_cfg = cfg.get("phase_conditioning", {})
    if isinstance(phase_cfg, bool):
        phase_enabled = phase_cfg
        phase_embedding_dim = 64
    else:
        phase_enabled = bool(phase_cfg.get("enabled", False))
        phase_embedding_dim = int(phase_cfg.get("embedding_dim", 64))
    anchor_key = cfg.get("residual_anchor_key")
    residual_anchor_index = None
    if anchor_key is not None:
        if anchor_key not in input_keys:
            raise ValueError(f"generator.residual_anchor_key={anchor_key!r} is not present in data.input_keys")
        residual_anchor_index = input_keys.index(anchor_key)
    return ResidualUNet3D(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        base_channels=int(cfg["base_channels"]),
        channel_multipliers=tuple(int(v) for v in cfg["channel_multipliers"]),
        attention=bool(cfg.get("attention", True)),
        checkpoint_blocks=bool(cfg.get("checkpoint", False)),
        norm=str(cfg.get("norm", "group")),
        norm_groups=int(cfg.get("norm_groups", 8)),
        phase_conditioning=phase_enabled,
        phase_embedding_dim=phase_embedding_dim,
        residual_anchor_index=residual_anchor_index,
    )


def build_discriminator(config: dict) -> nn.Module:
    cfg = config["model"]["discriminator"]
    input_keys = config["data"].get("input_keys", [])
    target_keys = config["data"].get("target_keys", [])
    expected_channels = len(input_keys) + len(target_keys)
    if expected_channels and int(cfg["in_channels"]) != expected_channels:
        raise ValueError(f"discriminator.in_channels={cfg['in_channels']} but input+target channels={expected_channels}")
    kwargs = {
        "in_channels": int(cfg["in_channels"]),
        "base_channels": int(cfg["base_channels"]),
        "channel_multipliers": tuple(int(v) for v in cfg["channel_multipliers"]),
    }
    if bool(cfg.get("multiscale", False)):
        return MultiScalePatchDiscriminator3D(num_scales=int(cfg.get("num_scales", 2)), **kwargs)
    return PatchDiscriminator3D(**kwargs)
