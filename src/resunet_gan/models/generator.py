from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def make_norm_3d(channels: int, norm: str, groups: int) -> nn.Module:
    norm = norm.lower()
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "group":
        if groups < 1:
            raise ValueError("norm_groups must be at least 1")
        group_count = min(groups, channels)
        while channels % group_count != 0:
            group_count -= 1
        return nn.GroupNorm(group_count, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D normalization: {norm}")


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        attention: bool = False,
        checkpoint_block: bool = False,
        norm: str = "group",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.checkpoint_block = checkpoint_block
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm_3d(channels, norm, norm_groups),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm_3d(channels, norm, norm_groups),
        )
        self.attention = ChannelAttention3D(channels) if attention else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.block(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_block and self.training and x.requires_grad:
            residual = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            residual = self._forward_impl(x)
        return self.act(x + residual)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: bool,
        checkpoint_block: bool,
        downsample: bool,
        norm: str,
        norm_groups: int,
    ) -> None:
        super().__init__()
        stride = (1, 2, 2) if downsample else (1, 1, 1)
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            make_norm_3d(out_channels, norm, norm_groups),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResidualBlock3D(
            out_channels,
            attention=attention,
            checkpoint_block=checkpoint_block,
            norm=norm,
            norm_groups=norm_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.proj(x))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention: bool,
        checkpoint_block: bool,
        norm: str,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm_3d(out_channels, norm, norm_groups),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResidualBlock3D(
            out_channels,
            attention=attention,
            checkpoint_block=checkpoint_block,
            norm=norm,
            norm_groups=norm_groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.fuse(x))


class PhaseConditionedOutput3D(nn.Module):
    """Generate ordered phases with learned phase-specific feature modulation."""

    def __init__(self, channels: int, num_phases: int, embedding_dim: int) -> None:
        super().__init__()
        if num_phases < 1 or embedding_dim < 1:
            raise ValueError("num_phases and embedding_dim must be positive")
        self.phase_embeddings = nn.Embedding(num_phases, embedding_dim)
        self.modulation = nn.Linear(embedding_dim, 2 * channels)
        self.phase_heads = nn.ModuleList([nn.Conv3d(channels, 1, kernel_size=1) for _ in range(num_phases)])
        nn.init.normal_(self.phase_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.modulation.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for phase_index, head in enumerate(self.phase_heads):
            phase_id = torch.tensor(phase_index, device=x.device)
            gamma, beta = self.modulation(self.phase_embeddings(phase_id)).chunk(2)
            gamma = gamma.view(1, -1, 1, 1, 1)
            beta = beta.view(1, -1, 1, 1, 1)
            outputs.append(head(x * (1.0 + gamma) + beta))
        return torch.cat(outputs, dim=1)


class ResidualUNet3D(nn.Module):
    """3D Residual U-Net for signed SUB generation. Final output is linear."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 5,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention: bool = True,
        checkpoint_blocks: bool = False,
        norm: str = "group",
        norm_groups: int = 8,
        phase_conditioning: bool = False,
        phase_embedding_dim: int = 64,
        residual_anchor_index: int | None = None,
    ) -> None:
        super().__init__()
        if residual_anchor_index is not None and not 0 <= residual_anchor_index < in_channels:
            raise ValueError("residual_anchor_index must identify an input channel")
        self.residual_anchor_index = residual_anchor_index
        channels = [base_channels * m for m in channel_multipliers]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for index, ch in enumerate(channels):
            self.encoders.append(
                EncoderBlock(
                    prev,
                    ch,
                    attention=attention,
                    checkpoint_block=checkpoint_blocks,
                    downsample=index > 0,
                    norm=norm,
                    norm_groups=norm_groups,
                )
            )
            prev = ch
        self.bottleneck = ResidualBlock3D(
            channels[-1],
            attention=attention,
            checkpoint_block=checkpoint_blocks,
            norm=norm,
            norm_groups=norm_groups,
        )
        decoders: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        current = reversed_channels[0]
        for skip_ch in reversed_channels[1:]:
            decoders.append(
                DecoderBlock(
                    current,
                    skip_ch,
                    skip_ch,
                    attention=attention,
                    checkpoint_block=checkpoint_blocks,
                    norm=norm,
                    norm_groups=norm_groups,
                )
            )
            current = skip_ch
        self.decoders = nn.ModuleList(decoders)
        if phase_conditioning:
            self.out = PhaseConditionedOutput3D(channels[0], out_channels, phase_embedding_dim)
        else:
            self.out = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = None
        if self.residual_anchor_index is not None:
            anchor = x[:, self.residual_anchor_index : self.residual_anchor_index + 1]
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        x = self.bottleneck(skips[-1])
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = decoder(x, skip)
        output = self.out(x)
        if anchor is not None:
            output = output + anchor
        return output
