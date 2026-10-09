from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchDiscriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 16,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for index, multiplier in enumerate(channel_multipliers):
            out_ch = base_channels * multiplier
            stride = (1, 2, 2) if index > 0 else (1, 1, 1)
            layers.append(nn.Conv3d(prev, out_ch, kernel_size=4, stride=stride, padding=1))
            if index > 0:
                layers.append(nn.InstanceNorm3d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = out_ch
        layers.append(nn.Conv3d(prev, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


class MultiScalePatchDiscriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 16,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        num_scales: int = 2,
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator3D(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    channel_multipliers=channel_multipliers,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs, _ = self.forward_features(x)
        return outputs

    def forward_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs: list[torch.Tensor] = []
        all_features: list[list[torch.Tensor]] = []
        current = x
        for index, discriminator in enumerate(self.discriminators):
            logits, features = discriminator.forward_features(current)
            outputs.append(logits)
            all_features.append(features)
            if index != len(self.discriminators) - 1:
                current = F.avg_pool3d(current, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return outputs, all_features
