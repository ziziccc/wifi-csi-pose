from __future__ import annotations

from inspect import signature
from typing import Any

import torch
from torch import nn


class FastReceiverEncoder(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        mid_channels = max(out_channels // 2, 16)
        self.layers = nn.Sequential(
            nn.Conv2d(3, mid_channels, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiESPFastPoseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int = 24,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        receiver_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        if in_channels % 3 != 0:
            raise ValueError(
                "MultiESPFastPoseCNN expects input channels shaped as node_count * 3 "
                f"(base, delta, mask). Received in_channels={in_channels}."
            )

        self.node_count = in_channels // 3
        self.encoder = FastReceiverEncoder(out_channels=int(receiver_feature_dim))

        flattened_dim = self.node_count * int(receiver_feature_dim) * 8 * 4
        fusion_dim = max(hidden_dim, 96)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _slice_receiver_channels(self, x: torch.Tensor, node_index: int) -> torch.Tensor:
        base = x[:, node_index : node_index + 1, :, :]
        delta = x[:, self.node_count + node_index : self.node_count + node_index + 1, :, :]
        mask = x[:, self.node_count * 2 + node_index : self.node_count * 2 + node_index + 1, :, :]
        return torch.cat([base, delta, mask], dim=1)

    def _encode_receivers(self, x: torch.Tensor) -> torch.Tensor:
        receiver_features: list[torch.Tensor] = []
        for node_index in range(self.node_count):
            receiver_features.append(self.encoder(self._slice_receiver_channels(x, node_index)))
        return torch.cat(receiver_features, dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        combined = self._encode_receivers(x)
        hidden = combined
        for layer in self.head[:-1]:
            hidden = layer(hidden)
        return hidden

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head[-1](features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.predict_from_features(features)


MODEL_REGISTRY = {
    "multi_esp_fast_pose_cnn": MultiESPFastPoseCNN,
}


def build_model(model_name: str, **kwargs: object) -> nn.Module:
    try:
        builder = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name={model_name!r}. Available: {available}") from exc
    accepted = signature(builder).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return builder(**filtered_kwargs)


def build_model_from_checkpoint(checkpoint_path: str) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) or {}
    model_name = str(checkpoint.get("model_name") or config.get("model_name"))
    if model_name != "multi_esp_fast_pose_cnn":
        raise ValueError(f"ML2 only supports multi_esp_fast_pose_cnn checkpoints, received {model_name!r}.")
    input_channels = int(checkpoint.get("input_channels"))
    hidden_dim = int(config.get("hidden_dim", 128))
    dropout = float(config.get("dropout", 0.15))

    model = build_model(
        model_name,
        in_channels=input_channels,
        output_dim=24,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
