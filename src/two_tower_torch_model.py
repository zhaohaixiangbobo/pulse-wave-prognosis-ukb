"""
文件说明：Two-Tower Torch 模型的共享波形解析器、波形编码器和结构化特征塔。

本文件保持旧接口兼容：`parse_p4205_i0_to_tensor()` 默认返回一维 `[100]` 波形；
当显式启用 `return_channels=True` 或 `WaveformConfig.return_channels=True` 时，返回
双通道 `[2, 100]`，其中第 0 通道为标准化原始波形，第 1 通道为一阶差分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RANDOM_STATE = 42
TEST_SIZE = 0.15


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """设置 torch/cuda/numpy 随机种子，尽量保证实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_test_split_indices(
    n_samples: int,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """生成与历史脚本兼容的随机 train/test 索引。"""
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正数。")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size 必须在 (0, 1) 之间。")

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = int(np.ceil(n_samples * test_size))
    return indices[n_test:], indices[:n_test]


@dataclass(frozen=True)
class WaveformConfig:
    """p4205_i0 波形解析配置。"""

    expected_length: int = 100
    normalize: bool = True
    clip_value: float | None = None
    return_channels: bool = False
    channel_count: int = 2
    percentile_precision: str = "numpy_2_2"


def percentile_float32_241(raw, q):
    """复现NumPy 2.4.1标量分位数：float64位置、弱标量float32线性插值。

    来源：NumPy PR30623；仅用于读取2.4.1环境产生的已冻结主模型。
    用显式精度固定预处理，避免为旧权重反复切换整个Python环境。
    """
    ordered = np.sort(raw)
    position = (len(ordered) - 1) * (q / 100)
    lower = int(position)
    gamma = float(position - lower)
    left, right = ordered[lower], ordered[min(lower + 1, len(ordered) - 1)]
    difference = right - left
    return right - difference * (1 - gamma) if gamma >= .5 else left + difference * gamma


def _parse_wave_values(wave_str: Any) -> list[float]:
    values: list[float] = []
    if not isinstance(wave_str, str):
        return values

    text = wave_str.strip()
    if "|" not in text:
        return values

    for point in text.split("|")[1:]:
        if "," not in point:
            continue
        _, raw_value = point.split(",", 1)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def parse_p4205_i0_to_tensor(
    wave_str: Any,
    config: WaveformConfig = WaveformConfig(),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    return_channels: bool | None = None,
) -> torch.Tensor:
    """将 `p4205_i0` 字符串解析为定长 Tensor。

    默认返回 `[L]`，用于兼容旧回归/分类脚本。启用双通道后返回 `[2, L]`：
    第 0 通道为标准化波形，第 1 通道为一阶差分，最后一个差分点补 0。
    """
    use_channels = config.return_channels if return_channels is None else bool(return_channels)
    channel_count = int(config.channel_count)
    if channel_count not in {1, 2, 4, 5}:
        raise ValueError("channel_count 仅支持 1、2、4、5。")
    values = _parse_wave_values(wave_str)

    if not values:
        shape = (channel_count, config.expected_length) if use_channels else (config.expected_length,)
        return torch.zeros(shape, device=device, dtype=dtype)

    raw = np.asarray(values, dtype=np.float32)
    if raw.shape[0] >= config.expected_length:
        raw = raw[: config.expected_length]
    else:
        pad_width = config.expected_length - raw.shape[0]
        raw = np.pad(raw, (0, pad_width), mode="constant", constant_values=0.0)

    arr = raw.astype(np.float32, copy=True)
    if config.normalize:
        mean_value = float(arr.mean())
        std_value = float(arr.std())
        arr = (arr - mean_value) / std_value if std_value > 0 else arr - mean_value

    if config.clip_value is not None and config.clip_value > 0:
        arr = np.clip(arr, -config.clip_value, config.clip_value)

    if not use_channels:
        return torch.as_tensor(arr, device=device, dtype=dtype)

    derivative = np.diff(arr, append=arr[-1]).astype(np.float32)
    derivative[-1] = 0.0
    channels = [arr.astype(np.float32), derivative]
    if channel_count >= 4:
        raw_median = float(np.median(raw))
        if config.percentile_precision == "numpy_2_4_1":
            raw_scale = float(percentile_float32_241(raw, 95) - percentile_float32_241(raw, 5))
        elif config.percentile_precision == "numpy_2_2":
            raw_scale = float(np.percentile(raw, 95) - np.percentile(raw, 5))
        else:
            raise ValueError("未知波形分位数精度配置")
        raw_robust = (raw - raw_median) / raw_scale if raw_scale > 0 else raw - raw_median
        raw_diff = np.diff(raw_robust, append=raw_robust[-1]).astype(np.float32)
        raw_diff[-1] = 0.0
        channels.extend([raw_robust.astype(np.float32), raw_diff])
    if channel_count == 5:
        channels.append(np.linspace(0.0, 1.0, config.expected_length, dtype=np.float32))
    matrix = np.stack(channels[:channel_count], axis=0)
    return torch.as_tensor(matrix, device=device, dtype=dtype)


def batch_parse_p4205_i0(
    wave_list: Iterable[Any],
    config: WaveformConfig = WaveformConfig(),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    return_channels: bool | None = None,
) -> torch.Tensor:
    """批量解析 `p4205_i0` 字符串列表。"""
    tensors = [
        parse_p4205_i0_to_tensor(
            value,
            config=config,
            device=device,
            dtype=dtype,
            return_channels=return_channels,
        )
        for value in wave_list
    ]
    return torch.stack(tensors, dim=0)


class WaveEncoder1DCNN(nn.Module):
    """使用 1D CNN 将单通道或双通道波形编码为定长 embedding。"""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 `[B, L]` 或 `[B, C, L]`，输出 `[B, embed_dim]`。"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        feats = self.net(x).squeeze(-1)
        return self.proj(feats)


class TCNBlock(nn.Module):
    """TCN 残差块：dilated Conv1d + 残差连接。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _chomp(x: torch.Tensor, chomp: int) -> torch.Tensor:
        return x if chomp <= 0 else x[:, :, :-chomp]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self._chomp(out, self.conv1.padding[0])
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self._chomp(out, self.conv2.padding[0])
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        return out + residual


class WaveEncoderTCN(nn.Module):
    """使用 TCN 将单通道或双通道波形编码为定长 embedding。"""

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TCNBlock(channels=channels, kernel_size=kernel_size, dilation=2**index, dropout=dropout)
                for index in range(num_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.out_proj = nn.Linear(channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 `[B, L]` 或 `[B, C, L]`，输出 `[B, embed_dim]`。"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        out = self.in_proj(x)
        out = self.blocks(out)
        out = self.pool(out).squeeze(-1)
        return self.out_proj(out)


class WaveEncoderTransformer(nn.Module):
    """轻量 Transformer 波形编码器，用于学习 p4205 多通道序列形态。"""

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_length: int = 100,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, max_length + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 `[B, L]` 或 `[B, C, L]`，输出 `[B, embed_dim]`。"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        tokens = self.input_proj(x.transpose(1, 2))
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0, :])


class TabularMLP(nn.Module):
    """结构化特征塔：dense 向量输入，输出 embedding。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = (in_dim,) + hidden_dims
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(nn.BatchNorm1d(dims[index + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dims[-1] if hidden_dims else in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("tabular 输入必须是 [B, F] 的二维张量。")
        out = self.backbone(x) if len(self.backbone) > 0 else x
        return self.proj(out)


class TwoTowerRegressor(nn.Module):
    """旧版回归双塔：保留用于历史回归脚本兼容。"""

    def __init__(
        self,
        tabular_in_dim: int,
        wave_encoder: str = "tcn",
        tower_embed_dim: int = 128,
        fusion_hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        wave_in_channels: int = 1,
    ) -> None:
        super().__init__()
        if wave_encoder not in {"cnn", "tcn"}:
            raise ValueError("wave_encoder 仅支持 'cnn' 或 'tcn'。")

        if wave_encoder == "cnn":
            self.wave_tower = WaveEncoder1DCNN(in_channels=wave_in_channels, embed_dim=tower_embed_dim)
        else:
            self.wave_tower = WaveEncoderTCN(in_channels=wave_in_channels, embed_dim=tower_embed_dim)

        self.tab_tower = TabularMLP(in_dim=tabular_in_dim, embed_dim=tower_embed_dim, dropout=dropout)

        fusion_in = tower_embed_dim * 2
        dims = (fusion_in,) + fusion_hidden_dims
        fusion_layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            fusion_layers.append(nn.Linear(dims[index], dims[index + 1]))
            fusion_layers.append(nn.ReLU(inplace=True))
            fusion_layers.append(nn.Dropout(dropout))
        self.fusion = nn.Sequential(*fusion_layers)
        self.head = nn.Linear(fusion_hidden_dims[-1] if fusion_hidden_dims else fusion_in, 1)

    def forward(self, wave_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        tab_emb = self.tab_tower(tabular_x)
        fused = torch.cat([wave_emb, tab_emb], dim=1)
        fused = self.fusion(fused) if len(self.fusion) > 0 else fused
        return self.head(fused).squeeze(-1)
