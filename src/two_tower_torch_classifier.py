"""
文件说明：Two-Tower 分类模型定义。

本文件同时保留旧的多分类 logits 模型，并新增 10 年死亡风险二分类专用模型。
二分类模型输出 `[B]` logit，训练脚本使用 sigmoid 概率、Focal BCE/Weighted BCE 等损失。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from two_tower_torch_model import TabularMLP, TwoTowerRegressor, WaveEncoder1DCNN, WaveEncoderTCN, WaveEncoderTransformer


def make_wave_tower(wave_encoder: str, wave_in_channels: int, embed_dim: int, dropout: float) -> nn.Module:
    """创建 CNN 或 TCN 波形塔。"""
    if wave_encoder == "cnn":
        return WaveEncoder1DCNN(in_channels=wave_in_channels, embed_dim=embed_dim)
    if wave_encoder == "tcn":
        return WaveEncoderTCN(in_channels=wave_in_channels, embed_dim=embed_dim, dropout=dropout)
    if wave_encoder == "transformer":
        return WaveEncoderTransformer(in_channels=wave_in_channels, embed_dim=embed_dim, dropout=dropout)
    raise ValueError("wave_encoder 仅支持 'cnn' 或 'tcn'。")


class TwoTowerClassifier(nn.Module):
    """旧版多分类双塔：波形塔 + 结构化塔 + concat 融合，输出 `[B, C]` logits。"""

    def __init__(
        self,
        tabular_in_dim: int,
        num_classes: int = 5,
        wave_encoder: str = "tcn",
        tower_embed_dim: int = 128,
        fusion_hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        wave_in_channels: int = 1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")

        backbone = TwoTowerRegressor(
            tabular_in_dim=tabular_in_dim,
            wave_encoder=wave_encoder,
            tower_embed_dim=tower_embed_dim,
            fusion_hidden_dims=fusion_hidden_dims,
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
        self.wave_tower = backbone.wave_tower
        self.tab_tower = backbone.tab_tower
        self.fusion = backbone.fusion
        self.head = nn.Linear(backbone.head.in_features, num_classes)

    def forward(self, wave_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        tab_emb = self.tab_tower(tabular_x)
        fused = torch.cat([wave_emb, tab_emb], dim=1)
        fused = self.fusion(fused) if len(self.fusion) > 0 else fused
        return self.head(fused)


class WaveOnlyClassifier(nn.Module):
    """旧版多分类波形单塔，输出 `[B, C]` logits。"""

    def __init__(
        self,
        num_classes: int = 5,
        wave_encoder: str = "tcn",
        embed_dim: int = 128,
        dropout: float = 0.1,
        wave_in_channels: int = 1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")

        self.wave_tower = make_wave_tower(wave_encoder, wave_in_channels, embed_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, wave_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        return self.head(self.mlp(wave_emb))


class TabularOnlyClassifier(nn.Module):
    """旧版多分类结构化单塔，输出 `[B, C]` logits。"""

    def __init__(
        self,
        tabular_in_dim: int,
        num_classes: int = 5,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")

        dims = (tabular_in_dim,) + hidden_dims
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(nn.BatchNorm1d(dims[index + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1] if hidden_dims else tabular_in_dim, num_classes)

    def forward(self, tabular_x: torch.Tensor) -> torch.Tensor:
        if tabular_x.dim() != 2:
            raise ValueError("tabular 输入必须是 [B, F] 的二维张量。")
        x = self.backbone(tabular_x) if len(self.backbone) > 0 else tabular_x
        return self.head(x)


class BinaryTwoTowerClassifier(nn.Module):
    """二分类 concat 双塔：输出 `[B]` logit。"""

    def __init__(
        self,
        tabular_in_dim: int,
        wave_encoder: str = "cnn",
        tower_embed_dim: int = 128,
        fusion_hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        wave_in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.wave_tower = make_wave_tower(wave_encoder, wave_in_channels, tower_embed_dim, dropout)
        self.tab_tower = TabularMLP(tabular_in_dim, embed_dim=tower_embed_dim, dropout=dropout)

        dims = (tower_embed_dim * 2,) + fusion_hidden_dims
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.fusion = nn.Sequential(*layers)
        self.head = nn.Linear(fusion_hidden_dims[-1] if fusion_hidden_dims else tower_embed_dim * 2, 1)

    def forward(self, wave_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        tab_emb = self.tab_tower(tabular_x)
        fused = torch.cat([wave_emb, tab_emb], dim=1)
        fused = self.fusion(fused) if len(self.fusion) > 0 else fused
        return self.head(fused).squeeze(-1)


class BinaryGatedTwoTowerClassifier(nn.Module):
    """二分类 gated 双塔：让模型按样本动态控制波形塔贡献。"""

    def __init__(
        self,
        tabular_in_dim: int,
        wave_encoder: str = "cnn",
        tower_embed_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.2,
        wave_in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.wave_tower = make_wave_tower(wave_encoder, wave_in_channels, tower_embed_dim, dropout)
        self.tab_tower = TabularMLP(tabular_in_dim, embed_dim=tower_embed_dim, dropout=dropout)
        self.wave_projection = nn.Sequential(
            nn.Linear(tower_embed_dim, tower_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(tower_embed_dim * 2, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(tower_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, wave_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        tab_emb = self.tab_tower(tabular_x)
        gate = self.gate_mlp(torch.cat([tab_emb, wave_emb], dim=1))
        fused = tab_emb + gate * self.wave_projection(wave_emb)
        return self.head(fused).squeeze(-1)


class BinaryThreeTowerClassifier(nn.Module):
    """二分类三塔：原始波形塔 + p4205 人工特征塔 + 基线结构化塔。"""

    def __init__(
        self,
        tabular_in_dim: int,
        p4205_feature_dim: int,
        wave_encoder: str = "cnn",
        tower_embed_dim: int = 128,
        fusion_hidden_dims: tuple[int, ...] = (192, 64),
        dropout: float = 0.2,
        wave_in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.wave_tower = make_wave_tower(wave_encoder, wave_in_channels, tower_embed_dim, dropout)
        self.p4205_feature_tower = TabularMLP(p4205_feature_dim, embed_dim=tower_embed_dim, dropout=dropout)
        self.tab_tower = TabularMLP(tabular_in_dim, embed_dim=tower_embed_dim, dropout=dropout)

        dims = (tower_embed_dim * 3,) + fusion_hidden_dims
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.fusion = nn.Sequential(*layers)
        self.head = nn.Linear(fusion_hidden_dims[-1] if fusion_hidden_dims else tower_embed_dim * 3, 1)

    def forward(self, wave_x: torch.Tensor, p4205_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        p4205_emb = self.p4205_feature_tower(p4205_x)
        tab_emb = self.tab_tower(tabular_x)
        fused = torch.cat([wave_emb, p4205_emb, tab_emb], dim=1)
        fused = self.fusion(fused) if len(self.fusion) > 0 else fused
        return self.head(fused).squeeze(-1)


class BinaryGatedThreeTowerClassifier(nn.Module):
    """二分类 gated 三塔：以结构化塔为主干，动态接收 p4205 波形和人工特征增量。"""

    def __init__(
        self,
        tabular_in_dim: int,
        p4205_feature_dim: int,
        wave_encoder: str = "cnn",
        tower_embed_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.2,
        wave_in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.wave_tower = make_wave_tower(wave_encoder, wave_in_channels, tower_embed_dim, dropout)
        self.p4205_feature_tower = TabularMLP(p4205_feature_dim, embed_dim=tower_embed_dim, dropout=dropout)
        self.tab_tower = TabularMLP(tabular_in_dim, embed_dim=tower_embed_dim, dropout=dropout)
        self.p4205_projection = nn.Sequential(
            nn.Linear(tower_embed_dim * 2, tower_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(tower_embed_dim * 3, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(tower_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, wave_x: torch.Tensor, p4205_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        wave_emb = self.wave_tower(wave_x)
        p4205_emb = self.p4205_feature_tower(p4205_x)
        tab_emb = self.tab_tower(tabular_x)
        p4205_pair = torch.cat([wave_emb, p4205_emb], dim=1)
        gate = self.gate_mlp(torch.cat([tab_emb, wave_emb, p4205_emb], dim=1))
        fused = tab_emb + gate * self.p4205_projection(p4205_pair)
        return self.head(fused).squeeze(-1)


class BinaryWaveOnlyClassifier(nn.Module):
    """二分类波形单塔：输出 `[B]` logit。"""

    def __init__(
        self,
        wave_encoder: str = "cnn",
        embed_dim: int = 128,
        dropout: float = 0.2,
        wave_in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.wave_tower = make_wave_tower(wave_encoder, wave_in_channels, embed_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, wave_x: torch.Tensor) -> torch.Tensor:
        return self.head(self.wave_tower(wave_x)).squeeze(-1)


class BinaryTabularOnlyClassifier(nn.Module):
    """二分类结构化单塔：输出 `[B]` logit。"""

    def __init__(
        self,
        tabular_in_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dims = (tabular_in_dim,) + hidden_dims
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(nn.BatchNorm1d(dims[index + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1] if hidden_dims else tabular_in_dim, 1)

    def forward(self, tabular_x: torch.Tensor) -> torch.Tensor:
        if tabular_x.dim() != 2:
            raise ValueError("tabular 输入必须是 [B, F] 的二维张量。")
        x = self.backbone(tabular_x) if len(self.backbone) > 0 else tabular_x
        return self.head(x).squeeze(-1)
