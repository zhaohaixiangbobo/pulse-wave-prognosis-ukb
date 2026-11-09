"""
文件说明：定义 SAGM-Net（Structured-Anchored Gated Multimodal Network）。

本文件提供面向 P4205 短波形的 CNN、ResNet1D、PatchTST 和 InceptionTime-lite 编码器，以及以结构化
临床特征为主干、动态注入 P4205 raw waveform 和 engineered features 的门控多模态模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from two_tower_torch_model import TabularMLP


GATED_FULL_MODE = "sagm_gated"
VECTOR_GATED_FULL_MODE = "sagm_vector_gated"
GATED_RAW_ONLY_MODE = "sagm_gated_raw_only"
GATED_FEATURE_ONLY_MODE = "sagm_gated_feature_only"
GATED_CROSS_FEATURE_MODE = "sagm_gated_cross_feature"
GATED_MODES = {
    GATED_FULL_MODE,
    GATED_RAW_ONLY_MODE,
    GATED_FEATURE_ONLY_MODE,
    GATED_CROSS_FEATURE_MODE,
}


class GatedLinearUnit(nn.Module):
    """GLU 门控线性层：沿用 ICR 变量选择网络的核心门控单元。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """带残差、GLU 和 LayerNorm 的小型特征变换模块。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = F.relu(self.hidden(x), inplace=True)
        transformed = self.dropout(self.output(transformed))
        return self.norm(self.skip(x) + self.glu(transformed))


class GroupedVariableSelectionEncoder(nn.Module):
    """字段级 VSN 编码器。

    输入仍是训练折内完成填补、标准化和编码后的 dense 向量；group_indices 将同一原始字段
    的数值、one-hot 与缺失指示重新绑定，再用 ICR 风格的软选择权重得到塔 embedding。
    """

    def __init__(
        self,
        in_dim: int,
        group_indices: dict[str, list[int] | tuple[int, ...]],
        embed_dim: int,
        variable_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or embed_dim <= 0 or variable_dim <= 0:
            raise ValueError("VSN 维度必须为正数。")
        if not group_indices:
            raise ValueError("VSN 至少需要一个非空变量组。")

        normalized: dict[str, tuple[int, ...]] = {}
        used_indices: list[int] = []
        for name, values in group_indices.items():
            indices = tuple(int(value) for value in values)
            if not indices:
                raise ValueError(f"VSN 变量组 {name} 为空。")
            if min(indices) < 0 or max(indices) >= in_dim:
                raise ValueError(f"VSN 变量组 {name} 的索引超出输入维度。")
            normalized[str(name)] = indices
            used_indices.extend(indices)
        if sorted(used_indices) != list(range(in_dim)):
            raise ValueError("VSN 变量组必须无重叠且完整覆盖全部编码列。")

        self.group_names = tuple(normalized)
        self.group_indices = normalized
        self.variable_dim = int(variable_dim)
        self.input_projections = nn.ModuleDict(
            {name: nn.Linear(len(indices), variable_dim) for name, indices in normalized.items()}
        )
        self.variable_grns = nn.ModuleDict(
            {
                name: GatedResidualNetwork(variable_dim, max(variable_dim, embed_dim), variable_dim, dropout)
                for name in normalized
            }
        )
        self.context_grn = GatedResidualNetwork(
            len(normalized) * variable_dim,
            embed_dim,
            embed_dim,
            dropout,
        )
        self.weight_head = nn.Linear(embed_dim, len(normalized))
        self.output_projection = nn.Sequential(nn.Linear(variable_dim, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError("VSN 输入必须是 [B, F] 的二维张量。")
        variables: list[torch.Tensor] = []
        for name in self.group_names:
            indices = self.group_indices[name]
            selected = x[:, list(indices)]
            variables.append(self.variable_grns[name](self.input_projections[name](selected)))
        stacked = torch.stack(variables, dim=1)
        context = self.context_grn(torch.cat(variables, dim=1))
        weights = torch.softmax(self.weight_head(context), dim=1)
        selected_embedding = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return self.output_projection(selected_embedding), weights

class ResidualConvBlock1D(nn.Module):
    """短序列 1D ResNet 残差块，适合 100 点 P4205 波形。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.BatchNorm1d(out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class WaveEncoderResNet1D(nn.Module):
    """轻量 ResNet1D 波形编码器。"""

    def __init__(self, in_channels: int, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock1D(32, 32, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ResidualConvBlock1D(32, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ResidualConvBlock1D(64, 128, kernel_size=3, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        return self.proj(self.pool(self.blocks(self.stem(x))))


class InceptionBranch1D(nn.Module):
    """InceptionTime-lite 的单个多尺度卷积分支。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        branch_channels = max(1, out_channels // len(kernel_sizes))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for kernel in kernel_sizes
            ]
        )
        merged_channels = branch_channels * len(kernel_sizes)
        self.out = nn.Sequential(
            nn.Conv1d(merged_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([branch(x) for branch in self.branches], dim=1))


class WaveEncoderInceptionLite1D(nn.Module):
    """轻量 InceptionTime 风格编码器，支持按阶段调整感受野和通道宽度。

    默认配置保持既有实验完全一致；调参实验只改变 ``kernel_sets`` 或
    ``stage_channels``，因此不会意外改动三塔的其他输入或融合逻辑。
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        dropout: float = 0.2,
        kernel_sets: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = (
            (3, 7, 15),
            (3, 9, 21),
            (3, 11, 31),
        ),
        stage_channels: tuple[int, int, int] = (64, 96, 128),
    ) -> None:
        super().__init__()
        if len(kernel_sets) != 3 or any(len(values) != 3 for values in kernel_sets):
            raise ValueError("InceptionTime-lite 必须提供 3 个阶段、每阶段 3 个卷积核。")
        if len(stage_channels) != 3 or any(int(value) <= 0 for value in stage_channels):
            raise ValueError("InceptionTime-lite 必须提供 3 个正整数阶段通道数。")
        for values in kernel_sets:
            if any(int(kernel) <= 0 or int(kernel) % 2 == 0 for kernel in values):
                raise ValueError("InceptionTime-lite 卷积核必须是正奇数，才能保持波形长度不变。")

        c1, c2, c3 = (int(value) for value in stage_channels)
        self.kernel_sets = tuple(tuple(int(kernel) for kernel in values) for values in kernel_sets)
        self.stage_channels = (c1, c2, c3)
        self.net = nn.Sequential(
            InceptionBranch1D(in_channels, c1, kernel_sizes=self.kernel_sets[0]),
            nn.Dropout(dropout),
            InceptionBranch1D(c1, c2, kernel_sizes=self.kernel_sets[1]),
            nn.Dropout(dropout),
            InceptionBranch1D(c2, c3, kernel_sizes=self.kernel_sets[2]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c3, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        return self.net(x)

class WaveEncoderCNN1D(nn.Module):
    """三层 CNN 波形编码器，作为短 P4205 波形的轻量卷积基线。"""

    def __init__(self, in_channels: int, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        return self.net(x)


class WaveEncoderPatchTST1D(nn.Module):
    """P4205 短波形的 PatchTST 分类适配器。

    各通道独立切分为 10 点 patch，经过共享 Transformer 编码后在通道间平均；
    这保留 PatchTST 的 channel-independent 思路，同时输出统一的分类表征。
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        dropout: float = 0.2,
        patch_size: int = 10,
        stride: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        max_patches: int = 32,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("PatchTST embed_dim 必须能被 num_heads 整除。")
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.max_patches = int(max_patches)
        self.patch_embed = nn.Linear(self.patch_size, embed_dim)
        self.position = nn.Parameter(torch.zeros(1, self.max_patches, embed_dim))
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("wave 输入必须是 [B, L] 或 [B, C, L]。")
        if x.shape[-1] < self.patch_size:
            raise ValueError("PatchTST 输入长度小于 patch_size。")
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        batch, channels, patch_count, _ = patches.shape
        if patch_count > self.max_patches:
            raise ValueError("PatchTST patch 数超过位置编码容量。")
        tokens = self.patch_embed(patches).reshape(batch * channels, patch_count, -1)
        tokens = tokens + self.position[:, :patch_count, :]
        encoded = self.encoder(tokens).mean(dim=1).reshape(batch, channels, -1)
        return self.norm(encoded.mean(dim=1))


def make_sagm_wave_encoder(
    name: str,
    in_channels: int,
    embed_dim: int,
    dropout: float,
    inception_kernel_sets: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None,
    inception_stage_channels: tuple[int, int, int] | None = None,
) -> nn.Module:
    """创建 SAGM 使用的短序列波形编码器。"""
    if name == "cnn":
        return WaveEncoderCNN1D(in_channels=in_channels, embed_dim=embed_dim, dropout=dropout)
    if name == "resnet1d":
        return WaveEncoderResNet1D(in_channels=in_channels, embed_dim=embed_dim, dropout=dropout)
    if name == "patchtst":
        return WaveEncoderPatchTST1D(in_channels=in_channels, embed_dim=embed_dim, dropout=dropout)
    if name == "inception_lite":
        return WaveEncoderInceptionLite1D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            kernel_sets=inception_kernel_sets or ((3, 7, 15), (3, 9, 21), (3, 11, 31)),
            stage_channels=inception_stage_channels or (64, 96, 128),
        )
    raise ValueError("wave_encoder 仅支持 cnn、resnet1d、patchtst 或 inception_lite。")



def make_tabular_tower(
    in_dim: int,
    encoder: str,
    group_indices: dict[str, list[int] | tuple[int, ...]] | None,
    embed_dim: int,
    variable_dim: int,
    dropout: float,
) -> nn.Module:
    """按配置构建传统 MLP 或字段级 VSN 塔。"""
    if encoder == "mlp":
        return TabularMLP(in_dim, embed_dim=embed_dim, dropout=dropout)
    if encoder == "vsn":
        if group_indices is None:
            raise ValueError("使用 VSN 时必须提供完整的变量组索引。")
        return GroupedVariableSelectionEncoder(
            in_dim=in_dim,
            group_indices=group_indices,
            embed_dim=embed_dim,
            variable_dim=variable_dim,
            dropout=dropout,
        )
    raise ValueError("tabular encoder 仅支持 mlp 或 vsn。")
class SAGMNet(nn.Module):
    """结构化主干锚定的 P4205 门控多模态网络。

    ablation_mode 用于固定同一训练脚本下的严格消融。模型只创建当前消融真正
    需要的分支，避免 structured-only 这类基线被无关 waveform encoder 影响。
    """

    def __init__(
        self,
        structured_dim: int,
        p4205_feature_dim: int,
        wave_channels: int,
        output_dim: int = 1,
        wave_encoder: str = "resnet1d",
        ablation_mode: str = "sagm_gated",
        embed_dim: int = 128,
        dropout: float = 0.2,
        structured_encoder: str = "mlp",
        p4205_feature_encoder: str = "mlp",
        structured_group_indices: dict[str, list[int] | tuple[int, ...]] | None = None,
        p4205_group_indices: dict[str, list[int] | tuple[int, ...]] | None = None,
        vsn_variable_dim: int = 32,
        inception_kernel_sets: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None,
        inception_stage_channels: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.ablation_mode = ablation_mode
        self.output_dim = int(output_dim)
        self.structured_encoder_name = structured_encoder
        self.p4205_feature_encoder_name = p4205_feature_encoder
        self.uses_structured = ablation_mode in {
            "structured_only",
            "two_tower_raw",
            "structured_engineered",
            "structured_p4205_cross",
            "three_tower_concat",
            "sagm_gated",
            "sagm_vector_gated",
            "sagm_gated_raw_only",
            "sagm_gated_feature_only",
            "sagm_gated_cross_feature",
            "sagm_gated_cross_three_tower",
            "sagm_residual_delta",
        }
        self.uses_wave = ablation_mode in {
            "raw_waveform_only",
            "two_tower_raw",
            "three_tower_concat",
            "sagm_gated",
            "sagm_vector_gated",
            "sagm_gated_raw_only",
            "sagm_gated_cross_three_tower",
            "sagm_residual_delta",
        }
        self.uses_p4205_feature = ablation_mode in {
            "engineered_p4205_only",
            "engineered_p4205_cross_only",
            "structured_engineered",
            "structured_p4205_cross",
            "three_tower_concat",
            "sagm_gated",
            "sagm_vector_gated",
            "sagm_gated_feature_only",
            "sagm_gated_cross_feature",
            "sagm_gated_cross_three_tower",
            "sagm_residual_delta",
        }
        if not (self.uses_structured or self.uses_wave or self.uses_p4205_feature):
            raise ValueError(f"未知 ablation_mode：{ablation_mode}")

        self.structured_tower = (
            make_tabular_tower(
                structured_dim,
                structured_encoder,
                structured_group_indices,
                embed_dim,
                vsn_variable_dim,
                dropout,
            )
            if self.uses_structured
            else None
        )
        self.p4205_feature_tower = (
            make_tabular_tower(
                p4205_feature_dim,
                p4205_feature_encoder,
                p4205_group_indices,
                embed_dim,
                vsn_variable_dim,
                dropout,
            )
            if self.uses_p4205_feature
            else None
        )
        self.wave_tower = (
            make_sagm_wave_encoder(
                wave_encoder,
                wave_channels,
                embed_dim,
                dropout,
                inception_kernel_sets=inception_kernel_sets,
                inception_stage_channels=inception_stage_channels,
            )
            if self.uses_wave
            else None
        )
        self.wave_projection = (
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
            if ablation_mode in {"sagm_gated", "sagm_vector_gated", "sagm_gated_raw_only", "sagm_gated_cross_three_tower"}
            else None
        )
        self.feature_projection = (
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
            if ablation_mode in {"sagm_gated", "sagm_vector_gated", "sagm_gated_feature_only", "sagm_gated_cross_feature", "sagm_gated_cross_three_tower"}
            else None
        )
        self.gate_wave = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            if ablation_mode in {"sagm_gated", "sagm_gated_raw_only", "sagm_gated_cross_three_tower"}
            else None
        )
        # 向量 gate 只服务严格三塔全模型：两座 P4205 塔各自生成 embed_dim 维的逐表示门控。

        self.vector_gate_wave = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, embed_dim),
                nn.Sigmoid(),
            )
            if ablation_mode == VECTOR_GATED_FULL_MODE
            else None
        )
        self.vector_gate_feature = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, embed_dim),
                nn.Sigmoid(),
            )
            if ablation_mode == VECTOR_GATED_FULL_MODE
            else None
        )
        # 仅向量融合在相加后归一化，标量 gate 保持历史实现不变以保证直接比较。

        self.vector_fusion_norm = nn.LayerNorm(embed_dim) if ablation_mode == VECTOR_GATED_FULL_MODE else None
        self.delta_gate = (
            nn.Sequential(
                nn.Linear(embed_dim * 3, 96),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(96, 1),
                nn.Sigmoid(),
            )
            if ablation_mode == "sagm_residual_delta"
            else None
        )
        self.delta_fusion = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            if ablation_mode == "sagm_residual_delta"
            else None
        )
        self.delta_head = nn.Linear(64, self.output_dim) if ablation_mode == "sagm_residual_delta" else None
        self.gate_feature = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            if ablation_mode in {"sagm_gated", "sagm_gated_feature_only", "sagm_gated_cross_feature", "sagm_gated_cross_three_tower"}
            else None
        )
        self.concat_fusion = (
            nn.Sequential(
                nn.Linear(embed_dim * 3, 192),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(192, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            if ablation_mode == "three_tower_concat"
            else None
        )
        self.two_fusion = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            if ablation_mode in {"two_tower_raw", "structured_engineered", "structured_p4205_cross"}
            else None
        )
        self.anchor_head = (
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, self.output_dim),
            )
            if ablation_mode in {
                "structured_only",
                "sagm_gated",
            "sagm_vector_gated",
            "sagm_gated_raw_only",
                "sagm_gated_feature_only",
                "sagm_gated_cross_feature",
                "sagm_gated_cross_three_tower",
                "sagm_residual_delta",
            }
            else None
        )
        self.concat_head = (
            nn.Linear(64, self.output_dim)
            if ablation_mode in {"two_tower_raw", "structured_engineered", "structured_p4205_cross", "three_tower_concat"}
            else None
        )
        self.single_head = (
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, self.output_dim),
            )
            if ablation_mode in {"raw_waveform_only", "engineered_p4205_only", "engineered_p4205_cross_only"}
            else None
        )

    def forward(
        self,
        wave_x: torch.Tensor,
        p4205_x: torch.Tensor,
        structured_x: torch.Tensor,
        return_aux: bool = False,
        gate_override: dict[str, torch.Tensor | float] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = int(structured_x.shape[0])
        zero = structured_x.new_zeros(batch_size)
        structured_weights = structured_x.new_zeros((batch_size, 0))
        p4205_weights = structured_x.new_zeros((batch_size, 0))
        if isinstance(self.structured_tower, GroupedVariableSelectionEncoder):
            structured_emb, structured_weights = self.structured_tower(structured_x)
        else:
            structured_emb = self.structured_tower(structured_x) if self.structured_tower is not None else None
        wave_emb = self.wave_tower(wave_x) if self.wave_tower is not None else None
        if isinstance(self.p4205_feature_tower, GroupedVariableSelectionEncoder):
            feature_emb, p4205_weights = self.p4205_feature_tower(p4205_x)
        else:
            feature_emb = self.p4205_feature_tower(p4205_x) if self.p4205_feature_tower is not None else None
        gate_wave = zero
        gate_feature = zero
        # 非向量 gate 模式保留空矩阵，保证审计导出接口统一。
        gate_wave_vector = structured_x.new_zeros((batch_size, 0))
        gate_feature_vector = structured_x.new_zeros((batch_size, 0))
        if self.ablation_mode == "structured_only":
            if structured_emb is None or self.anchor_head is None:
                raise RuntimeError("structured_only 分支未初始化。")
            logits = self.anchor_head(structured_emb)
        elif self.ablation_mode == "raw_waveform_only":
            if wave_emb is None or self.single_head is None:
                raise RuntimeError("raw_waveform_only 分支未初始化。")
            logits = self.single_head(wave_emb)
        elif self.ablation_mode in {"engineered_p4205_only", "engineered_p4205_cross_only"}:
            if feature_emb is None or self.single_head is None:
                raise RuntimeError(f"{self.ablation_mode} 分支未初始化。")
            logits = self.single_head(feature_emb)
        elif self.ablation_mode == "two_tower_raw":
            if structured_emb is None or wave_emb is None or self.two_fusion is None or self.concat_head is None:
                raise RuntimeError("two_tower_raw 分支未初始化。")
            fused = self.two_fusion(torch.cat([structured_emb, wave_emb], dim=1))
            logits = self.concat_head(fused)
        elif self.ablation_mode == "structured_engineered":
            if structured_emb is None or feature_emb is None or self.two_fusion is None or self.concat_head is None:
                raise RuntimeError("structured_engineered 分支未初始化。")
            fused = self.two_fusion(torch.cat([structured_emb, feature_emb], dim=1))
            logits = self.concat_head(fused)
        elif self.ablation_mode == "structured_p4205_cross":
            if structured_emb is None or feature_emb is None or self.two_fusion is None or self.concat_head is None:
                raise RuntimeError("structured_p4205_cross 分支未初始化。")
            fused = self.two_fusion(torch.cat([structured_emb, feature_emb], dim=1))
            logits = self.concat_head(fused)
        elif self.ablation_mode == "three_tower_concat":
            if (
                structured_emb is None
                or wave_emb is None
                or feature_emb is None
                or self.concat_fusion is None
                or self.concat_head is None
            ):
                raise RuntimeError("three_tower_concat 分支未初始化。")
            fused = self.concat_fusion(torch.cat([structured_emb, wave_emb, feature_emb], dim=1))
            logits = self.concat_head(fused)
        elif self.ablation_mode in {"sagm_gated", "sagm_gated_cross_three_tower"}:
            if (
                structured_emb is None
                or wave_emb is None
                or feature_emb is None
                or self.gate_wave is None
                or self.gate_feature is None
                or self.wave_projection is None
                or self.feature_projection is None
                or self.anchor_head is None
            ):
                raise RuntimeError(f"{self.ablation_mode} 分支未初始化。")
            gate_wave = self.gate_wave(torch.cat([structured_emb, wave_emb], dim=1)).squeeze(-1)
            gate_feature = self.gate_feature(torch.cat([structured_emb, feature_emb], dim=1)).squeeze(-1)
            # 解释性干预只覆盖明确指定的 gate。默认 None 路径不增加数值运算，
            # 因而与历史 forward 保持逐元素一致。
            if gate_override is not None:
                gate_wave = self._resolve_gate_override(gate_wave, gate_override.get("gate_wave"), "gate_wave")
                gate_feature = self._resolve_gate_override(
                    gate_feature, gate_override.get("gate_feature"), "gate_feature"
                )
            fused = (
                structured_emb
                + gate_wave.unsqueeze(1) * self.wave_projection(wave_emb)
                + gate_feature.unsqueeze(1) * self.feature_projection(feature_emb)
            )
            logits = self.anchor_head(fused)
        elif self.ablation_mode == VECTOR_GATED_FULL_MODE:
            if (
                structured_emb is None
                or wave_emb is None
                or feature_emb is None
                or self.vector_gate_wave is None
                or self.vector_gate_feature is None
                or self.vector_fusion_norm is None
                or self.wave_projection is None
                or self.feature_projection is None
                or self.anchor_head is None
            ):
                raise RuntimeError("sagm_vector_gated 分支未初始化。")
            gate_wave_vector = self.vector_gate_wave(torch.cat([structured_emb, wave_emb], dim=1))
            gate_feature_vector = self.vector_gate_feature(torch.cat([structured_emb, feature_emb], dim=1))
            # 仅为与标量 gate 输出表兼容而保存逐维均值；完整向量由独立审计文件导出。
            gate_wave = gate_wave_vector.mean(dim=1)
            gate_feature = gate_feature_vector.mean(dim=1)
            fused = self.vector_fusion_norm(
                structured_emb
                + gate_wave_vector * self.wave_projection(wave_emb)
                + gate_feature_vector * self.feature_projection(feature_emb)
            )
            logits = self.anchor_head(fused)
        elif self.ablation_mode == "sagm_gated_raw_only":
            if (
                structured_emb is None
                or wave_emb is None
                or self.gate_wave is None
                or self.wave_projection is None
                or self.anchor_head is None
            ):
                raise RuntimeError("sagm_gated_raw_only 分支未初始化。")
            gate_wave = self.gate_wave(torch.cat([structured_emb, wave_emb], dim=1)).squeeze(-1)
            fused = structured_emb + gate_wave.unsqueeze(1) * self.wave_projection(wave_emb)
            logits = self.anchor_head(fused)
        elif self.ablation_mode in {"sagm_gated_feature_only", "sagm_gated_cross_feature"}:
            if (
                structured_emb is None
                or feature_emb is None
                or self.gate_feature is None
                or self.feature_projection is None
                or self.anchor_head is None
            ):
                raise RuntimeError(f"{self.ablation_mode} 分支未初始化。")
            gate_feature = self.gate_feature(torch.cat([structured_emb, feature_emb], dim=1)).squeeze(-1)
            fused = structured_emb + gate_feature.unsqueeze(1) * self.feature_projection(feature_emb)
            logits = self.anchor_head(fused)
        elif self.ablation_mode == "sagm_residual_delta":
            if (
                structured_emb is None
                or wave_emb is None
                or feature_emb is None
                or self.anchor_head is None
                or self.delta_gate is None
                or self.delta_fusion is None
                or self.delta_head is None
            ):
                raise RuntimeError("sagm_residual_delta 分支未初始化。")
            # 中文说明：结构化分支给出基准 logit，P4205 分支只学习 gated residual correction。
            anchor_logits = self.anchor_head(structured_emb)
            p4205_delta = self.delta_head(self.delta_fusion(torch.cat([wave_emb, feature_emb], dim=1)))
            gate_wave = self.delta_gate(torch.cat([structured_emb, wave_emb, feature_emb], dim=1)).squeeze(-1)
            gate_feature = gate_wave
            logits = anchor_logits + gate_wave.unsqueeze(1) * p4205_delta
        else:
            raise ValueError(f"未知 ablation_mode：{self.ablation_mode}")

        if self.output_dim == 1:
            logits = logits.squeeze(-1)
        if not return_aux:
            return logits
        aux = {
            "gate_wave": gate_wave,
            "gate_feature": gate_feature,
            "gate_wave_vector": gate_wave_vector,
            "gate_feature_vector": gate_feature_vector,
            "structured_embedding_norm": (
                torch.linalg.vector_norm(structured_emb, dim=1) if structured_emb is not None else zero
            ),
            "wave_embedding_norm": torch.linalg.vector_norm(wave_emb, dim=1) if wave_emb is not None else zero,
            "p4205_feature_embedding_norm": (
                torch.linalg.vector_norm(feature_emb, dim=1) if feature_emb is not None else zero
            ),
            "structured_vsn_weights": structured_weights,
            "p4205_vsn_weights": p4205_weights,
        }
        return logits, aux

    @staticmethod
    def _resolve_gate_override(
        learned_gate: torch.Tensor,
        override: torch.Tensor | float | None,
        gate_name: str,
    ) -> torch.Tensor:
        """把标量或逐样本 gate 覆盖值安全转换到当前 batch；未指定时原样返回。"""
        if override is None:
            return learned_gate
        value = torch.as_tensor(override, dtype=learned_gate.dtype, device=learned_gate.device).reshape(-1)
        if value.numel() == 1:
            value = value.expand_as(learned_gate)
        elif value.shape != learned_gate.shape:
            raise ValueError(
                f"{gate_name} override 形状不匹配：expected={tuple(learned_gate.shape)}, actual={tuple(value.shape)}"
            )
        if not torch.isfinite(value).all() or bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError(f"{gate_name} override 必须是 [0, 1] 内的有限值。")
        return value







