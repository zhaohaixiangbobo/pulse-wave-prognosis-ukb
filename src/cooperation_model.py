"""在本地SAGM副本上实现两种独立协作机制，保持原三塔及双塔输入边界。"""
import math
import torch
from torch import nn
from sagm_torch_model import SAGMNet
from two_tower_torch_model import TabularMLP


class CooperativeSAGMNet(SAGMNet):
    """none为原标量门控，joint为联合条件门控，aux为训练辅助监督。"""

    def __init__(self, cooperation='none', auxiliary_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        if cooperation not in ('none', 'joint', 'aux', 'bilinear', 'additive_control'):
            raise ValueError(cooperation)
        if self.ablation_mode not in ('sagm_gated', 'sagm_gated_raw_only', 'sagm_gated_feature_only'):
            raise ValueError('协作实验仅支持完整三塔及两个配套双塔。')
        self.cooperation = cooperation
        self.auxiliary_weight = float(auxiliary_weight)
        self.collect_cooperation_diagnostics = True
        self.inference_override = None
        self.interaction_a = self.interaction_b = self.interaction_q = None
        if cooperation in ('bilinear', 'additive_control') and self.ablation_mode == 'sagm_gated':
            # 保存随机流：原共享层与原Full同初值，新模块不改变后续Dropout随机流。
            with torch.random.fork_rng():
                self.interaction_a = nn.Linear(128, 32, bias=False)
                self.interaction_b = nn.Linear(128, 32, bias=False)
                self.interaction_q = nn.Linear(32, 1, bias=False)
                nn.init.zeros_(self.interaction_q.weight)
        if cooperation == 'joint':
            dim = kwargs.get('embed_dim', 128)
            dropout = kwargs.get('dropout', 0.2)

            def gate():
                return nn.Sequential(nn.Linear(dim * 3, 64), nn.ReLU(inplace=True),
                                     nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
            if self.uses_wave:
                self.gate_wave = gate()
            if self.uses_p4205_feature:
                self.gate_feature = gate()

    def _representations(self, wave_x, p4205_x, structured_x, gate_override=None):
        """原模式保持编码、门控、投影的调用顺序，确保identity可数值复现。"""
        s = self.structured_tower(structured_x)
        w = self.wave_tower(wave_x) if self.uses_wave else torch.zeros_like(s)
        e = self.p4205_feature_tower(
            p4205_x) if self.uses_p4205_feature else torch.zeros_like(s)
        zero = s.new_zeros(len(s))
        if self.cooperation == 'joint':
            context = torch.cat([s, w, e], dim=1)
            gw = self.gate_wave(
                context).squeeze(-1) if self.uses_wave else zero
            ge = self.gate_feature(
                context).squeeze(-1) if self.uses_p4205_feature else zero
        else:
            gw = self.gate_wave(
                torch.cat([s, w], dim=1)).squeeze(-1) if self.uses_wave else zero
            ge = self.gate_feature(
                torch.cat([s, e], dim=1)).squeeze(-1) if self.uses_p4205_feature else zero
        override = gate_override if gate_override is not None else self.inference_override
        if override:
            gw = self._resolve_gate_override(
                gw, override.get('gate_wave'), 'gate_wave')
            ge = self._resolve_gate_override(
                ge, override.get('gate_feature'), 'gate_feature')
        uw = gw.unsqueeze(
            1) * self.wave_projection(w) if self.uses_wave else torch.zeros_like(s)
        ue = ge.unsqueeze(
            1) * self.feature_projection(e) if self.uses_p4205_feature else torch.zeros_like(s)
        if self.uses_wave and self.uses_p4205_feature:
            fused = s + uw + ue
        else:
            fused = s + (uw if self.uses_wave else ue)
        empty = s.new_zeros((len(s), 0))
        aux = {'gate_wave': gw, 'gate_feature': ge, 'gate_wave_vector': empty, 'gate_feature_vector': empty,
               'structured_vsn_weights': empty, 'p4205_vsn_weights': empty,
               'structured_embedding_norm': torch.linalg.vector_norm(s, dim=1),
               'wave_embedding_norm': torch.linalg.vector_norm(w, dim=1),
               'p4205_feature_embedding_norm': torch.linalg.vector_norm(e, dim=1),
               'wave_contribution_norm': torch.linalg.vector_norm(uw, dim=1),
               'feature_contribution_norm': torch.linalg.vector_norm(ue, dim=1)}
        return fused, s, uw, ue, aux

    def _head(self, value):
        logits = self.anchor_head(value)
        return logits.squeeze(-1) if self.output_dim == 1 else logits

    def forward(self, wave_x, p4205_x, structured_x, return_aux=False, gate_override=None):
        fused, _, uw, ue, aux = self._representations(
            wave_x, p4205_x, structured_x, gate_override)
        logits = self._head(fused)
        if self.interaction_q is not None:
            a, b = torch.tanh(self.interaction_a(
                uw)), torch.tanh(self.interaction_b(ue))
            interaction = a * b if self.cooperation == 'bilinear' else a + b
            aux['interaction_product'] = interaction
            logits = logits + self.interaction_q(interaction).squeeze(-1)
        return (logits, aux) if return_aux else logits

    def cooperation_loss(self, wave, features, structured, labels, masks, criterion):
        """辅助权重为零时走原前向路径；双塔不重复监督同一路径。"""
        if self.cooperation != 'aux' or self.auxiliary_weight == 0 or self.ablation_mode != 'sagm_gated':
            logits = self(wave, features, structured)
            return criterion(logits.reshape_as(labels), labels, masks)
        fused, s, uw, ue, _ = self._representations(wave, features, structured)
        full = criterion(self._head(fused).reshape_as(labels), labels, masks)
        raw = criterion(self._head(s + uw).reshape_as(labels), labels, masks)
        engineered = criterion(self._head(
            s + ue).reshape_as(labels), labels, masks)
        weight = self.auxiliary_weight
        return (full + weight * raw + weight * engineered) / (1 + 2 * weight)


class StagedSAGMNet(nn.Module):
    """严格冻结双塔/结构化anchor；缓存仅替换冻结前向，部署使用同一份anchor权重。"""

    def __init__(self, cooperation, anchor_kwargs, engineered_dim, initialization_seed=0):
        super().__init__()
        if cooperation not in ('staged_full', 'staged_capacity', 'staged_no_raw'):
            raise ValueError(cooperation)
        self.cooperation = cooperation
        self.uses_wave = cooperation != 'staged_no_raw'
        self.uses_p4205_feature = cooperation != 'staged_capacity'
        expected = 'sagm_gated_raw_only' if self.uses_wave else 'structured_only'
        if anchor_kwargs['ablation_mode'] != expected:
            raise ValueError('anchor架构与分支边界不一致')
        self.ablation_mode = 'sagm_gated'
        self.input_is_cached = False
        self.collect_cooperation_diagnostics = True
        self.include_epoch_zero = True
        with torch.random.fork_rng():
            self.anchor = SAGMNet(**anchor_kwargs)
            torch.manual_seed(initialization_seed)
            self.increment_tower = TabularMLP(engineered_dim if self.uses_p4205_feature else 256,
                                              embed_dim=128, dropout=0.2)
            # 维度不同的第一层不强行配对；其余同形状层用独立固定流逐层配对。
            for index, (name, module) in enumerate(self.increment_tower.named_modules()):
                if isinstance(module, nn.Linear) and name != 'backbone.0':
                    torch.manual_seed(initialization_seed + 100 + index)
                    module.reset_parameters()
            torch.manual_seed(initialization_seed + 50)
            first = self.increment_tower.backbone[0]
            # 不同输入宽度的第一层bias仍可配对，使用固定边界初始化。
            nn.init.uniform_(first.bias, -1 / math.sqrt(256),
                             1 / math.sqrt(256))
            torch.manual_seed(initialization_seed + 1)
            self.delta_head = nn.Sequential(
                nn.Linear(384, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        self.anchor.requires_grad_(False)
        self.anchor.eval()

    @property
    def wave_tower(self):
        return self.anchor.wave_tower

    @property
    def structured_tower(self):
        return self.anchor.structured_tower

    @property
    def p4205_feature_tower(self):
        return self.increment_tower if self.uses_p4205_feature else None

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    @torch.no_grad()
    def anchor_features(self, wave, structured):
        self.anchor.eval()
        s = self.anchor.structured_tower(structured)
        if self.uses_wave:
            w = self.anchor.wave_tower(wave)
            gate = self.anchor.gate_wave(torch.cat([s, w], dim=1))
            fused = s + gate * self.anchor.wave_projection(w)
        else:
            w, fused = torch.zeros_like(s), s
        z0 = self.anchor.anchor_head(fused)
        return torch.cat([s, w, z0], dim=1)

    def forward(self, wave_x, p4205_x, structured_x, return_aux=False):
        packed = structured_x if self.input_is_cached else self.anchor_features(
            wave_x, structured_x)
        if packed.shape[1] != 257:
            raise ValueError('冻结anchor缓存必须包含128维s、128维w及原始logit')
        s, w, z0 = packed[:, :128], packed[:, 128:256], packed[:, 256]
        e = self.increment_tower(
            p4205_x if self.uses_p4205_feature else torch.cat([s, w], dim=1))
        logits = z0 + self.delta_head(torch.cat([s, w, e], dim=1)).squeeze(-1)
        if not return_aux:
            return logits
        zero, empty = s.new_zeros(len(s)), s.new_zeros((len(s), 0))
        aux = {key: empty for key in (
            'gate_wave_vector', 'gate_feature_vector', 'structured_vsn_weights', 'p4205_vsn_weights')}
        # A没有新增gate；这些兼容字段为占位，不能解释为门控依赖强度。
        aux.update(gate_wave=zero, gate_feature=zero, structured_embedding_norm=s.norm(dim=1),
                   wave_embedding_norm=w.norm(dim=1), p4205_feature_embedding_norm=e.norm(dim=1),
                   wave_contribution_norm=zero, feature_contribution_norm=zero)
        return logits, aux

    def cooperation_loss(self, wave, features, structured, labels, masks, criterion):
        return criterion(self(wave, features, structured).reshape_as(labels), labels, masks)


def restore_model(checkpoint):
    """checkpoint包含整个anchor和新增模块，不依赖原实验代码或外部权重文件。"""
    kind = StagedSAGMNet if checkpoint['model_kwargs']['cooperation'].startswith(
        'staged_') else CooperativeSAGMNet
    model = kind(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def gradient_rms(model):
    """按参数数归一化的梯度RMS；只记录，不参与优化规则。"""
    result = {}
    for name in ('structured_tower', 'wave_tower', 'p4205_feature_tower'):
        module = getattr(model, name, None)
        gradients = [] if module is None else [
            p.grad for p in module.parameters() if p.grad is not None]
        count = sum(g.numel() for g in gradients)
        square = sum(g.detach().float().square().sum() for g in gradients)
        result[name +
               '_gradient_rms'] = math.sqrt(float(square) / count) if count else 0.0
    return result
