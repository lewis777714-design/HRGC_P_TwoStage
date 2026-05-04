"""
TwoStageModelV7 — Hierarchical multi-label classifier with flat-head fusion.

Architecture (top-down):

    BERT encoder (output_hidden_states=True)
       │
       ScalarMix over the 13 hidden states (learned weighted sum)
       │
       ┌──────────────── Stage 1 ────────────────┐
       │  MultiPool [CLS ; MeanPool ; AttnPool]  │  →  3 main-factor logits
       └─────────────────────────────────────────┘
       │
       │ (Stage-1 logits, scaled)  ─ used to condition Stage 2 ─
       │
       ┌──────────────── Stage 2 (per category) ─────────────────┐
       │  SemanticLabelAttention                                 │   attn logits
       │     ├─ label queries init from BERT-encoded label names │
       │     └─ cross-attention over BERT tokens                 │
       │  + FlatStage2Head (small MLP from mean-pool)            │   flat logits
       │  → final = σ(α) · attn + (1 − σ(α)) · flat              │   mixed logits
       └─────────────────────────────────────────────────────────┘

Loss:
    PerClassAsymmetricLoss with per-class positive-sample weighting.

Ablation switches exposed in the constructor:
    use_semantic_queries     – init label queries from BERT vs random
    stage1_pool ∈ {multi,cls,mean}
    use_s1_conditioning      – whether Stage-2 attention sees Stage-1 logits
    use_flat_head            – whether the flat fusion branch is built
    flat_alpha_init          – initial sigmoid(α) value (default 0.7)

Self-contained — only depends on torch / transformers.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────
# Pooling and attention utilities
# ─────────────────────────────────────────────────────────────────

class ScalarMix(nn.Module):
    """Learned weighted combination of BERT layer outputs (ELMo-style)."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.scalar_params = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, tensors):
        w = F.softmax(self.scalar_params, dim=0)
        stack = torch.stack(list(tensors), dim=0)              # [L, B, S, H]
        mixed = (stack * w.view(-1, 1, 1, 1)).sum(dim=0)       # [B, S, H]
        return self.gamma * mixed


class AttentionPooling(nn.Module):
    """Single-head additive attention pool over the token sequence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence_output, attention_mask):
        scores = self.attention(sequence_output).squeeze(-1)   # [B, S]
        # dtype-safe mask fill (works in fp16/bf16/fp32)
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(attention_mask == 0, neg)
        weights = F.softmax(scores, dim=-1)
        pooled = (sequence_output * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class MultiPoolStage1(nn.Module):
    """Concatenate [CLS ; MeanPool ; AttnPool] then project back to H."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
        )

    def forward(self, sequence_output, attention_mask):
        cls = sequence_output[:, 0]
        mask_f = attention_mask.float().unsqueeze(-1)
        mean = (sequence_output * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
        attn, attn_w = self.attn_pool(sequence_output, attention_mask)
        pooled = torch.cat([cls, mean, attn], dim=-1)
        return self.proj(pooled), attn_w


def mean_pool(sequence_output, attention_mask):
    m = attention_mask.float().unsqueeze(-1)
    return (sequence_output * m).sum(1) / m.sum(1).clamp(min=1e-6)


# ─────────────────────────────────────────────────────────────────
# Semantic label-aware attention (Stage 2 head)
# ─────────────────────────────────────────────────────────────────

class SemanticLabelAttention(nn.Module):
    """Cross-attention with K trainable label queries that can be initialised
    from BERT-encoded subcategory names.  Queries are conditioned on the
    Stage-1 main-factor logits (scaled), preserving confidence magnitude
    even when probabilities saturate near 1."""

    def __init__(self, hidden_size: int, num_labels: int,
                 num_factors: int = 3, num_heads: int = 4,
                 dropout: float = 0.1, init_queries: Optional[torch.Tensor] = None,
                 use_conditioning: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.use_conditioning = use_conditioning

        if init_queries is not None:
            assert init_queries.shape == (num_labels, hidden_size)
            self.label_queries = nn.Parameter(init_queries.clone())
        else:
            self.label_queries = nn.Parameter(
                torch.randn(num_labels, hidden_size) * 0.02
            )

        self.condition_proj = nn.Sequential(
            nn.Linear(num_factors, hidden_size), nn.Tanh(),
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_output, stage1_cond, attention_mask):
        B = sequence_output.size(0)
        queries = self.label_queries.unsqueeze(0).expand(B, -1, -1)
        if self.use_conditioning:
            cond = self.condition_proj(stage1_cond).unsqueeze(1)
            queries = queries + cond
        kpm = (attention_mask == 0)
        attended, _ = self.cross_attn(
            queries, sequence_output, sequence_output, key_padding_mask=kpm,
        )
        queries = self.norm1(queries + self.dropout(attended))
        queries = self.norm2(queries + self.dropout(self.ffn(queries)))
        return queries


def encode_subcategory_names(tokenizer, bert, names: List[str],
                             device: str = "cpu") -> torch.Tensor:
    """Run a list of label-name strings through BERT-CLS and return the
    [N, H] matrix used to initialise SemanticLabelAttention queries."""
    cleaned = [
        re.sub(r"\s+", " ",
               re.sub(r"[^A-Za-z0-9/\- ]", " ", n)).strip().lower()
        for n in names
    ]
    bert.eval()
    bert.to(device)
    out = []
    with torch.no_grad():
        for s in cleaned:
            enc = tokenizer(s, truncation=True, max_length=32,
                            return_tensors="pt", padding="max_length")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            hid = bert(input_ids=ids, attention_mask=mask).last_hidden_state
            out.append(hid[0, 0].detach().cpu().float())
    return torch.stack(out, dim=0)


# ─────────────────────────────────────────────────────────────────
# Flat Stage-2 head (V7's ensemble partner)
# ─────────────────────────────────────────────────────────────────

class FlatStage2Head(nn.Module):
    """Tiny MLP from a pooled doc representation to subcategory logits."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, doc_repr):
        return self.net(doc_repr)


# ─────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────

class PerClassAsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ben-Baruch et al. 2020) extended with per-class
    positive-sample weighting for rare-class recall.

        loss_pos =  log(p) * (1 - p)^gamma_pos     * w_class
        loss_neg =  log(1 - p_clip) * p_clip^gamma_neg
            where p_clip = clamp(1 - p + clip, max=1)
        loss     = -(loss_pos + loss_neg)            (mean reduction)

    Set ``pos_class_weights = None`` to fall back to standard ASL.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, eps: float = 1e-8,
                 pos_class_weights: Optional[np.ndarray] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        if pos_class_weights is not None:
            self.register_buffer(
                "pos_w",
                torch.as_tensor(pos_class_weights, dtype=torch.float32),
            )
        else:
            self.pos_w = None

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1 - targets)
            loss_pos = loss_pos * torch.pow(1 - pt_pos, self.gamma_pos)
            loss_neg = loss_neg * torch.pow(1 - pt_neg, self.gamma_neg)

        if self.pos_w is not None:
            loss_pos = loss_pos * self.pos_w.to(loss_pos.device).view(1, -1)

        loss = -(loss_pos + loss_neg)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_pos_class_weights(labels_matrix: np.ndarray, power: float = 0.5,
                              w_max: float = 8.0) -> np.ndarray:
    """Per-class positive weight = (N_majority / N_pos) ** power, clamped to
    [1, w_max].  Use w_max=1 to disable reweighting entirely."""
    M = np.asarray(labels_matrix, dtype=np.float32)
    n_pos = M.sum(axis=0)
    n_majority = float(n_pos.max()) if n_pos.max() > 0 else 1.0
    raw = (n_majority / np.maximum(n_pos, 1.0)) ** power
    return np.minimum(np.maximum(raw, 1.0), w_max).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# TwoStageModelV7
# ─────────────────────────────────────────────────────────────────

class TwoStageModelV7(nn.Module):
    """The V7 model with all ablation knobs exposed in the constructor.

    For the default V7 configuration, leave every flag at its default value.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        dropout: float = 0.3,
        subcategory_mappings: Optional[Dict[str, Dict[str, int]]] = None,
        tokenizer=None,
        # Ablation switches
        use_semantic_queries: bool = True,
        stage1_pool: str = "multi",          # "multi" | "cls" | "mean"
        use_s1_conditioning: bool = True,
        use_flat_head: bool = True,
        flat_alpha_init: float = 0.7,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_model_name, output_hidden_states=True,
        )
        self.hidden_size = self.bert.config.hidden_size
        self.layer_mix = ScalarMix(self.bert.config.num_hidden_layers + 1)

        # Stage-1 pooling
        self.stage1_pool_kind = stage1_pool
        if stage1_pool == "multi":
            self.stage1_pool = MultiPoolStage1(self.hidden_size)
        elif stage1_pool == "mean":
            self.stage1_pool = None  # mean computed inline
        elif stage1_pool == "cls":
            self.stage1_pool = None  # CLS read inline
        else:
            raise ValueError(f"Unknown stage1_pool: {stage1_pool}")

        self.stage1_dropout = nn.Dropout(dropout)
        self.stage1_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 3),
        )

        # Stage-2 sizes / names
        if subcategory_mappings is None:
            nw, ne, nh = 6, 13, 16
            w_names = [f"warning_{i}" for i in range(nw)]
            e_names = [f"env_{i}" for i in range(ne)]
            h_names = [f"human_{i}" for i in range(nh)]
        else:
            def _names(m):
                return [n for n, _ in sorted(m.items(), key=lambda x: x[1])]
            w_names = _names(subcategory_mappings["warning"])
            e_names = _names(subcategory_mappings["environmental"])
            h_names = _names(subcategory_mappings["human"])
            nw, ne, nh = len(w_names), len(e_names), len(h_names)

        # Optional semantic-query init
        w_init = e_init = h_init = None
        if use_semantic_queries and tokenizer is not None:
            w_init = encode_subcategory_names(tokenizer, self.bert, w_names)
            e_init = encode_subcategory_names(tokenizer, self.bert, e_names)
            h_init = encode_subcategory_names(tokenizer, self.bert, h_names)

        self.use_s1_conditioning = use_s1_conditioning
        self.stage2_warning_attn = SemanticLabelAttention(
            self.hidden_size, nw, num_factors=3, num_heads=4,
            dropout=dropout, init_queries=w_init,
            use_conditioning=use_s1_conditioning,
        )
        self.stage2_env_attn = SemanticLabelAttention(
            self.hidden_size, ne, num_factors=3, num_heads=4,
            dropout=dropout, init_queries=e_init,
            use_conditioning=use_s1_conditioning,
        )
        self.stage2_human_attn = SemanticLabelAttention(
            self.hidden_size, nh, num_factors=3, num_heads=4,
            dropout=dropout, init_queries=h_init,
            use_conditioning=use_s1_conditioning,
        )

        H = self.hidden_size
        self.stage2_warning_cls = nn.Sequential(
            nn.Linear(H * 2, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 1),
        )
        self.stage2_env_cls = nn.Sequential(
            nn.Linear(H * 2, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 1),
        )
        self.stage2_human_cls = nn.Sequential(
            nn.Linear(H * 2, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 1),
        )

        # Flat fusion branch
        self.use_flat_head = use_flat_head
        if use_flat_head:
            self.flat_warning_head = FlatStage2Head(H, nw, dropout=dropout)
            self.flat_env_head = FlatStage2Head(H, ne, dropout=dropout)
            self.flat_human_head = FlatStage2Head(H, nh, dropout=dropout)
            a0 = max(min(float(flat_alpha_init), 1.0 - 1e-4), 1e-4)
            alpha0 = math.log(a0 / (1.0 - a0))
            self.alpha_warning = nn.Parameter(torch.tensor(alpha0))
            self.alpha_env = nn.Parameter(torch.tensor(alpha0))
            self.alpha_human = nn.Parameter(torch.tensor(alpha0))

        self.subcategory_mappings = subcategory_mappings

    # ------------------------------------------------------------
    def enable_gradient_checkpointing(self):
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            try:
                self.bert.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except TypeError:
                self.bert.gradient_checkpointing_enable()
        if hasattr(self.bert, "config"):
            self.bert.config.use_cache = False

    def _stage1_repr(self, sequence_output, attention_mask):
        if self.stage1_pool_kind == "multi":
            return self.stage1_pool(sequence_output, attention_mask)
        if self.stage1_pool_kind == "cls":
            return sequence_output[:, 0], None
        # mean
        return mean_pool(sequence_output, attention_mask), None

    def _attn_logits(self, label_reprs, mp, head):
        mp_exp = mp.unsqueeze(1).expand(-1, label_reprs.size(1), -1)
        x = torch.cat([label_reprs, mp_exp], dim=-1)
        return head(x).squeeze(-1)

    def _mix(self, attn_logits, flat_logits, alpha_param):
        if not self.use_flat_head:
            return attn_logits
        a = torch.sigmoid(alpha_param)
        return a * attn_logits + (1.0 - a) * flat_logits

    # ------------------------------------------------------------
    def forward(self, input_ids, attention_mask, sentence_boundaries=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.layer_mix(bert_out.hidden_states)

        # Stage 1
        doc_repr, _ = self._stage1_repr(sequence_output, attention_mask)
        s1_logits = self.stage1_classifier(self.stage1_dropout(doc_repr))
        s1_cond = s1_logits / 5.0

        mp = mean_pool(sequence_output, attention_mask)

        out = {"stage1": {"logits": s1_logits,
                          "probs": torch.sigmoid(s1_logits)},
               "stage2": {}}

        for cat_key, attn_mod, cls_head in [
            ("warning",       self.stage2_warning_attn, self.stage2_warning_cls),
            ("environmental", self.stage2_env_attn,     self.stage2_env_cls),
            ("human",         self.stage2_human_attn,   self.stage2_human_cls),
        ]:
            reprs = attn_mod(sequence_output, s1_cond, attention_mask)
            attn_logits = self._attn_logits(reprs, mp, cls_head)
            flat_logits = None
            if self.use_flat_head:
                if cat_key == "warning":
                    flat_logits = self.flat_warning_head(mp)
                    alpha = self.alpha_warning
                elif cat_key == "environmental":
                    flat_logits = self.flat_env_head(mp)
                    alpha = self.alpha_env
                else:
                    flat_logits = self.flat_human_head(mp)
                    alpha = self.alpha_human
                final_logits = self._mix(attn_logits, flat_logits, alpha)
                alpha_val = torch.sigmoid(alpha)
            else:
                final_logits = attn_logits
                alpha_val = None

            out["stage2"][cat_key] = {
                "subcategory_logits": final_logits,
                "subcategory_probs": torch.sigmoid(final_logits),
                "attn_logits": attn_logits,
                "flat_logits": flat_logits,
                "alpha": alpha_val,
            }
        return out


# ─────────────────────────────────────────────────────────────────
# Layer-wise LR decay parameter groups
# ─────────────────────────────────────────────────────────────────

def get_layer_lrd_param_groups(
    model, bert_lr: float, head_lr: float, layer_decay: float = 0.85,
    weight_decay: float = 0.01,
    no_decay_keywords=("bias", "LayerNorm.weight", "LayerNorm.bias"),
):
    """Build AdamW parameter groups with layer-wise LR decay on the BERT
    backbone.  Non-BERT parameters use head_lr."""
    bert = model.bert
    L = bert.config.num_hidden_layers

    def _split(named):
        d, nd = [], []
        for n, p in named:
            if not p.requires_grad:
                continue
            if any(k in n for k in no_decay_keywords):
                nd.append(p)
            else:
                d.append(p)
        return d, nd

    groups = []

    if hasattr(bert, "embeddings"):
        emb_lr = bert_lr * (layer_decay ** (L + 1))
        d, nd = _split(bert.embeddings.named_parameters())
        if d:  groups.append({"params": d,  "lr": emb_lr,
                              "weight_decay": weight_decay})
        if nd: groups.append({"params": nd, "lr": emb_lr,
                              "weight_decay": 0.0})

    if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
        for i, layer in enumerate(bert.encoder.layer):
            lr_i = bert_lr * (layer_decay ** (L - i))
            d, nd = _split(layer.named_parameters())
            if d:  groups.append({"params": d,  "lr": lr_i,
                                  "weight_decay": weight_decay})
            if nd: groups.append({"params": nd, "lr": lr_i,
                                  "weight_decay": 0.0})

    if hasattr(bert, "pooler") and bert.pooler is not None:
        d, nd = _split(bert.pooler.named_parameters())
        if d:  groups.append({"params": d,  "lr": bert_lr,
                              "weight_decay": weight_decay})
        if nd: groups.append({"params": nd, "lr": bert_lr,
                              "weight_decay": 0.0})

    bert_ids = set(id(p) for p in bert.parameters())
    head_named = [(n, p) for n, p in model.named_parameters()
                  if id(p) not in bert_ids]
    d, nd = _split(head_named)
    if d:  groups.append({"params": d,  "lr": head_lr,
                          "weight_decay": weight_decay})
    if nd: groups.append({"params": nd, "lr": head_lr,
                          "weight_decay": 0.0})

    return [g for g in groups if g["params"]]
