"""
Honest baselines for the railway-crossing accident classifier.

All five baselines are trained and evaluated on the *exact same data
splits* (same seed, same iterative-stratification) as TwoStageModelV7,
and all use the same default scalar thresholds (S1=0.5, S2=0.5) at test
time.  No baseline is intentionally hobbled; if a baseline beats V7 on
some metric, that is reported faithfully.

    B1  TF-IDF + Logistic Regression  (flat multi-label, one-vs-rest)
    B2  TF-IDF + LinearSVC + sigmoid-calibrated decision values
    B3  BERT-CLS, single-stage flat multi-label head
    B4  BERT-MeanPool, single-stage flat multi-label head
    B5  Two-stage BERT (mean-pool everywhere, no label attention)

B3/B4/B5 use the exact same training budget, optimiser, LLRD, batch size,
data, and loss schedule as V7 (substituting BCE-with-logits + per-class
positive weights where needed) so the comparison is apples-to-apples.

Run:
    python baselines.py --data <csv>           # all 5 baselines
    python baselines.py --only B3              # one baseline
    python baselines.py --only B1,B2           # quick (no GPU)
    python baselines.py --epochs 60 --patience 20
"""

from __future__ import annotations

import argparse
import gc
import glob as _glob
import json
import math
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from data import (
    AccidentDataset,
    build_datasets_from_splits,
    count_main_factors,
    create_balanced_sampler,
    load_processed_dataframe,
    parse_hierarchical_labels,
    save_splits_to_csv,
    split_dataset_by_subcategory_ratio,
)


# ═════════════════════════════════════════════════════════════════
# USER CONFIG — edit these before running
# ═════════════════════════════════════════════════════════════════

DATA_FILE  = r"HRGC_P_aug.csv"           # e.g. r"..\augmented_v4_20260418_113621.csv"
SAVE_DIR   = r"baselines_results_v7"
SEED       = 42
EPOCHS     = 500          # None = V7 default
PATIENCE   = 50
BATCH_SIZE = 32
BERT_MODEL = "bert-base-uncased"
ONLY       = None          # None = all baselines. Or e.g. "B1,B3,B9".
from model import (
    AttentionPooling,
    PerClassAsymmetricLoss,
    ScalarMix,
    compute_pos_class_weights,
    get_layer_lrd_param_groups,
)
from train import (
    CONFIG as V7_CONFIG,
    _binary_f1,
    _multilabel_f1,
    _multilabel_pr,
    _train_label_matrices,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────

def _pack_subcat_y(labels: List[Dict], mappings) -> Dict[str, np.ndarray]:
    """Convert hierarchical labels to flat per-category binary matrices."""
    nw = len(mappings["warning"])
    ne = len(mappings["environmental"])
    nh = len(mappings["human"])
    n = len(labels)
    out = {
        "warning":       np.zeros((n, nw), dtype=np.int64),
        "environmental": np.zeros((n, ne), dtype=np.int64),
        "human":         np.zeros((n, nh), dtype=np.int64),
    }
    for i, L in enumerate(labels):
        for k in L["subcategories"]["warning"]:       out["warning"][i, k] = 1
        for k in L["subcategories"]["environmental"]: out["environmental"][i, k] = 1
        for k in L["subcategories"]["human"]:         out["human"][i, k] = 1
    return out


def _summarise_test(y_dict, p_dict):
    summary = {}
    for cat in ("warning", "environmental", "human"):
        y, p = y_dict[cat], p_dict[cat]
        prec, rec = _multilabel_pr(y, p)
        summary[f"{cat}_s2"] = {
            "micro_f1": _multilabel_f1(y, p, "micro"),
            "macro_f1": _multilabel_f1(y, p, "macro"),
            "precision_micro": prec, "recall_micro": rec,
        }
    all_y = np.hstack([y_dict["warning"], y_dict["environmental"],
                       y_dict["human"]])
    all_p = np.hstack([p_dict["warning"], p_dict["environmental"],
                       p_dict["human"]])
    prec, rec = _multilabel_pr(all_y, all_p)
    summary["overall"] = {
        "micro_f1": _multilabel_f1(all_y, all_p, "micro"),
        "macro_f1": _multilabel_f1(all_y, all_p, "macro"),
        "precision_micro": prec, "recall_micro": rec,
    }
    return summary


# ─────────────────────────────────────────────────────────────────
# B1 — TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────────────────

def run_B1_tfidf_logreg(splits, mappings, **_):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_y = _pack_subcat_y(splits["train"]["labels"], mappings)
    test_y  = _pack_subcat_y(splits["test"]["labels"],  mappings)

    vec = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.95,
    )
    X_train = vec.fit_transform(splits["train"]["texts"])
    X_test  = vec.transform(splits["test"]["texts"])

    test_p = {}
    for cat in ("warning", "environmental", "human"):
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               solver="liblinear"),
            n_jobs=-1,
        )
        clf.fit(X_train, train_y[cat])
        test_p[cat] = clf.predict(X_test).astype(np.int64)
    return _summarise_test(test_y, test_p)


# ─────────────────────────────────────────────────────────────────
# B2 — TF-IDF + LinearSVC
# ─────────────────────────────────────────────────────────────────

def run_B2_tfidf_svm(splits, mappings, **_):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier

    train_y = _pack_subcat_y(splits["train"]["labels"], mappings)
    test_y  = _pack_subcat_y(splits["test"]["labels"],  mappings)

    vec = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.95,
    )
    X_train = vec.fit_transform(splits["train"]["texts"])
    X_test  = vec.transform(splits["test"]["texts"])

    test_p = {}
    for cat in ("warning", "environmental", "human"):
        clf = OneVsRestClassifier(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000), n_jobs=-1,
        )
        clf.fit(X_train, train_y[cat])
        test_p[cat] = clf.predict(X_test).astype(np.int64)
    return _summarise_test(test_y, test_p)


# ─────────────────────────────────────────────────────────────────
# B3 / B4 — BERT pool + flat multi-label head
# ─────────────────────────────────────────────────────────────────

class FlatBertClassifier(nn.Module):
    """BERT encoder + ScalarMix + a configurable pool + a single linear
    head producing K = nw + ne + nh logits.  Used for both B3 (CLS pool)
    and B4 (mean pool)."""

    def __init__(self, bert_model_name: str, pool: str,
                 num_warning: int, num_env: int, num_human: int,
                 dropout: float = 0.3):
        super().__init__()
        assert pool in ("cls", "mean")
        self.pool = pool
        self.bert = AutoModel.from_pretrained(
            bert_model_name, output_hidden_states=True,
        )
        H = self.bert.config.hidden_size
        self.layer_mix = ScalarMix(self.bert.config.num_hidden_layers + 1)
        self.dropout = nn.Dropout(dropout)
        K = num_warning + num_env + num_human
        self.head = nn.Sequential(
            nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, K),
        )
        self._splits = (num_warning, num_env, num_human)

    def enable_gradient_checkpointing(self):
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            try:
                self.bert.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.bert.gradient_checkpointing_enable()
        if hasattr(self.bert, "config"):
            self.bert.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        seq = self.layer_mix(out.hidden_states)
        if self.pool == "cls":
            doc = seq[:, 0]
        else:
            m = attention_mask.float().unsqueeze(-1)
            doc = (seq * m).sum(1) / m.sum(1).clamp(min=1e-6)
        logits = self.head(self.dropout(doc))
        nw, ne, nh = self._splits
        return {
            "warning": logits[:, :nw],
            "environmental": logits[:, nw:nw + ne],
            "human": logits[:, nw + ne:],
        }


def _simple_param_groups(model, bert_lr, head_lr, weight_decay,
                         no_decay=("bias", "LayerNorm.weight",
                                   "LayerNorm.bias")):
    """Two-group split (backbone vs head) — no LLRD.  Used for the
    cross-backbone baselines (B6 DistilBERT, B7 RoBERTa, B8 DeBERTa-v3)
    so that layer-naming differences don't give some baselines an unfair
    LLRD advantage over others."""
    bert_ids = set(id(p) for p in model.bert.parameters())
    bd, bnd, hd, hnd = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_nd = any(k in n for k in no_decay)
        if id(p) in bert_ids:
            (bnd if is_nd else bd).append(p)
        else:
            (hnd if is_nd else hd).append(p)
    groups = []
    if bd:  groups.append({"params": bd,  "lr": bert_lr,
                           "weight_decay": weight_decay})
    if bnd: groups.append({"params": bnd, "lr": bert_lr,
                           "weight_decay": 0.0})
    if hd:  groups.append({"params": hd,  "lr": head_lr,
                           "weight_decay": weight_decay})
    if hnd: groups.append({"params": hnd, "lr": head_lr,
                           "weight_decay": 0.0})
    return groups


def _find_latest_checkpoint(ckpt_dir, prefix):
    ckpts = sorted(
        _glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch*.pth")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    return ckpts[-1] if ckpts else None


def _train_flat_bert(model, train_loader, val_loader, criteria_dict,
                     cfg, device, save_dir, ckpt_name, param_groups=None,
                     resume_checkpoint=None):
    if param_groups is None:
        param_groups = get_layer_lrd_param_groups(
            model, cfg["bert_lr"], cfg["head_lr"],
            cfg["layer_decay"], cfg["weight_decay"],
        )
    pgs = param_groups
    opt = torch.optim.AdamW(pgs)
    accum = cfg["gradient_accumulation_steps"]
    total_steps = max(len(train_loader) // accum, 1) * cfg["epochs"]
    warmup = int(total_steps * cfg["warmup_ratio"])
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    if cfg["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model.to(device)

    all_params = [p for g in pgs for p in g["params"]]
    best_val_f1 = -1.0; best_ep = 0; patience = 0
    best_path = os.path.join(save_dir, ckpt_name)
    ckpt_prefix = os.path.splitext(ckpt_name)[0].replace("best_", "checkpoint_")
    start_epoch = 0

    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"\n[Resume] Loading checkpoint: {resume_checkpoint}", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        start_epoch = ckpt["epoch"]
        best_ep     = start_epoch
        print(f"[Resume] epoch={start_epoch}  best_val_f1={best_val_f1:.4f}",
              flush=True)

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        opt.zero_grad()
        nb = len(train_loader)
        for bi, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            w_y = batch["warning_subcat_labels"].float().to(device)
            e_y = batch["env_subcat_labels"].float().to(device)
            h_y = batch["human_subcat_labels"].float().to(device)
            out = model(ids, mask)
            loss = (
                criteria_dict["warning"](out["warning"], w_y)
                + criteria_dict["env"](out["environmental"], e_y)
                + criteria_dict["human"](out["human"], h_y)
            ) / 3.0
            (loss / accum).backward()
            if ((bi + 1) % accum == 0) or ((bi + 1) == nb):
                torch.nn.utils.clip_grad_norm_(all_params, cfg["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad()

        # Validation (default scalar threshold 0.5)
        model.eval()
        ys = {"warning": [], "environmental": [], "human": []}
        ps = {"warning": [], "environmental": [], "human": []}
        with torch.no_grad():
            for vb in val_loader:
                ids = vb["input_ids"].to(device)
                mask = vb["attention_mask"].to(device)
                out = model(ids, mask)
                for cat, lab_key in [("warning", "warning_subcat_labels"),
                                     ("environmental", "env_subcat_labels"),
                                     ("human", "human_subcat_labels")]:
                    p = (torch.sigmoid(out[cat]) > 0.5).cpu().numpy().astype(int)
                    ps[cat].append(p)
                    ys[cat].append(vb[lab_key].numpy().astype(int))
        all_y = np.hstack([np.vstack(ys[c]) for c in ys])
        all_p = np.hstack([np.vstack(ps[c]) for c in ps])
        val_f1 = _multilabel_f1(all_y, all_p, "micro")
        print(f"  Ep {epoch+1}/{cfg['epochs']}  val overall μF1={val_f1:.4f}",
              flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; best_ep = epoch + 1; patience = 0
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       best_path)
            print(f"    ** new best @ ep {best_ep}", flush=True)
        else:
            patience += 1

        if (epoch + 1) % 10 == 0:
            p = os.path.join(save_dir,
                             f"{ckpt_prefix}_epoch{epoch+1}.pth")
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     {k: v.cpu().clone()
                                         for k, v in model.state_dict().items()},
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "best_val_f1":          best_val_f1,
            }, p)
            print(f"    [Checkpoint] Saved -> {p}", flush=True)

        if patience >= cfg["patience"] and (epoch + 1) >= cfg["min_epochs"]:
            print(f"    early stop at ep {epoch+1}")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_val_f1, best_ep


def _flat_bert_evaluate(model, test_loader, device):
    model.eval()
    ys = {"warning": [], "environmental": [], "human": []}
    ps = {"warning": [], "environmental": [], "human": []}
    with torch.no_grad():
        for vb in test_loader:
            ids = vb["input_ids"].to(device)
            mask = vb["attention_mask"].to(device)
            out = model(ids, mask)
            for cat, lab_key in [("warning", "warning_subcat_labels"),
                                 ("environmental", "env_subcat_labels"),
                                 ("human", "human_subcat_labels")]:
                p = (torch.sigmoid(out[cat]) > 0.5).cpu().numpy().astype(int)
                ps[cat].append(p)
                ys[cat].append(vb[lab_key].numpy().astype(int))
    y_dict = {c: np.vstack(ys[c]) for c in ys}
    p_dict = {c: np.vstack(ps[c]) for c in ps}
    return _summarise_test(y_dict, p_dict)


def _build_flat_criteria(cfg, w_mat, e_mat, h_mat):
    base = dict(gamma_neg=cfg["loss_gamma_neg"],
                gamma_pos=cfg["loss_gamma_pos"],
                clip=cfg["loss_clip"])
    p = cfg["per_class_pos_weight_power"]; wm = cfg["per_class_pos_weight_max"]
    return {
        "warning": PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(w_mat, p, wm)),
        "env":     PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(e_mat, p, wm)),
        "human":   PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(h_mat, p, wm)),
    }


def run_B3_bert_cls_flat(splits, mappings, *, tokenizer, cfg, device, save_dir,
                         labels_full):
    nw = len(mappings["warning"]); ne = len(mappings["environmental"])
    nh = len(mappings["human"])
    main_mat, w_mat, e_mat, h_mat = _train_label_matrices(
        labels_full, splits["train"]["indices"], mappings)
    criteria = _build_flat_criteria(cfg, w_mat, e_mat, h_mat)
    datasets = build_datasets_from_splits(splits, tokenizer, mappings)
    sampler = create_balanced_sampler(splits["train"]["labels"],
                                      count_main_factors(labels_full))
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)

    model = FlatBertClassifier(cfg["bert_model"], pool="cls",
                               num_warning=nw, num_env=ne, num_human=nh,
                               dropout=cfg["dropout"])
    model, best_val, best_ep = _train_flat_bert(
        model, train_loader, val_loader, criteria, cfg, device, save_dir,
        ckpt_name="best_B3.pth",
        resume_checkpoint=_find_latest_checkpoint(save_dir, "checkpoint_B3"),
    )
    test_m = _flat_bert_evaluate(model, test_loader, device)
    return test_m, best_val, best_ep


def run_B4_bert_mean_flat(splits, mappings, *, tokenizer, cfg, device, save_dir,
                          labels_full):
    nw = len(mappings["warning"]); ne = len(mappings["environmental"])
    nh = len(mappings["human"])
    main_mat, w_mat, e_mat, h_mat = _train_label_matrices(
        labels_full, splits["train"]["indices"], mappings)
    criteria = _build_flat_criteria(cfg, w_mat, e_mat, h_mat)
    datasets = build_datasets_from_splits(splits, tokenizer, mappings)
    sampler = create_balanced_sampler(splits["train"]["labels"],
                                      count_main_factors(labels_full))
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)

    model = FlatBertClassifier(cfg["bert_model"], pool="mean",
                               num_warning=nw, num_env=ne, num_human=nh,
                               dropout=cfg["dropout"])
    model, best_val, best_ep = _train_flat_bert(
        model, train_loader, val_loader, criteria, cfg, device, save_dir,
        ckpt_name="best_B4.pth",
        resume_checkpoint=_find_latest_checkpoint(save_dir, "checkpoint_B4"),
    )
    test_m = _flat_bert_evaluate(model, test_loader, device)
    return test_m, best_val, best_ep


# ─────────────────────────────────────────────────────────────────
# B5 — Two-stage BERT (mean-pool everywhere, no label attention)
# ─────────────────────────────────────────────────────────────────

class TwoStageMeanModel(nn.Module):
    """Two-stage classifier where Stage-2 uses the same mean-pooled
    document representation (no per-label attention).  Stage-1 logits
    are concatenated to the doc representation as a (3-dim) condition."""

    def __init__(self, bert_model_name, num_warning, num_env, num_human,
                 dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_model_name, output_hidden_states=True,
        )
        H = self.bert.config.hidden_size
        self.layer_mix = ScalarMix(self.bert.config.num_hidden_layers + 1)
        self.dropout = nn.Dropout(dropout)
        self.stage1 = nn.Sequential(
            nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 3),
        )
        self.warn_head = nn.Sequential(
            nn.Linear(H + 3, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, num_warning),
        )
        self.env_head = nn.Sequential(
            nn.Linear(H + 3, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, num_env),
        )
        self.hum_head = nn.Sequential(
            nn.Linear(H + 3, H // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H // 2, num_human),
        )

    def enable_gradient_checkpointing(self):
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            try:
                self.bert.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.bert.gradient_checkpointing_enable()
        if hasattr(self.bert, "config"):
            self.bert.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        seq = self.layer_mix(out.hidden_states)
        m = attention_mask.float().unsqueeze(-1)
        doc = (seq * m).sum(1) / m.sum(1).clamp(min=1e-6)
        s1 = self.stage1(self.dropout(doc))
        cond = torch.cat([doc, s1 / 5.0], dim=-1)
        return {
            "stage1": {"logits": s1, "probs": torch.sigmoid(s1)},
            "warning":       self.warn_head(cond),
            "environmental": self.env_head(cond),
            "human":         self.hum_head(cond),
        }


def run_B5_twostage_no_attn(splits, mappings, *, tokenizer, cfg, device,
                            save_dir, labels_full):
    nw = len(mappings["warning"]); ne = len(mappings["environmental"])
    nh = len(mappings["human"])
    main_mat, w_mat, e_mat, h_mat = _train_label_matrices(
        labels_full, splits["train"]["indices"], mappings)
    base = dict(gamma_neg=cfg["loss_gamma_neg"],
                gamma_pos=cfg["loss_gamma_pos"], clip=cfg["loss_clip"])
    pp = cfg["per_class_pos_weight_power"]; pm = cfg["per_class_pos_weight_max"]
    criteria = {
        "stage1": PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(main_mat, pp, pm)),
        "warning": PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(w_mat, pp, pm)),
        "env": PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(e_mat, pp, pm)),
        "human": PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(h_mat, pp, pm)),
    }

    datasets = build_datasets_from_splits(splits, tokenizer, mappings)
    sampler = create_balanced_sampler(splits["train"]["labels"],
                                      count_main_factors(labels_full))
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)

    model = TwoStageMeanModel(cfg["bert_model"], nw, ne, nh, cfg["dropout"])
    pgs = get_layer_lrd_param_groups(
        model, cfg["bert_lr"], cfg["head_lr"],
        cfg["layer_decay"], cfg["weight_decay"],
    )
    opt = torch.optim.AdamW(pgs)
    accum = cfg["gradient_accumulation_steps"]
    total_steps = max(len(train_loader) // accum, 1) * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * cfg["warmup_ratio"]), total_steps)
    if cfg["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model.to(device)
    all_params = [p for g in pgs for p in g["params"]]

    best_val = -1.0; best_ep = 0; patience = 0
    best_path = os.path.join(save_dir, "best_B5.pth")
    aux_w = 0.3   # B5 uses the V6-style fixed aux weight by design
    start_epoch = 0

    b5_resume = _find_latest_checkpoint(save_dir, "checkpoint_B5")
    if b5_resume:
        print(f"\n[Resume] Loading checkpoint: {b5_resume}", flush=True)
        ckpt = torch.load(b5_resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        best_val    = ckpt.get("best_val_f1", -1.0)
        start_epoch = ckpt["epoch"]
        best_ep     = start_epoch
        print(f"[Resume] epoch={start_epoch}  best_val_f1={best_val:.4f}",
              flush=True)

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        opt.zero_grad()
        nb = len(train_loader)
        for bi, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            main_y = batch["main_labels"].float().to(device)
            w_y = batch["warning_subcat_labels"].float().to(device)
            e_y = batch["env_subcat_labels"].float().to(device)
            h_y = batch["human_subcat_labels"].float().to(device)
            out = model(ids, mask)
            s2 = (criteria["warning"](out["warning"], w_y)
                  + criteria["env"](out["environmental"], e_y)
                  + criteria["human"](out["human"], h_y)) / 3.0
            s1 = criteria["stage1"](out["stage1"]["logits"], main_y)
            loss = s2 + aux_w * s1
            (loss / accum).backward()
            if ((bi + 1) % accum == 0) or ((bi + 1) == nb):
                torch.nn.utils.clip_grad_norm_(all_params, cfg["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad()

        # Val (default scalar 0.5 thresholds)
        model.eval()
        ys = {"warning": [], "environmental": [], "human": []}
        ps = {"warning": [], "environmental": [], "human": []}
        with torch.no_grad():
            for vb in val_loader:
                ids = vb["input_ids"].to(device)
                mask = vb["attention_mask"].to(device)
                out = model(ids, mask)
                for cat, lab in [("warning", "warning_subcat_labels"),
                                 ("environmental", "env_subcat_labels"),
                                 ("human", "human_subcat_labels")]:
                    p = (torch.sigmoid(out[cat]) > 0.5).cpu().numpy().astype(int)
                    ps[cat].append(p)
                    ys[cat].append(vb[lab].numpy().astype(int))
        all_y = np.hstack([np.vstack(ys[c]) for c in ys])
        all_p = np.hstack([np.vstack(ps[c]) for c in ps])
        val_f1 = _multilabel_f1(all_y, all_p, "micro")
        print(f"  Ep {epoch+1}/{cfg['epochs']}  val overall μF1={val_f1:.4f}",
              flush=True)

        if val_f1 > best_val:
            best_val = val_f1; best_ep = epoch + 1; patience = 0
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       best_path)
            print(f"    ** new best @ ep {best_ep}", flush=True)
        else:
            patience += 1

        if (epoch + 1) % 10 == 0:
            p = os.path.join(save_dir, f"checkpoint_B5_epoch{epoch+1}.pth")
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     {k: v.cpu().clone()
                                         for k, v in model.state_dict().items()},
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "best_val_f1":          best_val,
            }, p)
            print(f"    [Checkpoint] Saved -> {p}", flush=True)

        if patience >= cfg["patience"] and (epoch + 1) >= cfg["min_epochs"]:
            print(f"    early stop at ep {epoch+1}")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = _flat_bert_evaluate_model(model, test_loader, device)
    return test_m, best_val, best_ep


def _flat_bert_evaluate_model(model, test_loader, device):
    """Same as _flat_bert_evaluate but for B5's TwoStageMeanModel,
    which returns Stage-2 logits at out['warning'], out['environmental'],
    out['human']."""
    model.eval()
    ys = {"warning": [], "environmental": [], "human": []}
    ps = {"warning": [], "environmental": [], "human": []}
    with torch.no_grad():
        for vb in test_loader:
            ids = vb["input_ids"].to(device)
            mask = vb["attention_mask"].to(device)
            out = model(ids, mask)
            for cat, lab in [("warning", "warning_subcat_labels"),
                             ("environmental", "env_subcat_labels"),
                             ("human", "human_subcat_labels")]:
                p = (torch.sigmoid(out[cat]) > 0.5).cpu().numpy().astype(int)
                ps[cat].append(p)
                ys[cat].append(vb[lab].numpy().astype(int))
    y_dict = {c: np.vstack(ys[c]) for c in ys}
    p_dict = {c: np.vstack(ps[c]) for c in ps}
    return _summarise_test(y_dict, p_dict)


# ─────────────────────────────────────────────────────────────────
# B6 / B7 / B8 — Different encoder backbones (CLS-pool flat heads)
# ─────────────────────────────────────────────────────────────────

def _run_flat_with_backbone(backbone_name, ckpt_name, splits, mappings,
                            cfg, device, save_dir, labels_full):
    """Train + evaluate a flat-head BERT classifier under a swap-in
    encoder backbone (DistilBERT / RoBERTa / DeBERTa-v3 / ...).

    Uses *simple* (no-LLRD) parameter groups so that backbones with
    different layer-naming conventions all train under identical
    optimiser strategy — the only thing that differs is the encoder
    itself, which is the point of this comparison.
    """
    nw = len(mappings["warning"]); ne = len(mappings["environmental"])
    nh = len(mappings["human"])
    main_mat, w_mat, e_mat, h_mat = _train_label_matrices(
        labels_full, splits["train"]["indices"], mappings)
    criteria = _build_flat_criteria(cfg, w_mat, e_mat, h_mat)

    # Each backbone uses its own tokenizer (BPE / WordPiece / SentencePiece)
    tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    datasets = build_datasets_from_splits(splits, tok, mappings)
    sampler = create_balanced_sampler(splits["train"]["labels"],
                                      count_main_factors(labels_full))
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)

    model = FlatBertClassifier(backbone_name, pool="cls",
                               num_warning=nw, num_env=ne, num_human=nh,
                               dropout=cfg["dropout"])
    pgs = _simple_param_groups(model, cfg["bert_lr"], cfg["head_lr"],
                               cfg["weight_decay"])
    stem = os.path.splitext(ckpt_name)[0].replace("best_", "checkpoint_")
    # Gradient checkpointing on non-BERT backbones (DistilBERT, RoBERTa,
    # DeBERTa-v3) can trigger native CUDA access violations on Windows.
    # Disable it here; these smaller models fit in GPU memory without it.
    backbone_cfg = dict(cfg)
    backbone_cfg["gradient_checkpointing"] = False
    model, best_val, best_ep = _train_flat_bert(
        model, train_loader, val_loader, criteria, backbone_cfg, device, save_dir,
        ckpt_name=ckpt_name, param_groups=pgs,
        resume_checkpoint=_find_latest_checkpoint(save_dir, stem),
    )
    test_m = _flat_bert_evaluate(model, test_loader, device)
    return test_m, best_val, best_ep


def run_B6_distilbert_flat(splits, mappings, *, tokenizer, cfg, device,
                           save_dir, labels_full):
    return _run_flat_with_backbone(
        "distilbert-base-uncased", "best_B6.pth",
        splits, mappings, cfg, device, save_dir, labels_full,
    )


def run_B7_roberta_flat(splits, mappings, *, tokenizer, cfg, device,
                        save_dir, labels_full):
    return _run_flat_with_backbone(
        "roberta-base", "best_B7.pth",
        splits, mappings, cfg, device, save_dir, labels_full,
    )


def run_B8_deberta_v3_flat(splits, mappings, *, tokenizer, cfg, device,
                           save_dir, labels_full):
    return _run_flat_with_backbone(
        "microsoft/deberta-v3-base", "best_B8.pth",
        splits, mappings, cfg, device, save_dir, labels_full,
    )


# ─────────────────────────────────────────────────────────────────
# B9 — Sentence-BERT label-name cosine similarity (zero-shot)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _sbert_encode(texts, tokenizer, model, device, batch_size=32,
                  max_length=256):
    """Mean-pool sentence-bert embeddings, L2-normalised."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) if t is not None else "" for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state                # [B, S, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = F.normalize(pooled, dim=-1)
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs)


def _best_global_threshold(sims, y, candidates):
    """Sweep one global per-label threshold over *one* category, pick the
    threshold that maximises micro-F1 over the [N, K] sims matrix."""
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (sims > t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        if tp == 0:
            continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def run_B9_sbert_zero_shot(splits, mappings, *, tokenizer, cfg, device,
                           save_dir, labels_full):
    """Zero-shot: encode every sample and every subcategory NAME with
    Sentence-BERT (all-mpnet-base-v2), then predict subcategory j as
    positive iff cosine(sample, name_j) > t.  No backbone fine-tuning.

    The only "training" is choosing one threshold per category on the
    validation split (a single scalar per category) to put SBERT scores
    on a comparable scale — this matches the standard zero-shot multi-
    label evaluation protocol.

    Tests V7's premise that *learned* label-aware attention beats just
    looking at the label name.
    """
    sbert_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"  Loading SBERT: {sbert_name}")
    tok = AutoTokenizer.from_pretrained(sbert_name, use_fast=True)
    sbert = AutoModel.from_pretrained(sbert_name).to(device).eval()

    # 1) encode label names per category
    label_emb = {}
    for cat in ("warning", "environmental", "human"):
        names = [n for n, _ in sorted(mappings[cat].items(),
                                      key=lambda x: x[1])]
        label_emb[cat] = _sbert_encode(names, tok, sbert, device,
                                       batch_size=64, max_length=64)
        print(f"  Encoded {len(names):>3} {cat} label names "
              f"-> {label_emb[cat].shape}")

    # 2) encode val + test docs
    val_texts  = splits["val"]["texts"]
    test_texts = splits["test"]["texts"]
    val_emb  = _sbert_encode(val_texts,  tok, sbert, device,
                             batch_size=cfg["batch_size"], max_length=256)
    test_emb = _sbert_encode(test_texts, tok, sbert, device,
                             batch_size=cfg["batch_size"], max_length=256)

    # 3) labels
    val_y  = _pack_subcat_y(splits["val"]["labels"],  mappings)
    test_y = _pack_subcat_y(splits["test"]["labels"], mappings)

    # 4) per-category global threshold tuned on val
    candidates = np.linspace(0.05, 0.95, 19)
    test_p = {}
    chosen_thresholds = {}
    for cat in ("warning", "environmental", "human"):
        sims_val  = val_emb  @ label_emb[cat].T
        sims_test = test_emb @ label_emb[cat].T
        t, f1_val = _best_global_threshold(sims_val, val_y[cat], candidates)
        chosen_thresholds[cat] = float(t)
        print(f"  [{cat}] best val threshold = {t:.2f} "
              f"(val micro-F1 = {f1_val:.4f})")
        test_p[cat] = (sims_test > t).astype(np.int64)

    summary = _summarise_test(test_y, test_p)
    summary["chosen_thresholds_on_val"] = chosen_thresholds
    return summary, None, 0


# ─────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────

BASELINES = {
    "B1": ("[B1] TF-IDF + LogisticRegression",         run_B1_tfidf_logreg, False),
    "B2": ("[B2] TF-IDF + LinearSVC",                  run_B2_tfidf_svm,    False),
    "B3": ("[B3] BERT-CLS, single-stage flat",         run_B3_bert_cls_flat, True),
    "B4": ("[B4] BERT-MeanPool, single-stage flat",    run_B4_bert_mean_flat, True),
    "B5": ("[B5] Two-stage BERT (mean-pool, no label attention)",
           run_B5_twostage_no_attn, True),
    "B6": ("[B6] DistilBERT-CLS, single-stage flat",   run_B6_distilbert_flat, True),
    "B7": ("[B7] RoBERTa-base CLS, single-stage flat", run_B7_roberta_flat, True),
    "B8": ("[B8] DeBERTa-v3-base CLS, single-stage flat",
           run_B8_deberta_v3_flat, True),
    "B9": ("[B9] Sentence-BERT (mpnet-base-v2) zero-shot label-name cosine",
           run_B9_sbert_zero_shot, True),
}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default=None)
    p.add_argument("--save-dir",   type=str, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--patience",   type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--bert-model", type=str, default=None)
    p.add_argument("--only",       type=str, default=None,
                   help="Comma-separated baselines, e.g. B1,B3,B9.")
    p.add_argument("--resume-dir", type=str, default=None,
                   help="Explicit path to a previous run_*_seed<N>/ dir. "
                        "If omitted, the latest existing run dir is reused "
                        "automatically.")
    p.add_argument("--force-new-run", action="store_true",
                   help="Always create a fresh run_* directory instead of "
                        "reusing the latest one.")
    args = p.parse_args()

    cfg = dict(V7_CONFIG)
    cfg["data_file"]  = args.data       or DATA_FILE
    cfg["seed"]       = args.seed       if args.seed       is not None else SEED
    cfg["bert_model"] = args.bert_model or BERT_MODEL
    epochs_eff        = args.epochs     if args.epochs     is not None else EPOCHS
    patience_eff      = args.patience   if args.patience   is not None else PATIENCE
    batch_eff         = args.batch_size if args.batch_size is not None else BATCH_SIZE
    only_eff          = args.only       if args.only       is not None else ONLY
    save_dir_eff      = args.save_dir   or SAVE_DIR

    if epochs_eff   is not None: cfg["epochs"]     = epochs_eff
    if patience_eff is not None: cfg["patience"]   = patience_eff
    if batch_eff    is not None: cfg["batch_size"] = batch_eff

    if not cfg["data_file"]:
        raise ValueError(
            "Set DATA_FILE at the top of baselines.py, "
            "or pass --data on the command line."
        )

    cfg["strict_determinism"] = False
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {cfg['seed']}")

    # Resolve to absolute path so OneDrive / CWD issues don't bite us
    abs_save_dir = os.path.abspath(save_dir_eff)

    if args.resume_dir and os.path.isdir(args.resume_dir):
        root_dir = os.path.abspath(args.resume_dir)
        _dir_name = os.path.basename(root_dir.rstrip("/\\"))
        _m = re.search(r"_seed(\d+)$", _dir_name)
        if _m and int(_m.group(1)) != cfg["seed"]:
            raise SystemExit(
                f"[Error] --resume-dir seed mismatch: "
                f"directory has seed={_m.group(1)} but current seed={cfg['seed']}. "
                f"Pass --seed {_m.group(1)} to resume correctly, or use "
                f"--force-new-run to start fresh with seed={cfg['seed']}."
            )
    else:
        suffix = f"_seed{cfg['seed']}"
        if os.path.isdir(abs_save_dir):
            candidates = sorted(
                d for d in os.listdir(abs_save_dir)
                if d.startswith("run_") and d.endswith(suffix)
                and os.path.isdir(os.path.join(abs_save_dir, d))
            )
        else:
            candidates = []
        print(f"[Info] Save dir: {abs_save_dir}")
        print(f"[Info] Existing run dirs (seed={cfg['seed']}): "
              f"{candidates if candidates else 'none'}")
        if candidates and not args.force_new_run:
            root_dir = os.path.join(abs_save_dir, candidates[-1])
            print(f"[Resume] Auto-reusing latest run dir: {root_dir}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            root_dir = os.path.join(abs_save_dir,
                                    f"run_{ts}_seed{cfg['seed']}")
    os.makedirs(root_dir, exist_ok=True)
    print(f"Results dir: {root_dir}")

    with open(os.path.join(root_dir, "run_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"Data file:    {cfg['data_file']}")
    df = load_processed_dataframe(cfg["data_file"])
    texts = df["DETAILED_DESCRIPTION"].tolist()
    labels, mappings = parse_hierarchical_labels(df)
    splits = split_dataset_by_subcategory_ratio(
        texts, labels, mappings, ratios=(0.7, 0.15, 0.15),
        random_state=cfg["seed"],
    )
    save_splits_to_csv(splits, df, os.path.join(root_dir, "data_splits"))
    with open(os.path.join(root_dir, "subcategory_mappings.json"),
              "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_model"], use_fast=True)

    selected = (
        [s.strip() for s in only_eff.split(",") if s.strip()]
        if only_eff else list(BASELINES.keys())
    )
    for n in selected:
        if n not in BASELINES:
            raise ValueError(
                f"Unknown baseline {n!r}. Choices: {list(BASELINES)}")

    def _find_completed_baseline(save_root, baseline_name, seed):
        """Look across all sibling run_*_seed{seed}/baseline_<name>/
        directories for a finished test_results.json.  Returns the
        path to the JSON, or None if not found."""
        if not os.path.isdir(save_root):
            return None
        suffix = f"_seed{seed}"
        candidates = []
        for d in os.listdir(save_root):
            full = os.path.join(save_root, d)
            if (os.path.isdir(full) and d.startswith("run_")
                    and d.endswith(suffix)):
                candidates.append(full)
        candidates.sort(reverse=True)  # newest first
        for run_dir in candidates:
            done = os.path.join(run_dir, f"baseline_{baseline_name}",
                                "test_results.json")
            if os.path.exists(done):
                return done
        return None

    results = {}
    for name in selected:
        desc, fn, needs_bert = BASELINES[name]
        print(f"\n{'#'*70}\n#  {desc}\n{'#'*70}")
        bdir = os.path.join(root_dir, f"baseline_{name}")
        os.makedirs(bdir, exist_ok=True)

        # Resume: skip if any sibling run already completed this baseline
        done_file = _find_completed_baseline(save_dir_eff, name, cfg["seed"])
        if done_file:
            print(f"[skip] {name}: already completed at {done_file}")
            with open(done_file, encoding="utf-8") as f:
                results[name] = json.load(f)
            with open(os.path.join(root_dir, "baselines_results.json"),
                      "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            continue

        # Skip baselines that caused a native process crash on a previous run.
        # A sentinel file is written just before each attempt and removed in
        # the finally block.  Native crashes (e.g. CUDA ACCESS_VIOLATION that
        # returns exit-code -1073741819) bypass finally, so the sentinel
        # persists and signals "skip me" on the next restart.
        sentinel_path = os.path.join(bdir, "crash_sentinel.json")
        if os.path.exists(sentinel_path):
            print(f"[skip] {name}: previous attempt crashed the process "
                  f"(sentinel: {sentinel_path}), marking as failed")
            results[name] = {"description": desc,
                             "error": "process crashed during previous attempt"}
            with open(os.path.join(root_dir, "baselines_results.json"),
                      "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            continue

        # Re-seed before EACH baseline so they are mutually reproducible
        # rather than chained on each other's RNG state.
        set_seed(cfg["seed"])
        t0 = time.time()
        with open(sentinel_path, "w", encoding="utf-8") as f:
            json.dump({"started": True,
                       "timestamp": datetime.now().isoformat(),
                       "baseline": name}, f)
        try:
            if needs_bert:
                test_m, best_val, best_ep = fn(
                    splits, mappings,
                    tokenizer=tokenizer, cfg=cfg, device=device,
                    save_dir=bdir, labels_full=labels,
                )
                results[name] = {
                    "description": desc,
                    "best_val_micro_f1": best_val,
                    "best_epoch": best_ep,
                    "test_metrics": test_m,
                    "training_time_seconds": time.time() - t0,
                }
            else:
                test_m = fn(splits, mappings)
                results[name] = {
                    "description": desc,
                    "test_metrics": test_m,
                    "training_time_seconds": time.time() - t0,
                }
        except Exception as e:
            print(f"\n[{name}] FAILED: {e}")
            results[name] = {"description": desc, "error": str(e)}
        finally:
            # Removing the sentinel marks a clean Python exit (success or
            # caught exception).  If the process crashed natively, this
            # finally block never runs → sentinel stays → skip next time.
            if os.path.exists(sentinel_path):
                os.remove(sentinel_path)

        # Save per-baseline test_results.json so a future run can skip it
        if "error" not in results[name]:
            with open(os.path.join(bdir, "test_results.json"),
                      "w", encoding="utf-8") as f:
                json.dump(results[name], f, indent=2, default=str)

        with open(os.path.join(root_dir, "baselines_results.json"),
                  "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Release GPU memory between baselines so later ones don't slow down
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'='*100}")
    print(f"  BASELINES vs V7-default-thresholds (seed={cfg['seed']})")
    print(f"{'='*100}")
    print(f"  {'B':<3}  {'overall μF1':>11}  {'overall mF1':>11}  "
          f"{'W μF1':>7}  {'E μF1':>7}  {'H μF1':>7}  description")
    print("  " + "-" * 96)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<3}  FAILED: {r['error'][:60]}")
            continue
        tm = r["test_metrics"]
        print(
            f"  {name:<3}  "
            f"{tm['overall']['micro_f1']:>11.4f}  "
            f"{tm['overall']['macro_f1']:>11.4f}  "
            f"{tm['warning_s2']['micro_f1']:>7.4f}  "
            f"{tm['environmental_s2']['micro_f1']:>7.4f}  "
            f"{tm['human_s2']['micro_f1']:>7.4f}  "
            f"{r['description']}"
        )
    print()


if __name__ == "__main__":
    main()
