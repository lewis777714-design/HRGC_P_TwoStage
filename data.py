"""
Data utilities for the railway-crossing accident text classifier.

Self-contained: only depends on numpy / pandas / torch / transformers.
No imports from outside this submission folder.

Public surface used by train.py / infer.py / ablation.py / threshold_search.py /
baselines.py:

    parse_hierarchical_labels(df)            -> (labels, subcategory_mappings)
    parse_labels_with_mapping(df, mappings)  -> labels  (use a fixed mapping)
    AccidentDataset(texts, labels, tokenizer, mappings, max_length=512)
    split_dataset_by_subcategory_ratio(texts, labels, mappings, ratios, seed)
    build_datasets_from_splits(splits, tokenizer, mappings)
    build_dataloaders(datasets, batch_size, sampler=None)
    create_balanced_sampler(labels, main_factor_counts)
    save_splits_to_csv(splits, df, out_dir)
    load_processed_dataframe(csv_path)
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ─────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────

def load_processed_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["DETAILED_DESCRIPTION"].notna()].copy()
    return df


# ─────────────────────────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────────────────────────

def _split_factor_labels(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "Unidentified":
        return []
    return [
        token.strip()
        for token in text.split(",")
        if token.strip() and token.strip() != "Unidentified"
    ]


def parse_hierarchical_labels(
    df: pd.DataFrame,
) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """Parse the three main-factor columns into per-sample label dicts and
    build a {category -> {name -> index}} mapping. The mapping is built
    from the order of first appearance in *df* so it is deterministic
    given the input dataframe."""
    human_subcats: Dict[str, int] = {}
    env_subcats: Dict[str, int] = {}
    warning_subcats: Dict[str, int] = {}
    h_idx = e_idx = w_idx = 0

    cols = {
        "human": "HUMAN_FACTORS",
        "env": "ENVIRONMENTAL_FACTORS",
        "warning": "WARNING_DEVICE_ISSUES",
    }
    n = len(df)

    def col(name):
        return df[cols[name]].tolist() if cols[name] in df.columns else [None] * n

    h_vals, e_vals, w_vals = col("human"), col("env"), col("warning")

    # First pass — discover subcategory names in order of appearance
    for h_raw, e_raw, w_raw in zip(h_vals, e_vals, w_vals):
        for s in _split_factor_labels(h_raw):
            if s not in human_subcats:
                human_subcats[s] = h_idx; h_idx += 1
        for s in _split_factor_labels(e_raw):
            if s not in env_subcats:
                env_subcats[s] = e_idx; e_idx += 1
        for s in _split_factor_labels(w_raw):
            if s not in warning_subcats:
                warning_subcats[s] = w_idx; w_idx += 1

    mappings = {
        "human": human_subcats,
        "environmental": env_subcats,
        "warning": warning_subcats,
    }

    parsed: List[Dict] = []
    for h_raw, e_raw, w_raw in zip(h_vals, e_vals, w_vals):
        rec = {
            "main_factors": {
                "HUMAN_FACTORS": 0,
                "ENVIRONMENTAL_FACTORS": 0,
                "WARNING_DEVICE_ISSUES": 0,
            },
            "subcategories": {"human": [], "environmental": [], "warning": []},
        }
        h = _split_factor_labels(h_raw)
        if h:
            rec["main_factors"]["HUMAN_FACTORS"] = 1
            rec["subcategories"]["human"] = [
                human_subcats[s] for s in h if s in human_subcats]
        e = _split_factor_labels(e_raw)
        if e:
            rec["main_factors"]["ENVIRONMENTAL_FACTORS"] = 1
            rec["subcategories"]["environmental"] = [
                env_subcats[s] for s in e if s in env_subcats]
        w = _split_factor_labels(w_raw)
        if w:
            rec["main_factors"]["WARNING_DEVICE_ISSUES"] = 1
            rec["subcategories"]["warning"] = [
                warning_subcats[s] for s in w if s in warning_subcats]
        parsed.append(rec)

    return parsed, mappings


def parse_labels_with_mapping(
    df: pd.DataFrame, mappings: Dict[str, Dict[str, int]],
) -> List[Dict]:
    """Same as parse_hierarchical_labels but uses a *given* mapping (so the
    label index space matches between train/val/test or train/test runs)."""
    parsed: List[Dict] = []
    n = len(df)

    def col(name):
        return df[name].tolist() if name in df.columns else [None] * n

    h_vals = col("HUMAN_FACTORS")
    e_vals = col("ENVIRONMENTAL_FACTORS")
    w_vals = col("WARNING_DEVICE_ISSUES")

    for h_raw, e_raw, w_raw in zip(h_vals, e_vals, w_vals):
        rec = {
            "main_factors": {
                "HUMAN_FACTORS": 0,
                "ENVIRONMENTAL_FACTORS": 0,
                "WARNING_DEVICE_ISSUES": 0,
            },
            "subcategories": {"human": [], "environmental": [], "warning": []},
        }
        h = _split_factor_labels(h_raw)
        if h:
            rec["main_factors"]["HUMAN_FACTORS"] = 1
            rec["subcategories"]["human"] = [
                mappings["human"][s] for s in h if s in mappings["human"]]
        e = _split_factor_labels(e_raw)
        if e:
            rec["main_factors"]["ENVIRONMENTAL_FACTORS"] = 1
            rec["subcategories"]["environmental"] = [
                mappings["environmental"][s] for s in e
                if s in mappings["environmental"]]
        w = _split_factor_labels(w_raw)
        if w:
            rec["main_factors"]["WARNING_DEVICE_ISSUES"] = 1
            rec["subcategories"]["warning"] = [
                mappings["warning"][s] for s in w if s in mappings["warning"]]
        parsed.append(rec)
    return parsed


# ─────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────

class AccidentDataset(Dataset):
    """Pre-tokenises all texts at construction time so __getitem__ is
    purely tensor-indexed (no tokenizer in the data-loader worker)."""

    def __init__(self, texts: List[str], labels: List[Dict], tokenizer,
                 mappings: Dict[str, Dict[str, int]], max_length: int = 512):
        self.labels = labels
        self.mappings = mappings
        self.n_warning = len(mappings["warning"])
        self.n_env = len(mappings["environmental"])
        self.n_human = len(mappings["human"])

        self.input_ids: List[List[int]] = []
        self.attention_masks: List[List[int]] = []
        n_panic_fallback = 0
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            text = text.encode("utf-8", errors="ignore").decode("utf-8")
            text = "".join(c for c in text if c >= " " or c in "\n\t")
            try:
                enc = tokenizer(
                    text, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors=None,
                    return_attention_mask=True,
                )
            except BaseException:
                # HuggingFace fast tokenizer (Rust) sometimes panics with
                # "NormalizedString bad split" on awkward Unicode sequences.
                # Strip to ASCII and retry — costs us non-ASCII chars in
                # *that one row* but keeps the dataset build alive.
                ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
                enc = tokenizer(
                    ascii_text, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors=None,
                    return_attention_mask=True,
                )
                n_panic_fallback += 1
            self.input_ids.append(enc["input_ids"])
            self.attention_masks.append(enc["attention_mask"])
        if n_panic_fallback > 0:
            print(f"[AccidentDataset] {n_panic_fallback}/{len(texts)} rows "
                  f"fell back to ASCII due to fast-tokenizer panic.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        L = self.labels[idx]
        main_labels = torch.zeros(3)
        main_labels[0] = L["main_factors"]["WARNING_DEVICE_ISSUES"]
        main_labels[1] = L["main_factors"]["ENVIRONMENTAL_FACTORS"]
        main_labels[2] = L["main_factors"]["HUMAN_FACTORS"]

        w_lab = torch.zeros(self.n_warning)
        for k in L["subcategories"]["warning"]:
            w_lab[k] = 1
        e_lab = torch.zeros(self.n_env)
        for k in L["subcategories"]["environmental"]:
            e_lab[k] = 1
        h_lab = torch.zeros(self.n_human)
        for k in L["subcategories"]["human"]:
            h_lab[k] = 1

        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx],
                                           dtype=torch.long),
            "main_labels": main_labels,
            "warning_subcat_labels": w_lab,
            "env_subcat_labels": e_lab,
            "human_subcat_labels": h_lab,
        }


# ─────────────────────────────────────────────────────────────────
# Iterative stratified split (multi-label)
# ─────────────────────────────────────────────────────────────────

def _compute_targets(total: int, ratios: List[float]) -> List[int]:
    raw = [r * total for r in ratios]
    floors = [math.floor(v) for v in raw]
    rem = total - sum(floors)
    frac = sorted(
        [(raw[i] - floors[i], -ratios[i], i) for i in range(len(ratios))],
        reverse=True,
    )
    for k in range(rem):
        floors[frac[k][2]] += 1
    return floors


def _flatten_labels(labels, mappings):
    """Map each (category, subcat_idx) pair to a single flat index 0..K-1."""
    idx_map: Dict[Tuple[str, int], int] = {}
    rev_map: Dict[int, Tuple[str, int]] = {}
    cur = 0
    for cat in ("human", "environmental", "warning"):
        for _, sub_idx in sorted(mappings[cat].items(), key=lambda x: x[1]):
            idx_map[(cat, sub_idx)] = cur
            rev_map[cur] = (cat, sub_idx)
            cur += 1
    flat = []
    for L in labels:
        out = []
        for cat in ("human", "environmental", "warning"):
            for s in L["subcategories"][cat]:
                out.append(idx_map[(cat, s)])
        flat.append(out)
    return flat, rev_map


def split_dataset_by_subcategory_ratio(
    texts: List[str], labels: List[Dict],
    mappings: Dict[str, Dict[str, int]],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42,
) -> Dict[str, Dict[str, List]]:
    """Greedy iterative-stratification split. Honours the ratio for total
    samples AND for each subcategory's positive count."""
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must sum to 1")

    set_names = ["train", "val", "test"]
    flat, rev_map = _flatten_labels(labels, mappings)
    K = len(rev_map)

    target_samples = _compute_targets(len(texts), list(ratios))
    label_totals = [0] * K
    for inds in flat:
        for k in inds:
            label_totals[k] += 1
    target_label = {
        s: _compute_targets(label_totals[k], list(ratios))
        if False else None
        for s in range(3) for k in range(K)
    }
    # Build per-set per-label targets explicitly.
    target_label = {s: [0] * K for s in range(3)}
    for k in range(K):
        per = _compute_targets(label_totals[k], list(ratios))
        for s in range(3):
            target_label[s][k] = per[s]

    cur_label = {s: [0] * K for s in range(3)}
    cur_samples = [0] * 3

    rng = random.Random(random_state)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    indices.sort(key=lambda i: len(flat[i]), reverse=True)

    assignments: Dict[str, List[int]] = {n: [] for n in set_names}
    for idx in indices:
        L_inds = flat[idx]
        if not L_inds:
            avail = [s for s in range(3) if cur_samples[s] < target_samples[s]]
            if not avail:
                avail = [0, 1, 2]
            chosen = min(avail, key=lambda s: cur_samples[s])
        else:
            scores = []
            for s in range(3):
                cap_pen = max(0, cur_samples[s] + 1 - target_samples[s])
                deficit = sum(target_label[s][k] - cur_label[s][k]
                              for k in L_inds)
                scores.append((deficit, -cap_pen, -cur_samples[s], s))
            scores.sort(reverse=True)
            chosen = scores[0][3]

        assignments[set_names[chosen]].append(idx)
        cur_samples[chosen] += 1
        for k in L_inds:
            cur_label[chosen][k] += 1

    splits = {}
    for n in set_names:
        assignments[n].sort()
        splits[n] = {
            "indices": assignments[n],
            "texts": [texts[i] for i in assignments[n]],
            "labels": [labels[i] for i in assignments[n]],
        }
    return splits


# ─────────────────────────────────────────────────────────────────
# Loader builders
# ─────────────────────────────────────────────────────────────────

def build_datasets_from_splits(splits, tokenizer, mappings, max_length=512):
    out = {}
    for name, sp in splits.items():
        out[name] = AccidentDataset(
            sp["texts"], sp["labels"], tokenizer, mappings, max_length,
        )
    return out


def build_dataloaders(datasets, batch_size: int, sampler=None,
                      shuffle_train: bool = True):
    loaders = {}
    for name, ds in datasets.items():
        if name == "train":
            loaders[name] = DataLoader(
                ds, batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None and shuffle_train),
                num_workers=0, pin_memory=False,
            )
        else:
            loaders[name] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
            )
    return loaders


def create_balanced_sampler(labels: List[Dict],
                            main_factor_counts: Dict[str, int]):
    """Inverse-frequency weighted sampler over the 3 main-factor columns."""
    weights = []
    total = sum(main_factor_counts.values()) or 1
    for L in labels:
        w = 0.0
        for factor, val in L["main_factors"].items():
            if val == 1:
                c = main_factor_counts.get(factor, 1) or 1
                w += total / (3 * c)
        if w == 0:
            w = 1.0
        weights.append(w)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True,
    )


def save_splits_to_csv(splits, full_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, sp in splits.items():
        sub = full_df.iloc[sp["indices"]].copy()
        path = os.path.join(out_dir, f"{name}.csv")
        sub.to_csv(path, index=False)
        paths[name] = path
    return paths


def count_main_factors(labels: List[Dict]) -> Dict[str, int]:
    counts = {
        "WARNING_DEVICE_ISSUES": 0,
        "ENVIRONMENTAL_FACTORS": 0,
        "HUMAN_FACTORS": 0,
    }
    for L in labels:
        for k, v in L["main_factors"].items():
            if v == 1:
                counts[k] += 1
    return counts
