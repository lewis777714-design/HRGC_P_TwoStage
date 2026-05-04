"""
Inference for TwoStageModelV7 using DEFAULT scalar thresholds.

Edit USER CONFIG below before running.  The script does NOT use any
per-label optimised thresholds — it reports performance under fixed
scalar thresholds so the numbers are directly comparable across runs.

Run:
    python infer.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import (
    AccidentDataset,
    load_processed_dataframe,
    parse_hierarchical_labels,
    parse_labels_with_mapping,
)
from model import TwoStageModelV7
from train import set_seed


# ═════════════════════════════════════════════════════════════════
# USER CONFIG — edit these before running
# ═════════════════════════════════════════════════════════════════

# Path to the V7 experiment directory produced by train.py.
# Must contain: best_model_v7.pth, subcategory_mappings.json,
# data_splits/test.csv
EXPERIMENT_DIR   = r"training_results_v7\experiment_20260430_214316_seed42"        # e.g. r"training_results_v7\experiment_20260430_214316_seed42"

# Where to write inference_results_v7/<experiment_basename>/
OUTPUT_ROOT      = r"inference_results_v7"

STAGE1_THRESHOLD = 0.5
STAGE2_THRESHOLD = 0.95
BATCH_SIZE       = 64
BERT_MODEL       = "bert-base-uncased"
SEED             = 42

# Threshold mode:
#   True  → if EXPERIMENT_DIR/optimized_thresholds.json exists, load and
#           use those per-label thresholds.  Falls back to the scalar
#           STAGE1_THRESHOLD / STAGE2_THRESHOLD above when the JSON is
#           missing.
#   False → always use the scalar STAGE1_THRESHOLD / STAGE2_THRESHOLD.
USE_OPTIMIZED_THRESHOLDS = False


def _binary_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _multilabel_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    micro = _binary_f1(y_true.ravel(), y_pred.ravel())
    macro = float(np.mean([_binary_f1(y_true[:, j], y_pred[:, j])
                           for j in range(y_true.shape[1])]))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    per_label = {
        j: _binary_f1(y_true[:, j], y_pred[:, j])
        for j in range(y_true.shape[1])
    }
    return {
        "micro_f1": micro, "macro_f1": macro,
        "precision_micro": p, "recall_micro": r,
        "per_label": per_label,
    }


def load_model(model_path: str, mappings, device, bert_model="bert-base-uncased"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model = TwoStageModelV7(
        bert_model, dropout=0.3, subcategory_mappings=mappings,
        tokenizer=None,                      # skip semantic-query encode
        # state_dict will overwrite the (unused) random init
    )
    raw = torch.load(model_path, map_location=device, weights_only=False)
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ignorable = {"bert.embeddings.position_ids"}
    real_missing = [k for k in missing if k not in ignorable]
    real_unexpected = [k for k in unexpected if k not in ignorable]
    if real_missing:
        print(f"[WARN] Missing keys: {real_missing}")
    if real_unexpected:
        print(f"[WARN] Unexpected keys: {real_unexpected}")
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, s1_t, s2_t_w, s2_t_e, s2_t_h):
    """Forward pass on the loader; return raw probs, ground truth, and
    Stage-1 binary predictions.  Stage-2 thresholds are per-label vectors
    (length = nw / ne / nh).  PRED-GATING is applied in evaluate(), not
    here, so this function just reports the raw outputs.
    """
    s1_y_l = []
    w_y_l, e_y_l, h_y_l = [], [], []
    s1_probs_all, w_probs_all, e_probs_all, h_probs_all = [], [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = model(ids, mask)
        s1_probs_all.append(out["stage1"]["probs"].cpu().numpy())
        w_probs_all.append(
            out["stage2"]["warning"]["subcategory_probs"].cpu().numpy())
        e_probs_all.append(
            out["stage2"]["environmental"]["subcategory_probs"].cpu().numpy())
        h_probs_all.append(
            out["stage2"]["human"]["subcategory_probs"].cpu().numpy())
        s1_y_l.append(batch["main_labels"].numpy().astype(int))
        w_y_l.append(batch["warning_subcat_labels"].numpy().astype(int))
        e_y_l.append(batch["env_subcat_labels"].numpy().astype(int))
        h_y_l.append(batch["human_subcat_labels"].numpy().astype(int))

    s1_probs = np.vstack(s1_probs_all)
    w_probs  = np.vstack(w_probs_all)
    e_probs  = np.vstack(e_probs_all)
    h_probs  = np.vstack(h_probs_all)
    s1_y     = np.vstack(s1_y_l)
    w_y      = np.vstack(w_y_l)
    e_y      = np.vstack(e_y_l)
    h_y      = np.vstack(h_y_l)

    s1_p = (s1_probs > s1_t).astype(int)
    return {
        "s1":            {"y": s1_y, "p": s1_p, "probs": s1_probs,
                          "thresh": s1_t},
        "warning":       {"y": w_y, "probs": w_probs, "thresh": s2_t_w,
                          "s1_col": 0},
        "environmental": {"y": e_y, "probs": e_probs, "thresh": s2_t_e,
                          "s1_col": 1},
        "human":         {"y": h_y, "probs": h_probs, "thresh": s2_t_h,
                          "s1_col": 2},
    }


def evaluate_pred_gated(out):
    """PRED-GATED metrics matching threshold_search_v7:
    for each Stage-2 category, only evaluate samples whose Stage-1
    prediction for that category is positive (mask the rest out)."""
    s1_pred = out["s1"]["p"]
    summary = {"stage1": _multilabel_metrics(out["s1"]["y"], s1_pred)}

    summary["stage2"] = {}
    tp_t = fp_t = fn_t = 0
    for cat in ("warning", "environmental", "human"):
        d = out[cat]
        probs   = d["probs"]
        labels  = d["y"]
        s2_pred = (probs > d["thresh"][None, :]).astype(int) \
                    if isinstance(d["thresh"], np.ndarray) \
                    else (probs > d["thresh"]).astype(int)
        mask    = s1_pred[:, d["s1_col"]] > 0
        if mask.sum() == 0:
            summary["stage2"][cat] = {
                "micro_f1": 0.0, "macro_f1": 0.0,
                "precision_micro": 0.0, "recall_micro": 0.0,
                "n_samples_gated_in": 0,
            }
            continue
        yt = labels[mask]
        yp = s2_pred[mask]
        m  = _multilabel_metrics(yt, yp)
        m["n_samples_gated_in"] = int(mask.sum())
        summary["stage2"][cat] = m

        tp_t += int(((yt == 1) & (yp == 1)).sum())
        fp_t += int(((yt == 0) & (yp == 1)).sum())
        fn_t += int(((yt == 1) & (yp == 0)).sum())

    p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    summary["overall_subcategories"] = {
        "micro_f1": f, "precision_micro": p, "recall_micro": r,
    }
    return summary


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    # CLI overrides for any USER CONFIG constant above.  If a flag is not
    # passed, the script falls back to the constant in this file.
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=str, default=None)
    p.add_argument("--s1-threshold",   type=float, default=None)
    p.add_argument("--s2-threshold",   type=float, default=None)
    p.add_argument("--bert-model",     type=str,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--output-root",    type=str,   default=None)
    p.add_argument(
        "--no-optimized-thresholds", action="store_true",
        help="Force scalar STAGE1_THRESHOLD/STAGE2_THRESHOLD even if "
             "optimized_thresholds.json exists in the experiment dir.",
    )
    args = p.parse_args()

    exp_dir       = (args.experiment_dir or EXPERIMENT_DIR).strip()
    s1_thresh     = args.s1_threshold if args.s1_threshold is not None else STAGE1_THRESHOLD
    s2_thresh     = args.s2_threshold if args.s2_threshold is not None else STAGE2_THRESHOLD
    bert_model    = args.bert_model    or BERT_MODEL
    batch_size    = args.batch_size if args.batch_size is not None else BATCH_SIZE
    seed          = args.seed       if args.seed       is not None else SEED
    output_root   = args.output_root or OUTPUT_ROOT
    use_opt       = USE_OPTIMIZED_THRESHOLDS and not args.no_optimized_thresholds

    if not exp_dir:
        raise ValueError(
            "Set EXPERIMENT_DIR at the top of infer.py, "
            "or pass --experiment-dir on the command line."
        )
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Not a directory: {exp_dir}")

    model_path     = os.path.join(exp_dir, "best_model_v7.pth")
    mappings_path  = os.path.join(exp_dir, "subcategory_mappings.json")
    test_csv       = os.path.join(exp_dir, "data_splits", "test.csv")
    threshold_json = os.path.join(exp_dir, "optimized_thresholds.json")
    for p_ in (model_path, mappings_path, test_csv):
        if not os.path.exists(p_):
            raise FileNotFoundError(p_)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment:   {exp_dir}")

    with open(mappings_path, encoding="utf-8") as f:
        mappings = json.load(f)
    nw = len(mappings["warning"])
    ne = len(mappings["environmental"])
    nh = len(mappings["human"])

    # ── Resolve thresholds: optimized JSON if available, else scalar ──
    s1_vec = np.full(3, s1_thresh, dtype=np.float32)
    w_vec  = np.full(nw, s2_thresh, dtype=np.float32)
    e_vec  = np.full(ne, s2_thresh, dtype=np.float32)
    h_vec  = np.full(nh, s2_thresh, dtype=np.float32)
    threshold_mode = f"scalar (S1={s1_thresh}, S2={s2_thresh})"
    if use_opt and os.path.exists(threshold_json):
        with open(threshold_json, encoding="utf-8") as f:
            tj = json.load(f)
        s1_vec = np.asarray(tj["stage1"], dtype=np.float32)
        s2_vec = np.asarray(tj["stage2"], dtype=np.float32)
        w_vec  = s2_vec[:nw]
        e_vec  = s2_vec[nw:nw + ne]
        h_vec  = s2_vec[nw + ne:]
        threshold_mode = (
            f"per-label optimized (loaded from {threshold_json}; "
            f"S1 mean={s1_vec.mean():.3f}, S2 mean={s2_vec.mean():.3f})"
        )
    elif use_opt:
        print(f"[INFO] USE_OPTIMIZED_THRESHOLDS=True but no JSON at "
              f"{threshold_json} — falling back to scalars.")
    print(f"Thresholds:   {threshold_mode}")

    tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)
    model = load_model(model_path, mappings, device, bert_model)

    test_df = load_processed_dataframe(test_csv)
    test_texts = test_df["DETAILED_DESCRIPTION"].tolist()
    test_labels = parse_labels_with_mapping(test_df, mappings)
    test_ds = AccidentDataset(test_texts, test_labels, tokenizer, mappings)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    out = run_inference(
        model, test_loader, device,
        s1_t=s1_vec, s2_t_w=w_vec, s2_t_e=e_vec, s2_t_h=h_vec,
    )
    summary = evaluate_pred_gated(out)
    summary["thresholds_used"] = threshold_mode
    summary["stage1_threshold"] = s1_vec.tolist()
    summary["stage2_threshold"] = {
        "warning":       w_vec.tolist(),
        "environmental": e_vec.tolist(),
        "human":         h_vec.tolist(),
    }

    print("\n--- Stage 1 ---")
    print(f"  micro-F1={summary['stage1']['micro_f1']:.4f}  "
          f"macro-F1={summary['stage1']['macro_f1']:.4f}")
    for cat in ("warning", "environmental", "human"):
        d = summary["stage2"][cat]
        print(f"\n--- Stage 2 [{cat}] (PRED-GATED, "
              f"n={d.get('n_samples_gated_in', 0)}) ---")
        print(f"  micro-F1={d['micro_f1']:.4f}  macro-F1={d['macro_f1']:.4f}")

    o = summary["overall_subcategories"]
    print(f"\n--- Overall Subcategories (PRED-GATED) ---")
    print(f"  micro-F1={o['micro_f1']:.4f}  P={o['precision_micro']:.4f}  "
          f"R={o['recall_micro']:.4f}")

    if model.use_flat_head:
        with torch.no_grad():
            summary["flat_alphas"] = {
                "warning":       float(torch.sigmoid(model.alpha_warning).item()),
                "environmental": float(torch.sigmoid(model.alpha_env).item()),
                "human":         float(torch.sigmoid(model.alpha_human).item()),
            }
        try:
            print(f"\n  Flat-head α (attention share): {summary['flat_alphas']}")
        except UnicodeEncodeError:
            print(f"\n  Flat-head alpha (attention share): "
                  f"{summary['flat_alphas']}")

    out_dir = os.path.join(output_root, os.path.basename(exp_dir))
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "test_metrics_summary.json")
    # Convert per-label dict keys to strings for JSON
    def _stringify(d):
        if isinstance(d, dict):
            return {str(k): _stringify(v) for k, v in d.items()}
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        return d
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_stringify(summary), f, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
