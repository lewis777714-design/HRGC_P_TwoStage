"""
Train TwoStageModelV7 end-to-end.

This is the canonical V7 training entry point.  It:
  1. Loads + splits data (70 / 15 / 15 stratified by subcategory)
  2. Builds the V7 model
  3. Joint-trains BERT + Stage-1 head + Stage-2 attention/flat heads
  4. Saves the best checkpoint by validation overall μ-F1

Loss objective (per batch):
    L = L_S2_mixed
      + flat_aux_w · L_S2_flat            (keep flat branch alive)
      + aux_w(t)  · L_S1                  (cosine-decayed aux Stage-1 loss)

Run:
    python train.py
    python train.py --data path/to/data.csv --epochs 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import glob as _glob
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from data import (
    AccidentDataset,
    build_datasets_from_splits,
    build_dataloaders,
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

# Absolute or relative path to the training CSV.
DATA_FILE  = r"HRGC_P_aug.csv"           # e.g. r"..\augmented_v4_20260418_113621.csv"

# Where to write training_results_v7/experiment_<ts>_seed<N>/
SAVE_DIR   = r"training_results_v7"

SEED       = 123
EPOCHS     = 500
PATIENCE   = 50
BATCH_SIZE = 128
from model import (
    PerClassAsymmetricLoss,
    TwoStageModelV7,
    compute_pos_class_weights,
    get_layer_lrd_param_groups,
)


# ─────────────────────────────────────────────────────────────────
# Default training hyper-parameters
# ─────────────────────────────────────────────────────────────────

CONFIG = {
    "data_file": DATA_FILE,
    "bert_model": "bert-base-uncased",
    "seed": SEED,

    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": 4,
    "dropout": 0.3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,

    "bert_lr": 2e-5,
    "head_lr": 5e-4,
    "layer_decay": 0.85,

    "epochs": EPOCHS,
    "patience": PATIENCE,
    "min_epochs": 5,

    # Loss
    "loss_gamma_neg": 3.0,
    "loss_gamma_pos": 1.0,
    "loss_clip": 0.05,
    "per_class_pos_weight_max": 8.0,
    "per_class_pos_weight_power": 0.5,

    # Aux Stage-1 cosine schedule
    "aux_w_start": 0.5,
    "aux_w_end": 0.05,
    "aux_w_schedule_epochs": 30,

    # Flat-head fusion
    "flat_aux_weight": 0.2,
    "flat_alpha_init": 0.7,

    # Default thresholds for evaluation (V7 ships without a threshold-search
    # dependency; see threshold_search.py for tuning)
    "stage1_threshold": 0.5,
    "stage2_threshold": 0.5,

    "save_dir": SAVE_DIR,
    "checkpoint_every": 10,
}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def set_seed(seed: int, strict: bool = False):
    """Seed every RNG that influences training.

    When ``strict=True`` we additionally force cuDNN to its deterministic
    code path.  That removes the ~1e-4 run-to-run jitter you otherwise get
    on CUDA, but typical training is 10-15% slower.  Off by default to
    match V6's behaviour; enable with ``--strict-determinism``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def aux_w_at_epoch(epoch, schedule_epochs, w_start, w_end):
    """Cosine decay from w_start (epoch 0) to w_end (schedule_epochs).
    Stays flat at w_end for any epoch >= schedule_epochs."""
    if schedule_epochs <= 0 or epoch >= schedule_epochs:
        return float(w_end)
    t = float(epoch) / float(schedule_epochs)
    return float(w_end + 0.5 * (w_start - w_end) * (1.0 + math.cos(math.pi * t)))


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


def _multilabel_f1(y_true, y_pred, average="micro"):
    """Sklearn-free micro/macro F1 over a [N, K] binary matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if average == "micro":
        return _binary_f1(y_true.ravel(), y_pred.ravel())
    K = y_true.shape[1]
    return float(np.mean([_binary_f1(y_true[:, j], y_pred[:, j])
                          for j in range(K)]))


def _multilabel_pr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r


def _train_label_matrices(labels, train_idx, mappings):
    nw = len(mappings["warning"])
    ne = len(mappings["environmental"])
    nh = len(mappings["human"])
    main = np.zeros((len(train_idx), 3), dtype=np.float32)
    w = np.zeros((len(train_idx), nw), dtype=np.float32)
    e = np.zeros((len(train_idx), ne), dtype=np.float32)
    h = np.zeros((len(train_idx), nh), dtype=np.float32)
    for row, i in enumerate(train_idx):
        L = labels[i]
        mf = L["main_factors"]
        main[row, 0] = mf["WARNING_DEVICE_ISSUES"]
        main[row, 1] = mf["ENVIRONMENTAL_FACTORS"]
        main[row, 2] = mf["HUMAN_FACTORS"]
        for k in L["subcategories"]["warning"]:
            w[row, k] = 1.0
        for k in L["subcategories"]["environmental"]:
            e[row, k] = 1.0
        for k in L["subcategories"]["human"]:
            h[row, k] = 1.0
    return main, w, e, h


def build_criteria(config, main_mat, w_mat, e_mat, h_mat):
    base = dict(
        gamma_neg=config["loss_gamma_neg"],
        gamma_pos=config["loss_gamma_pos"],
        clip=config["loss_clip"],
    )
    p = config["per_class_pos_weight_power"]
    wm = config["per_class_pos_weight_max"]
    return {
        "stage1":  PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(main_mat, p, wm)),
        "warning": PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(w_mat, p, wm)),
        "env":     PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(e_mat, p, wm)),
        "human":   PerClassAsymmetricLoss(
            **base, pos_class_weights=compute_pos_class_weights(h_mat, p, wm)),
    }


# ─────────────────────────────────────────────────────────────────
# Evaluation (default scalar thresholds — see threshold_search.py for tuning)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    s1_t, s1_l = [], []
    w_t, w_l = [], []
    e_t, e_l = [], []
    h_t, h_l = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, mask)
        s1_l.append(out["stage1"]["logits"].cpu())
        s1_t.append(batch["main_labels"])
        w_l.append(out["stage2"]["warning"]["subcategory_logits"].cpu())
        w_t.append(batch["warning_subcat_labels"])
        e_l.append(out["stage2"]["environmental"]["subcategory_logits"].cpu())
        e_t.append(batch["env_subcat_labels"])
        h_l.append(out["stage2"]["human"]["subcategory_logits"].cpu())
        h_t.append(batch["human_subcat_labels"])
    return {
        "s1_true": torch.cat(s1_t).numpy(), "s1_logits": torch.cat(s1_l).numpy(),
        "w_true":  torch.cat(w_t).numpy(),  "w_logits":  torch.cat(w_l).numpy(),
        "e_true":  torch.cat(e_t).numpy(),  "e_logits":  torch.cat(e_l).numpy(),
        "h_true":  torch.cat(h_t).numpy(),  "h_logits":  torch.cat(h_l).numpy(),
    }


def evaluate(model, loader, device, s1_thresh=0.5, s2_thresh=0.5):
    raw = collect_predictions(model, loader, device)
    s1_p = (1.0 / (1.0 + np.exp(-raw["s1_logits"])) > s1_thresh).astype(int)
    w_p  = (1.0 / (1.0 + np.exp(-raw["w_logits"])) > s2_thresh).astype(int)
    e_p  = (1.0 / (1.0 + np.exp(-raw["e_logits"])) > s2_thresh).astype(int)
    h_p  = (1.0 / (1.0 + np.exp(-raw["h_logits"])) > s2_thresh).astype(int)
    s1_y = raw["s1_true"].astype(int)
    w_y  = raw["w_true"].astype(int)
    e_y  = raw["e_true"].astype(int)
    h_y  = raw["h_true"].astype(int)

    metrics = {
        "stage1":     {"micro_f1": _multilabel_f1(s1_y, s1_p, "micro"),
                       "macro_f1": _multilabel_f1(s1_y, s1_p, "macro")},
        "warning_s2": {"micro_f1": _multilabel_f1(w_y, w_p, "micro"),
                       "macro_f1": _multilabel_f1(w_y, w_p, "macro")},
        "env_s2":     {"micro_f1": _multilabel_f1(e_y, e_p, "micro"),
                       "macro_f1": _multilabel_f1(e_y, e_p, "macro")},
        "human_s2":   {"micro_f1": _multilabel_f1(h_y, h_p, "micro"),
                       "macro_f1": _multilabel_f1(h_y, h_p, "macro")},
    }
    all_y = np.hstack([w_y, e_y, h_y])
    all_p = np.hstack([w_p, e_p, h_p])
    p, r = _multilabel_pr(all_y, all_p)
    metrics["overall"] = {
        "micro_f1": _multilabel_f1(all_y, all_p, "micro"),
        "macro_f1": _multilabel_f1(all_y, all_p, "macro"),
        "precision": p, "recall": r,
    }
    return metrics, raw


def print_metrics(m, title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    for k in ("warning_s2", "env_s2", "human_s2", "overall", "stage1"):
        d = m.get(k)
        if d is None:
            continue
        print(f"  {k:<11}  micro={d.get('micro_f1', 0):.4f}  "
              f"macro={d.get('macro_f1', 0):.4f}")
    print()


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, criteria, config, device, save_dir,
          resume_checkpoint=None):
    pgs = get_layer_lrd_param_groups(
        model, config["bert_lr"], config["head_lr"],
        config["layer_decay"], config["weight_decay"],
    )
    optimizer = torch.optim.AdamW(pgs)

    accum = config["gradient_accumulation_steps"]
    total_steps = max(len(train_loader) // accum, 1) * config["epochs"]
    warmup = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    if config["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model.to(device)

    flat_w = config["flat_aux_weight"]
    aw_start = config["aux_w_start"]
    aw_end = config["aux_w_end"]
    aw_sched = config["aux_w_schedule_epochs"]
    max_grad = config["max_grad_norm"]
    all_params = [p for g in pgs for p in g["params"]]

    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    best_path = os.path.join(save_dir, "best_model_v7.pth")
    ckpt_every = config.get("checkpoint_every", 10)
    start_epoch = 0

    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"\n[Resume] Loading checkpoint: {resume_checkpoint}", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_f1     = ckpt.get("best_val_f1", -1.0)
        start_epoch = ckpt["epoch"]
        best_epoch  = start_epoch
        print(f"[Resume] epoch={start_epoch}  best_val_f1={best_f1:.4f}",
              flush=True)

    print(f"\n{'='*70}")
    print(f"  V7 Training  ({config['epochs']} max epochs, "
          f"patience={config['patience']})")
    print(f"  Effective batch = {config['batch_size'] * accum}")
    print(f"  Aux S1 weight: {aw_start} -> {aw_end} over {aw_sched} epochs")
    print(f"  Flat aux weight={flat_w}  Loss=PerClassASL("
          f"gn={config['loss_gamma_neg']}, "
          f"wmax={config['per_class_pos_weight_max']})")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, config["epochs"]):
        aux_w = aux_w_at_epoch(epoch, aw_sched, aw_start, aw_end)
        model.train()
        sums = {"loss": 0.0, "s1": 0.0, "s2": 0.0, "flat": 0.0}
        nb = len(train_loader)
        optimizer.zero_grad()

        for bi, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            main_y = batch["main_labels"].float().to(device)
            w_y = batch["warning_subcat_labels"].float().to(device)
            e_y = batch["env_subcat_labels"].float().to(device)
            h_y = batch["human_subcat_labels"].float().to(device)
            out = model(ids, mask)

            s1_loss = criteria["stage1"](out["stage1"]["logits"], main_y)
            s2_loss = (
                criteria["warning"](
                    out["stage2"]["warning"]["subcategory_logits"], w_y)
                + criteria["env"](
                    out["stage2"]["environmental"]["subcategory_logits"], e_y)
                + criteria["human"](
                    out["stage2"]["human"]["subcategory_logits"], h_y)
            ) / 3.0

            if model.use_flat_head:
                flat_loss = (
                    criteria["warning"](
                        out["stage2"]["warning"]["flat_logits"], w_y)
                    + criteria["env"](
                        out["stage2"]["environmental"]["flat_logits"], e_y)
                    + criteria["human"](
                        out["stage2"]["human"]["flat_logits"], h_y)
                ) / 3.0
            else:
                flat_loss = torch.zeros((), device=device)

            loss = s2_loss + flat_w * flat_loss + aux_w * s1_loss
            (loss / accum).backward()

            sums["loss"] += loss.item()
            sums["s1"]   += s1_loss.item()
            sums["s2"]   += s2_loss.item()
            sums["flat"] += flat_loss.item()

            if ((bi + 1) % accum == 0) or ((bi + 1) == nb):
                torch.nn.utils.clip_grad_norm_(all_params, max_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (bi + 1) % 20 == 0 or (bi + 1) == nb:
                print(f"  [Ep {epoch+1}] {bi+1}/{nb}  "
                      f"loss={loss.item():.4f}  "
                      f"(s1={s1_loss.item():.4f} s2={s2_loss.item():.4f} "
                      f"flat={flat_loss.item():.4f})  aux_w={aux_w:.3f}",
                      flush=True)

        avg = {k: v / max(nb, 1) for k, v in sums.items()}
        val_m, _ = evaluate(model, val_loader, device,
                            s1_thresh=config["stage1_threshold"],
                            s2_thresh=config["stage2_threshold"])
        val_f1 = val_m["overall"]["micro_f1"]

        msg = (
            f"\n  Ep {epoch+1}/{config['epochs']}  "
            f"train: loss={avg['loss']:.4f} (s1={avg['s1']:.4f} "
            f"s2={avg['s2']:.4f} flat={avg['flat']:.4f})\n"
            f"           val: S1 μF1={val_m['stage1']['micro_f1']:.4f}  "
            f"overall μF1={val_f1:.4f}  "
            f"| W={val_m['warning_s2']['micro_f1']:.4f}  "
            f"E={val_m['env_s2']['micro_f1']:.4f}  "
            f"H={val_m['human_s2']['micro_f1']:.4f}"
        )
        if model.use_flat_head:
            with torch.no_grad():
                a_w = torch.sigmoid(model.alpha_warning).item()
                a_e = torch.sigmoid(model.alpha_env).item()
                a_h = torch.sigmoid(model.alpha_human).item()
            msg += (f"\n           α: W={a_w:.3f} E={a_e:.3f} H={a_h:.3f}  "
                    f"aux_w={aux_w:.3f}")
        print(msg, flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience = 0
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       best_path)
            print(f"  ** new best @ epoch {best_epoch}: μF1={val_f1:.4f}",
                  flush=True)
        else:
            patience += 1
            print(f"  patience {patience}/{config['patience']}  "
                  f"(best={best_f1:.4f} @ {best_epoch})", flush=True)

        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            ckpt_path = os.path.join(save_dir,
                                     f"checkpoint_v7_epoch{epoch+1}.pth")
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     {k: v.cpu().clone()
                                         for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1":          best_f1,
            }, ckpt_path)
            print(f"  [Checkpoint] Saved -> {ckpt_path}", flush=True)

        if patience >= config["patience"] and (epoch + 1) >= config["min_epochs"]:
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"\nLoaded best (epoch {best_epoch}, val μF1={best_f1:.4f})")
    return model, best_f1, best_epoch


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--bert-lr", type=float, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--resume-dir", type=str, default=None,
                   help="Path to a previous experiment_*/ dir to resume from. "
                        "Automatically finds the latest checkpoint_v7_epoch*.pth.")
    p.add_argument(
        "--strict-determinism", action="store_true",
        help="Force cuDNN deterministic kernels for fully reproducible "
             "GPU training (~10-15%% slower).",
    )
    p.add_argument(
        "--force-retrain", action="store_true",
        help="Train even if a completed experiment_*_seed{seed}/ already "
             "exists in save_dir.  Default: skip with a friendly message.",
    )
    args = p.parse_args()

    cfg = dict(CONFIG)
    for k_attr, k_cfg in [
        ("data", "data_file"), ("seed", "seed"),
        ("epochs", "epochs"), ("patience", "patience"),
        ("batch_size", "batch_size"),
        ("bert_lr", "bert_lr"), ("head_lr", "head_lr"),
        ("save_dir", "save_dir"),
    ]:
        v = getattr(args, k_attr)
        if v is not None:
            cfg[k_cfg] = v

    cfg["strict_determinism"] = bool(args.strict_determinism)
    set_seed(cfg["seed"], strict=cfg["strict_determinism"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {cfg['seed']}  "
          f"strict_determinism={cfg['strict_determinism']}")

    # Skip-if-completed: reuse any sibling experiment_*_seed{seed}/ that
    # already has test_metrics.json (the marker written at end of training).
    def _find_completed_experiment(save_root, seed):
        if not os.path.isdir(save_root):
            return None
        suffix = f"_seed{seed}"
        cands = []
        for d in os.listdir(save_root):
            full = os.path.join(save_root, d)
            if (os.path.isdir(full) and d.startswith("experiment_")
                    and d.endswith(suffix)):
                cands.append(full)
        cands.sort(reverse=True)
        for exp in cands:
            if os.path.exists(os.path.join(exp, "test_metrics.json")):
                return exp
        return None

    if not args.force_retrain and not args.resume_dir:
        done = _find_completed_experiment(cfg["save_dir"], cfg["seed"])
        if done:
            print(f"\n[skip] Training already completed at {done}")
            print(f"       Pass --force-retrain to train a fresh model.")
            print(f"       Or delete the dir above to retrain.")
            return

    # Determine experiment directory and resume checkpoint
    resume_ckpt = None
    save_root = cfg["save_dir"]

    if args.resume_dir and os.path.isdir(args.resume_dir):
        exp_dir = args.resume_dir
        _dir_name = os.path.basename(os.path.abspath(args.resume_dir).rstrip("/\\"))
        _m = re.search(r"_seed(\d+)$", _dir_name)
        if _m and int(_m.group(1)) != cfg["seed"]:
            raise SystemExit(
                f"[Error] --resume-dir seed mismatch: "
                f"directory has seed={_m.group(1)} but current seed={cfg['seed']}. "
                f"Pass --seed {_m.group(1)} to resume correctly, or use "
                f"--force-retrain to start fresh with seed={cfg['seed']}."
            )
    else:
        # Auto-reuse the latest experiment_*_seed{N} dir unless forced new
        suffix = f"_seed{cfg['seed']}"
        existing = sorted(
            [os.path.join(save_root, d)
             for d in os.listdir(save_root)
             if d.startswith("experiment_") and d.endswith(suffix)
             and os.path.isdir(os.path.join(save_root, d))]
        ) if os.path.isdir(save_root) else []
        if existing and not args.force_retrain:
            exp_dir = existing[-1]
            print(f"[Resume] Auto-reusing latest experiment dir: {exp_dir}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(save_root,
                                   f"experiment_{ts}_seed{cfg['seed']}")

    os.makedirs(exp_dir, exist_ok=True)

    ckpts = sorted(
        _glob.glob(os.path.join(exp_dir, "checkpoint_v7_epoch*.pth")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if ckpts:
        resume_ckpt = ckpts[-1]
        print(f"[Resume] Found checkpoint -> {resume_ckpt}")
    print(f"Experiment dir: {exp_dir}")

    if not cfg["data_file"]:
        raise ValueError(
            "Set DATA_FILE at the top of train.py (or pass --data) "
            "to point at the training CSV."
        )
    print(f"Data file:    {cfg['data_file']}")
    df = load_processed_dataframe(cfg["data_file"])
    texts = df["DETAILED_DESCRIPTION"].tolist()
    labels, mappings = parse_hierarchical_labels(df)

    counts = count_main_factors(labels)
    print(f"\nMain-factor counts: {counts}")
    print(f"Subcategory counts: warning={len(mappings['warning'])}  "
          f"env={len(mappings['environmental'])}  "
          f"human={len(mappings['human'])}")

    with open(os.path.join(exp_dir, "subcategory_mappings.json"),
              "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    splits = split_dataset_by_subcategory_ratio(
        texts, labels, mappings, ratios=(0.7, 0.15, 0.15),
        random_state=cfg["seed"],
    )
    save_splits_to_csv(splits, df, os.path.join(exp_dir, "data_splits"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_model"], use_fast=True)
    datasets = build_datasets_from_splits(splits, tokenizer, mappings)

    train_labels = [labels[i] for i in splits["train"]["indices"]]
    sampler = create_balanced_sampler(train_labels, counts)
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg["batch_size"],
        sampler=sampler, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False,
    )

    main_mat, w_mat, e_mat, h_mat = _train_label_matrices(
        labels, splits["train"]["indices"], mappings)
    criteria = build_criteria(cfg, main_mat, w_mat, e_mat, h_mat)

    model = TwoStageModelV7(
        cfg["bert_model"], dropout=cfg["dropout"],
        subcategory_mappings=mappings, tokenizer=tokenizer,
        flat_alpha_init=cfg["flat_alpha_init"],
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, best_f1, best_ep = train(
        model, train_loader, val_loader, criteria, cfg, device, exp_dir,
        resume_checkpoint=resume_ckpt,
    )

    test_m, _ = evaluate(model, test_loader, device,
                         s1_thresh=cfg["stage1_threshold"],
                         s2_thresh=cfg["stage2_threshold"])
    print_metrics(test_m, "Test set (default thresholds)")

    with open(os.path.join(exp_dir, "training_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    with open(os.path.join(exp_dir, "test_metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump({
            "best_val_micro_f1": best_f1,
            "best_epoch": best_ep,
            "test_metrics": test_m,
        }, f, indent=2, default=str)
    print(f"\nAll artifacts saved to: {exp_dir}")


if __name__ == "__main__":
    main()
