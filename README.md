# Submission code — HRGC-P model

Self-contained implementation of the **HRGC-P model** and all reported
baselines / ablations. Every entry-point script depends only on `data.py`
and `model.py` inside this folder; nothing is imported from outside
`submission_code/`.

## Data

Two CSV files are bundled in this folder:

| file | role |
|------|------|
| `HRGC_P_ori.csv` | Original Highway-Rail Grade Crossing (Pedestrian) records |
| `HRGC_P_aug.csv` | Augmented version used for training (original + synthetic samples) |

The training / ablation / baseline scripts default to `HRGC_P_aug.csv`;
`baseline_llm.py` joins the LLM zero-shot test split back to
`HRGC_P_ori.csv` to recover the `NARR_SUMMARY` one-line summaries.

> **Training-set download.** The full augmented training file is too
> large to ship inside `submission_code/`. Download the training CSV
> from the code link in the paper and place it in **this same folder**
> (alongside `train.py`) before running any training script. The
> filename must match the `--data` argument (default `HRGC_P_aug.csv`).

## Files

| file | role |
|------|------|
| `data.py` | label parsing, `AccidentDataset`, iterative-stratification split, balanced sampler |
| `model.py` | `TwoStageModelV7` + helper modules (`ScalarMix`, `AttentionPooling`, `MultiPoolStage1`, `SemanticLabelAttention`, `FlatStage2Head`), `PerClassAsymmetricLoss`, `compute_pos_class_weights`, layer-wise LR groups |
| `train.py` | V7 training entry point (joint end-to-end, cosine-decayed aux Stage-1 weight, flat-head fusion) |
| `infer.py` | V7 inference at **default scalar thresholds** (S1=0.5, S2=0.5). No per-label tuning |
| `threshold_search.py` | per-label threshold sweep on the validation split |
| `ablation.py` | trains one model per V7 ablation (A1–A8) on the same data splits |
| `baselines.py` | nine honest baselines: B1 TF-IDF+LR · B2 TF-IDF+SVM · B3 BERT-CLS-flat · B4 BERT-MeanPool-flat · B5 Two-stage-mean · B6 DistilBERT-flat · B7 RoBERTa-flat · B8 DeBERTa-v3-flat · B9 Sentence-BERT zero-shot |
| `baseline_llm.py` | B10 LLM zero-shot (Google Gemini / OpenAI GPT) on **NARR_SUMMARY** one-line summaries |

## Requirements

```bash
pip install -r requirements.txt
```

| package | min version | used by |
|---------|-------------|---------|
| torch | 2.0 | every script |
| transformers | 4.30 | every script except baselines B1/B2 and B10 |
| numpy | 1.24 | every script |
| pandas | 1.5 | every script |
| scikit-learn | 1.0 | `baselines.py` (B1 LogReg, B2 LinearSVC) only |
| sentencepiece | 0.1.99 | `baselines.py` B7/B8 (RoBERTa / DeBERTa-v3 tokenizers) |
| protobuf | 3.20 | required by transformers when loading the SentencePiece tokenizer for DeBERTa-v3 |
| google-genai | 0.3 | `baseline_llm.py` B10 (`--provider gemini`, default) |
| openai | 1.0 | `baseline_llm.py` B10 (`--provider openai`) |

The last two rows are *optional* — install whichever provider you plan
to run for B10 (or both, to compare). Core training (V7 + B1-B9) only
needs the rows above them.

Tested with Python 3.11 on CUDA 12.4. A CUDA GPU with ≥ 16 GB is recommended
for `train.py` and the BERT baselines (B3 / B4 / B5). `gradient_checkpointing`
is on by default so that batch-size 96 at sequence length 512 fits comfortably.

## CSV column requirements

Both `HRGC_P_aug.csv` and `HRGC_P_ori.csv` must contain the columns:

- `DETAILED_DESCRIPTION` — input text
- `WARNING_DEVICE_ISSUES`, `ENVIRONMENTAL_FACTORS`, `HUMAN_FACTORS` —
  comma-separated subcategory names; the literal `Unidentified` is
  treated as no label

`HRGC_P_ori.csv` additionally needs a `NARR_SUMMARY` column for the B10
LLM baseline (one-line summaries that get joined back to the test split).

The label index space is built from the order of first appearance in the
CSV (`parse_hierarchical_labels`). When loading saved splits later,
`parse_labels_with_mapping` reuses the saved `subcategory_mappings.json`
so the index space matches across runs.

## Quick-start

> Make sure `HRGC_P_aug.csv` (training) and `HRGC_P_ori.csv` (B10 LLM
> baseline) are present in this folder. If they aren't, download them
> from the code link in the paper first.

```bash
# 1. Train HRGC-P
python train.py --data HRGC_P_aug.csv --seed 42

# 2. Evaluate on the test split (default scalar thresholds)
python infer.py --experiment-dir training_results_v7/experiment_XXXXXX

# 3. (Optional) Per-label threshold search on validation
python threshold_search.py --experiment-dir training_results_v7/experiment_XXXXXX

# 4. Reproduce the ablation table (A1 … A8)
python ablation.py --data HRGC_P_aug.csv --seed 42

# 5. Reproduce the baselines table (B1 … B9)
python baselines.py --data HRGC_P_aug.csv --seed 42

# 6. (Optional) LLM zero-shot baseline B10 — uses NARR_SUMMARY one-liners
#    Costs API credits.  Smoke-test with --limit 20 first.

#    Default: Google Gemini
#      PowerShell:  $env:GEMINI_API_KEY = "..."
#      bash / zsh:  export GEMINI_API_KEY=...
python baseline_llm.py \
    --experiment-dir training_results_v7/experiment_XXXXXX_seed42 \
    --narr-summary-csv HRGC_P_ori.csv \
    --limit 20

#    OpenAI for comparison
#      PowerShell:  $env:OPENAI_API_KEY = "..."
#      bash / zsh:  export OPENAI_API_KEY=...
python baseline_llm.py --provider openai --model gpt-4o-mini \
    --experiment-dir training_results_v7/experiment_XXXXXX_seed42 \
    --narr-summary-csv HRGC_P_ori.csv \
    --limit 20
```

## Baseline catalogue

| ID | Backbone / method | Input | Trained? |
|----|---|---|---|
| B1 | TF-IDF + LogReg (one-vs-rest) | DETAILED_DESCRIPTION | yes (sklearn) |
| B2 | TF-IDF + LinearSVC (one-vs-rest) | DETAILED_DESCRIPTION | yes (sklearn) |
| B3 | BERT-base-uncased + CLS pool + flat head | DETAILED_DESCRIPTION | yes (V7-equivalent budget) |
| B4 | BERT-base-uncased + MeanPool + flat head | DETAILED_DESCRIPTION | yes |
| B5 | Two-stage BERT (mean-pool, no label attention) | DETAILED_DESCRIPTION | yes |
| B6 | DistilBERT-base-uncased + CLS + flat | DETAILED_DESCRIPTION | yes (no LLRD; see note) |
| B7 | RoBERTa-base + CLS + flat | DETAILED_DESCRIPTION | yes (no LLRD) |
| B8 | DeBERTa-v3-base + CLS + flat | DETAILED_DESCRIPTION | yes (no LLRD) |
| B9 | SBERT (mpnet-base-v2) cosine to label-name | DETAILED_DESCRIPTION | **zero-shot**, only 3 thresholds tuned on val |
| B10 | Google Gemini *or* OpenAI GPT (JSON mode, system instruction) | **NARR_SUMMARY one-liner** | zero-shot, no training data |

B6/B7/B8 use simple two-group parameter splits (no LLRD) so that
DistilBERT, RoBERTa and DeBERTa-v3 train under exactly the same
optimiser strategy — only the encoder differs. B3/B4 keep V7's LLRD
because they share its BERT-base backbone.

B10 supports two providers selected with `--provider`:

- `--provider gemini` (default) → `gemini-2.5-flash`. Reads `GEMINI_API_KEY`
  (falls back to `GOOGLE_API_KEY`). Cost ≈ \$0.001/row → ~\$0.50 / 500 rows.
- `--provider openai` → `gpt-4o-mini`. Reads `OPENAI_API_KEY`. Cost
  ≈ \$0.0007/row → ~\$0.30 / 500 rows.

Both providers are asked for strict JSON (Gemini via
`response_mime_type="application/json"`, OpenAI via `response_format=
{"type": "json_object"}`), and the long subcategory list is sent as
the system instruction / system message so the provider's KV cache
amortises it across rows. Pass `--model` to override the default
model name (e.g. `gemini-2.5-pro`, `gpt-4o`).

For shorter ablation / baseline experiments, pass `--epochs 60 --patience 20`.
All scripts share the V7 hyper-parameters defined in `train.CONFIG`.

## Reproducibility

Every entry-point script accepts `--seed <int>` (default 42).  The seed
controls Python's `random`, `numpy`, `torch`, and `torch.cuda` RNGs via
`train.set_seed`.  All scripts persist the seed to disk:

- `train.py` writes `experiment_<timestamp>_seed<seed>/training_config.json`
- `ablation.py` writes `run_<timestamp>_seed<seed>/run_config.json`
- `baselines.py` writes `run_<timestamp>_seed<seed>/run_config.json`
- `infer.py` and `threshold_search.py` re-seed at startup for parity
  (their forward pass is deterministic regardless)

Both `ablation.py` and `baselines.py` re-seed **before each ablation /
baseline** so the comparisons are mutually reproducible — running A1
through A8 in sequence gives the same numbers as running just A5 alone.

For bit-exact GPU reproducibility, pass `train.py --strict-determinism`.
That forces cuDNN to its deterministic kernels and disables autotuning,
costing ~10-15% throughput.  Off by default to match V6.

## V7 design (one-paragraph summary)

`TwoStageModelV7` is a hierarchical multi-label classifier built on a
trainable BERT backbone. A learned ScalarMix combines the 13 BERT layer
states; a `MultiPoolStage1` pool ([CLS ; MeanPool ; AttentionPool] →
projection) feeds the 3-way Stage-1 main-factor head. For each Stage-2
category (warning / environmental / human), the model produces two
parallel logit streams: a `SemanticLabelAttention` head whose label
queries are initialised from BERT-encoded subcategory names and
conditioned on Stage-1 logits, and a small `FlatStage2Head` MLP from the
mean-pooled document representation. The two streams are blended with a
per-category learnable scalar `α ∈ (0, 1)`:

    final_logits = σ(α) · attn_logits + (1 − σ(α)) · flat_logits

The training objective is the V7 loss

    L = L_S2_mixed + flat_aux_w · L_S2_flat + aux_w(t) · L_S1

where every term is `PerClassAsymmetricLoss` with per-class positive
reweighting capped at `w_max = 8`, and `aux_w(t)` cosine-decays from
`0.5` to `0.05` over the first 30 epochs (then stays flat).
EMA is **off** by default (the V6→V7 ablation showed it hurts).

## Notes on the baselines

`baselines.py` trains the BERT baselines (B3/B4/B5) under the **same**
budget, optimiser, LLRD, batch size, and loss schedule as V7 — they get
the same `PerClassAsymmetricLoss`, the same balanced sampler, the same
LR warmup, and `gradient_checkpointing`. Only the architecture differs.
This makes the V7-vs-baseline comparison apples-to-apples.

If a baseline beats V7 on some metric, that's a real result; the script
does not adjust hyper-parameters to suppress baseline performance.
