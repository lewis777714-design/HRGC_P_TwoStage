# HRGC-P: Highway-Railroad Grade Crossing Pedestrian Dataset & TwoStage-SLA Baseline

Code for the paper **"A Fine-grained Multi-label Dataset for Pedestrian Risk Factor Prediction at Highway-Railroad Grade Crossings"** (NeurIPS 2026 submission).

- **Dataset**: https://doi.org/10.34740/kaggle/dsv/16094059
- **Anonymous code**: https://anonymous.4open.science/r/HRGC_P_TwoStage-48F1/

---

## Environment Setup

```bash
pip install -r requirements.txt
```

> **GPU note**: `torch==2.6.0` in `requirements.txt` is the CPU wheel. For CUDA 12.4 (used in the paper on NVIDIA RTX A5000):
> 
> ```bash
> pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
> ```

---

## Data Files

Place both files in the root of `/` before running any script. Download from the Kaggle link above.

| File             | Records | Description                                                 |
| ---------------- | ------- | ----------------------------------------------------------- |
| `HRGC_P_aug.csv` | 6,056   | Augmented benchmark dataset -- used for training all models |
| `HRGC_P_ori.csv` | 3,315   | Original manually-labeled dataset -- used for LLM baselines |

---

## File Structure

```
submission_code/
├── model.py                  # TwoStage-SLA model architecture (Section 4.1, Figure 3)
├── data.py                   # Data loading, label parsing, dataset classes (Section 3)
├── train.py                  # Training entry point for TwoStage-SLA (Section A.2.2)
├── infer.py                  # Test-set inference -- produces Table 1 TwoStage-SLA results
├── baselines.py              # Baselines B1-B9 -- produces Table 1 baseline results
├── baseline_llm_fullrow.py   # LLM zero-shot baselines -- produces Table 2 results
└── requirements.txt
```

---

## Running the Code

### Step 1 -- Train TwoStage-SLA

`train.py` trains the full TwoStage-SLA model (Section 4.1) and produces the checkpoint used by `infer.py`.

Open `train.py` and set the USER CONFIG block at the top:

```python
DATA_FILE  = r"HRGC_P_aug.csv"      # path to augmented dataset
SAVE_DIR   = r"training_results_v7"  # output directory
SEED       = 123                     # paper reports seeds {42, 123, 456, 789, 2024}
EPOCHS     = 500
PATIENCE   = 50
BATCH_SIZE = 128
```

Then run:

```bash
python train.py
```

You can also override settings from the command line without editing the file:

```bash
python train.py --data HRGC_P_aug.csv --seed 42 --epochs 500
```

Each run writes a timestamped experiment folder under `training_results_v7/`:

```
training_results_v7/
└── experiment_20260504_102818_seed123/
    ├── best_model_v7.pth          <- best checkpoint (by validation µF1)
    ├── subcategory_mappings.json  <- label-to-index maps
    ├── training_log.json          <- per-epoch train/val metrics
    └── data_splits/
        ├── train.csv
        ├── val.csv
        └── test.csv               <- held-out test split used by infer.py
```

To reproduce the mean +/- std in Table 1, run with each of the five seeds `{42, 123, 456, 789, 2024}` and average the results.

---

### Step 2 -- Evaluate TwoStage-SLA on the Test Set (Table 1)

`infer.py` loads the checkpoint produced by `train.py` and evaluates it on the held-out `test.csv`. This produces the **TwoStage-SLA** row in Table 1 (overall µF1 = 0.960 +/- 0.002).

Open `infer.py` and set the USER CONFIG block at the top:

```python
EXPERIMENT_DIR   = r"training_results_v7\experiment_20260504_102818_seed123"
STAGE1_THRESHOLD = 0.5
STAGE2_THRESHOLD = 0.95
BATCH_SIZE       = 12
BERT_MODEL       = "bert-base-uncased"
```

Then run:

```bash
python infer.py
```

The script prints Warning / Environmental / Human group µF1 scores and the overall µF1, and writes a detailed results JSON to `inference_results_v7/<experiment_name>/`.

---

### Step 3 -- Baselines B1-B9 (Table 1)

`baselines.py` trains and evaluates all classical and neural baselines on the **same data splits** produced during `train.py`, so the comparison is directly apples-to-apples with TwoStage-SLA.

| ID  | Method                             | Overall µF1 (paper) |
| --- | ---------------------------------- | ------------------- |
| B1  | TF-IDF + Logistic Regression       | 0.840 +/- 0.001     |
| B2  | TF-IDF + Linear SVM                | 0.918 +/- 0.004     |
| B3  | BERT-base CLS (flat)               | 0.936 +/- 0.001     |
| B4  | BERT-base MeanPool (flat)          | 0.931 +/- 0.002     |
| B5  | Two-stage BERT, no label attention | 0.933 +/- 0.007     |
| B6  | DistilBERT-base CLS (flat)         | 0.945 +/- 0.008     |
| B7  | RoBERTa-base CLS (flat)            | 0.936 +/- 0.012     |
| B9  | SBERT cosine zero-shot             | 0.127 +/- 0.000     |

Open `baselines.py` and set the USER CONFIG block at the top:

```python
DATA_FILE  = r"HRGC_P_aug.csv"
SAVE_DIR   = r"baselines_results_v7"
SEED       = 42
EPOCHS     = 500
PATIENCE   = 50
BATCH_SIZE = 128
ONLY       = None    # None = run all baselines
```

Then run:

```bash
# Run all baselines (B1 through B9)
python baselines.py

# Run only B1 and B2 (no GPU required)
python baselines.py --only B1,B2

# Run a single baseline
python baselines.py --only B3

# Quick sanity check with shorter training
python baselines.py --epochs 60 --patience 20
```

Results are saved to `baselines_results_v7/`.

---

### Step 4 -- LLM Zero-Shot Baselines (Table 2)

`baseline_llm_fullrow.py` decodes the full FRA structured form (all coded fields converted to natural language) plus the narrative summary into a single prompt, then calls the LLM to predict subcategory labels with no fine-tuning. This produces the **Table 2** results:

| Model            | Warning µF1 | Environmental µF1 | Human µF1 | Overall µF1 |
| ---------------- | ----------- | ----------------- | --------- | ----------- |
| GPT-4o-mini      | 0.201       | 0.706             | 0.561     | 0.518       |
| Gemini 2.5 Flash | 0.107       | 0.935             | 0.447     | 0.575       |

**Set your API key before running** (Windows PowerShell):

```bash
$env:GEMINI_API_KEY = "your_key_here"   # for Gemini
$env:OPENAI_API_KEY = "your_key_here"   # for OpenAI
```

Open `baseline_llm_fullrow.py` and set the USER CONFIG block at the top:

```python
EXPERIMENT_DIR = r"training_results_v7\experiment_20260504_102818_seed123"
DATA_CSV       = r"HRGC_P_ori.csv"   # original data -- real incidents only, no synthetic
OUT_DIR        = r"baseline_llm_fullrow_results"
PROVIDER       = "gemini"             # "gemini", "openai", or "both"
MODEL          = None                 # None -> provider default (gemini-2.5-flash / gpt-4o-mini)
LIMIT          = None                 # None -> full test split; set e.g. 20 for a smoke test
RESUME         = False
```

Then run:

```bash
# Gemini 2.5 Flash (default)
python baseline_llm_fullrow.py --provider gemini

# GPT-4o-mini
python baseline_llm_fullrow.py --provider openai --model gpt-4o-mini

# Both providers in one pass
python baseline_llm_fullrow.py --provider both

# Smoke test on first 20 rows before paying for the full run
python baseline_llm_fullrow.py --provider gemini --limit 20

# Resume an interrupted run (skips already-saved rows)
python baseline_llm_fullrow.py --provider gemini --resume
```

Results (per-row predictions + F1 scores) are saved to `baseline_llm_fullrow_results/`.

---

## Model Architecture (Section 4.1, Figure 3)

`model.py` is self-contained and implements all components of TwoStage-SLA:

| Component                                                            | Class                    | Paper section |
| -------------------------------------------------------------------- | ------------------------ | ------------- |
| Learned weighted sum of all 13 BERT layers                           | `ScalarMix`              | Section 4.1.1 |
| Multi-granularity pooling [CLS ; MeanPool ; AttnPool]                | `MultiPool`              | Section 4.1.2 |
| Stage-1 primary-factor MLP (3 binary outputs: Warning / Env / Human) | inside `TwoStageModelV7` | Section 4.1.2 |
| Semantic Label Attention with BERT-CLS initialized queries           | `SemanticLabelAttention` | Section 4.1.3 |
| Flat MLP auxiliary head + learned fusion gate sigmoid(alpha)         | `FlatStage2Head`         | Section 4.1.3 |
| Asymmetric Loss with per-class positive reweighting                  | `PerClassAsymmetricLoss` | Section A.2.1 |

---

## Key Hyperparameters (Section A.2.2)

| Parameter                         | Value                                          |
| --------------------------------- | ---------------------------------------------- |
| Backbone                          | `bert-base-uncased` (~110M parameters)         |
| BERT encoder LR                   | 2e-5                                           |
| Task-specific head LR             | 5e-4                                           |
| Layer-wise LR decay factor        | 0.85                                           |
| Batch size / grad accum steps     | 96 / 4 (effective batch 384)                   |
| LR warmup                         | 10% of total steps, then cosine decay to 0     |
| Gradient clip norm                | 1.0                                            |
| Weight decay                      | 0.01                                           |
| Dropout                           | 0.3                                            |
| Max epochs / early-stop patience  | 500 / 50                                       |
| ASL gamma+ / gamma- / margin m    | 1 / 3 / 0.05                                   |
| Positive reweight power / ceiling | 0.5 / 8.0                                      |
| Stage-1 aux loss lambda_aux       | cosine anneal 0.5 -> 0.05 over first 30 epochs |
| Flat-head aux weight lambda_flat  | 0.2 (fixed)                                    |
| Data split                        | 70 / 15 / 15 stratified by subcategory         |
| Evaluation seeds                  | {42, 123, 456, 789, 2024}                      |
| Hardware                          | NVIDIA RTX A5000, 24 GB VRAM                   |
