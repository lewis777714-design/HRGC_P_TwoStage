

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data import load_processed_dataframe, parse_labels_with_mapping


# ═════════════════════════════════════════════════════════════════
# USER CONFIG
# ═════════════════════════════════════════════════════════════════

EXPERIMENT_DIR  = r"training_results_v7\experiment_20260430_214316_seed42"
DATA_CSV        = r"HRGC_P_ori.csv"
OUT_DIR         = r"baseline_llm_fullrow_results"

PROVIDER        = "gemini"
MODEL           = None        # None → provider default
LIMIT           = None
THROTTLE_MS     = 0
RESUME          = False


# ─────────────────────────────────────────────────────────────────
# Coded-value decoders (FRA / HRGC standard codes)
# ─────────────────────────────────────────────────────────────────

_STATE = {
    1:"Alabama",2:"Alaska",4:"Arizona",5:"Arkansas",6:"California",8:"Colorado",
    9:"Connecticut",10:"Delaware",11:"DC",12:"Florida",13:"Georgia",15:"Hawaii",
    16:"Idaho",17:"Illinois",18:"Indiana",19:"Iowa",20:"Kansas",21:"Kentucky",
    22:"Louisiana",23:"Maine",24:"Maryland",25:"Massachusetts",26:"Michigan",
    27:"Minnesota",28:"Mississippi",29:"Missouri",30:"Montana",31:"Nebraska",
    32:"Nevada",33:"New Hampshire",34:"New Jersey",35:"New Mexico",36:"New York",
    37:"North Carolina",38:"North Dakota",39:"Ohio",40:"Oklahoma",41:"Oregon",
    42:"Pennsylvania",44:"Rhode Island",45:"South Carolina",46:"South Dakota",
    47:"Tennessee",48:"Texas",49:"Utah",50:"Vermont",51:"Virginia",53:"Washington",
    54:"West Virginia",55:"Wisconsin",56:"Wyoming",72:"Puerto Rico",
}

_WEATHER = {1:"Clear",2:"Cloudy",3:"Rain",4:"Fog",5:"Sleet/Hail",6:"Snow",7:"Other"}
_VISIBLTY = {1:"Dawn",2:"Daylight",3:"Dusk",4:"Dark (lighted)",
             5:"Dark (unlighted)",6:"Dark (lighting unknown)"}
_TYPVEH = {
    "A":"Automobile","B":"Bus","E":"Other","K":"Truck (Semi-trailer)",
    "L":"Light Truck/Van","M":"Motorcycle","P":"Pickup Truck","S":"School Bus",
    "T":"Truck (other)","X":"Not Reported",
}
_TRNDIR = {1:"North",2:"South",3:"East",4:"West"}
_VEHDIR = {1:"North",2:"South",3:"East",4:"West"}
_POSITION = {
    1:"Stalled on crossing",2:"Trapped on crossing",3:"Went around/through gate",
    4:"Stopped on crossing",5:"Moving over crossing",6:"Other",
}
_TYPEQ = {
    1:"Freight",2:"Passenger",3:"Commuter",4:"Work/Maintenance",
    5:"Single car",6:"Light loco(s)",7:"Cut of cars",8:"Yard/Transfer",
    9:"Other",
}
_SIGNAL = {1:"Automatic",2:"Manual block",3:"None"}
_WARNSIG = {
    1:"None",2:"Wigwag",3:"Gates",4:"Flagman",5:"Flashing lights only",
    6:"Flashing lights + gates",7:"Other active devices",8:"Stop signs",
    9:"Crossbucks only",10:"Other passive",
}
_LIGHTS = {1:"Yes",2:"No",3:"Unknown"}
_WHISBAN = {1:"Yes, ban in effect",2:"No ban"}
_ROADCOND_MAP = {"1":"Dry","2":"Wet","3":"Snow/Slush","4":"Ice","5":"Other"}
_LOCWARN = {
    1:"At crossing",2:"Within 1/4 mile",3:"Within 1/2 mile",
    4:"Within 1 mile",5:"More than 1 mile away",
}


def _dec(mapping, val, default=None):
    """Decode a coded value; return default (or the raw val) if not found."""
    if pd.isna(val):
        return default or "Unknown"
    try:
        key = int(float(val))
    except (ValueError, TypeError):
        key = str(val).strip()
    return mapping.get(key, str(val))


def _fmt(label, val, unit=""):
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN"):
        return None
    return f"{label}: {val}{unit}"


# ─────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────

# Columns that encode the labels we're predicting — MUST be excluded.
_LABEL_COLS = {
    "HUMAN_FACTORS", "HUMAN_FACTORS_EVIDENCE",
    "ENVIRONMENTAL_FACTORS", "ENVIRONMENTAL_FACTORS_EVIDENCE",
    "WARNING_DEVICE_ISSUES", "WARNING_DEVICE_ISSUES_EVIDENCE",
    "PRIMARY_FACTORS", "RECOMMENDATIONS",
    "DETAILED_DESCRIPTION",   # GPT-generated from labels; leaks signal
}


def build_row_text(row: pd.Series) -> str:
    """Convert one CSV row into a human-readable accident report for the LLM.
    All label-derived columns are excluded. Coded values are decoded where
    possible so the LLM doesn't have to guess what code '3' means."""

    parts: List[str] = ["=== RAILWAY CROSSING ACCIDENT REPORT ===\n"]

    # ── Time / Location ──
    loc_parts = []
    year  = row.get("YEAR4") or row.get("YEAR")
    month = row.get("MONTH")
    day   = row.get("DAY")
    hr    = row.get("TIMEHR")
    mn    = row.get("TIMEMIN")
    ampm  = row.get("AMPM", "")
    try:
        date_str = f"{int(float(month)):02d}/{int(float(day)):02d}/{int(float(year))}"
        time_str = (f"{int(float(hr)):02d}:{int(float(mn)):02d} {ampm}".strip()
                    if not pd.isna(hr) else "")
        loc_parts.append(f"Date/Time: {date_str} {time_str}".strip())
    except (ValueError, TypeError):
        pass
    state_code = row.get("STATE")
    state_name = _dec(_STATE, state_code, "Unknown")
    city  = row.get("CITY",  "")
    county = row.get("COUNTY", "")
    hwy   = row.get("HIGHWAY", "")
    if not pd.isna(city):   loc_parts.append(f"City: {city}")
    if not pd.isna(county): loc_parts.append(f"County: {county}")
    loc_parts.append(f"State: {state_name}")
    if not pd.isna(hwy):    loc_parts.append(f"Highway/Road: {hwy}")
    gxid = row.get("GXID")
    if not pd.isna(gxid):   loc_parts.append(f"Crossing ID: {gxid}")
    parts.append("[LOCATION]\n" + "\n".join(loc_parts))

    # ── Environment ──
    env_parts = []
    temp = row.get("TEMP")
    if not pd.isna(temp):
        env_parts.append(f"Temperature: {temp}°F")
    env_parts.append(f"Visibility: {_dec(_VISIBLTY, row.get('VISIBLTY'))}")
    env_parts.append(f"Weather: {_dec(_WEATHER, row.get('WEATHER'))}")
    roadcond = row.get("ROADCOND")
    if not pd.isna(roadcond):
        env_parts.append(f"Road Condition: {_ROADCOND_MAP.get(str(roadcond).strip(), str(roadcond))}")
    view = row.get("VIEW")
    view_map = {
        1:"No obstruction",2:"Obstructed by building",3:"Obstructed by vegetation",
        4:"Obstructed by terrain",5:"Obstructed by train/equipment",
        6:"Sun glare",7:"Obstructed by other",8:"Unknown",
    }
    if not pd.isna(view):
        env_parts.append(f"Driver View: {_dec(view_map, view)}")
    parts.append("[ENVIRONMENTAL CONDITIONS]\n" + "\n".join(env_parts))

    # ── Train ──
    tr_parts = []
    trnspd = row.get("TRNSPD")
    typspd = row.get("TYPSPD", "")
    speed_type = {"R":"Recorded","E":"Estimated","U":"Unknown"}.get(
        str(typspd).strip(), str(typspd))
    if not pd.isna(trnspd):
        tr_parts.append(f"Train Speed: {trnspd} mph ({speed_type})")
    tr_parts.append(f"Train Direction: {_dec(_TRNDIR, row.get('TRNDIR'))}")
    tr_parts.append(f"Train Type: {_dec(_TYPEQ, row.get('TYPEQ'))}")
    typtrk = row.get("TYPTRK")
    typtrk_map = {1:"Single main track",2:"Multiple main track",3:"Siding/Industry track",
                  4:"Yard track",5:"Other"}
    if not pd.isna(typtrk):
        tr_parts.append(f"Track Type: {_dec(typtrk_map, typtrk)}")
    trkname = row.get("TRKNAME")
    if not pd.isna(trkname):
        tr_parts.append(f"Track Name: {trkname}")
    trkclas = row.get("TRKCLAS")
    try:
        tr_parts.append(f"Track Class: {int(float(trkclas))}")
    except (ValueError, TypeError):
        pass
    nbl = row.get("NBRLOCOS"); nbc = row.get("NBRCARS")
    try:
        tr_parts.append(f"Locomotives: {int(float(nbl))}")
    except (ValueError, TypeError):
        pass
    try:
        tr_parts.append(f"Cars: {int(float(nbc))}")
    except (ValueError, TypeError):
        pass
    rrequip_map = {1:"Train",2:"Light engine(s)",3:"Car(s) only",4:"Other"}
    tr_parts.append(f"Equipment Involved: {_dec(rrequip_map, row.get('RREQUIP'))}")
    parts.append("[TRAIN INFORMATION]\n" + "\n".join(tr_parts))

    # ── Vehicle ──
    veh_parts = []
    typveh = row.get("TYPVEH")
    if not pd.isna(typveh):
        veh_parts.append(f"Vehicle Type: {_dec(_TYPVEH, str(typveh).strip())}")
    vehspd = row.get("VEHSPD")
    if not pd.isna(vehspd):
        veh_parts.append(f"Vehicle Speed: {vehspd} mph")
    veh_parts.append(f"Vehicle Direction: {_dec(_VEHDIR, row.get('VEHDIR'))}")
    veh_parts.append(f"Vehicle Position: {_dec(_POSITION, row.get('POSITION'))}")
    motorist = row.get("MOTORIST")
    if not pd.isna(motorist):
        veh_parts.append(f"Motorist: {motorist}")
    drivage = row.get("DRIVAGE"); drivgen = row.get("DRIVGEN")
    try:
        age_int = int(float(drivage))
        veh_parts.append(f"Driver Age: {age_int}, Gender: {_dec({1:'Male',2:'Female'}, drivgen)}")
    except (ValueError, TypeError):
        pass
    inveh = row.get("INVEH")
    try:
        veh_parts.append(f"Occupants in Vehicle: {int(float(inveh))}")
    except (ValueError, TypeError):
        pass
    standveh = row.get("STANDVEH")
    if not pd.isna(standveh):
        veh_parts.append(f"Vehicle Standing on Track: Yes")
    parts.append("[VEHICLE INFORMATION]\n" + "\n".join(veh_parts))

    # ── Warning devices ──
    warn_parts = []
    warn_parts.append(f"Signal Type: {_dec(_SIGNAL, row.get('SIGNAL'))}")
    warn_parts.append(f"Warning Signal(s): {_dec(_WARNSIG, row.get('WARNSIG'))}")
    warn_parts.append(f"Lights Functioning: {_dec(_LIGHTS, row.get('LIGHTS'))}")
    sigwarnx = row.get("SIGWARNX")
    if not pd.isna(sigwarnx):
        warn_parts.append(f"Crossing Sign/Warning Extra: {sigwarnx}")
    whisban = row.get("WHISBAN")
    if not pd.isna(whisban):
        warn_parts.append(f"Whistle Ban: {_dec(_WHISBAN, whisban)}")
    locwarn = row.get("LOCWARN")
    if not pd.isna(locwarn):
        warn_parts.append(f"Location of Warning: {_dec(_LOCWARN, locwarn)}")
    train2 = row.get("TRAIN2")
    if not pd.isna(train2):
        warn_parts.append(f"Second Train Present: Yes")
    parts.append("[WARNING DEVICES]\n" + "\n".join(warn_parts))

    # ── Casualties ──
    cas_parts = []
    totkld = row.get("TOTKLD"); totinj = row.get("TOTINJ")
    try:
        cas_parts.append(f"Total Killed: {int(float(totkld))}")
    except (ValueError, TypeError):
        pass
    try:
        cas_parts.append(f"Total Injured: {int(float(totinj))}")
    except (ValueError, TypeError):
        pass
    totocc = row.get("TOTOCC")
    try:
        cas_parts.append(f"Total Occupants: {int(float(totocc))}")
    except (ValueError, TypeError):
        pass
    userkld = row.get("USERKLD"); userinj = row.get("USERINJ")
    try:
        cas_parts.append(f"Highway Users Killed: {int(float(userkld))}")
    except (ValueError, TypeError):
        pass
    try:
        cas_parts.append(f"Highway Users Injured: {int(float(userinj))}")
    except (ValueError, TypeError):
        pass
    haz = row.get("HAZARD")
    hazard_map = {1:"No hazmat",2:"Hazmat released",3:"Hazmat not released",
                  4:"Unknown if hazmat",5:"No hazmat on train"}
    if not pd.isna(haz):
        cas_parts.append(f"Hazmat Status: {_dec(hazard_map, haz)}")
    if cas_parts:
        parts.append("[CASUALTIES & HAZMAT]\n" + "\n".join(cas_parts))

    # ── Narrative ──
    narr_parts = []
    narr_summary = row.get("NARR_SUMMARY")
    if not pd.isna(narr_summary) and str(narr_summary).strip():
        narr_parts.append(f"Summary: {narr_summary.strip()}")
    # Concatenate NARR1-5 (raw FRA narrative fields)
    raw_narr = " ".join(
        str(row.get(f"NARR{i}", "") or "").strip()
        for i in range(1, 6)
        if not pd.isna(row.get(f"NARR{i}"))
    ).strip()
    if raw_narr:
        narr_parts.append(f"Detailed Narrative: {raw_narr}")
    if narr_parts:
        parts.append("[NARRATIVE]\n" + "\n".join(narr_parts))

    return "\n\n".join(parts)


CATEGORY_LABEL_KEYS = {
    "warning":       "WARNING_DEVICE_ISSUES",
    "environmental": "ENVIRONMENTAL_FACTORS",
    "human":         "HUMAN_FACTORS",
}


def build_system_prompt(mappings: Dict[str, Dict[str, int]]) -> str:
    def _names(cat):
        return [n for n, _ in sorted(mappings[cat].items(), key=lambda x: x[1])]

    blocks = []
    for cat in ("human", "environmental", "warning"):
        head = CATEGORY_LABEL_KEYS[cat]
        bullets = "\n".join(f"  - {n}" for n in _names(cat))
        blocks.append(f"[{head}]\n{bullets}")

    return (
        "You classify a railway-crossing accident report into THREE "
        "multi-label factor categories: HUMAN_FACTORS, "
        "ENVIRONMENTAL_FACTORS, WARNING_DEVICE_ISSUES.\n\n"
        "Use ALL available information in the report (structured data, "
        "conditions, warning device status, narrative) to determine "
        "which factors contributed to the accident.\n\n"
        "For each category, output ALL applicable subcategories "
        "(zero, one, or many). If no subcategory applies, output "
        "an empty list.\n\n"
        "You MUST choose subcategory names verbatim from the lists "
        "below — do not invent, paraphrase, or truncate.\n\n"
        + "\n\n".join(blocks)
        + "\n\nOUTPUT FORMAT — strict JSON, no surrounding prose:\n"
        + '{"HUMAN_FACTORS": ["..."], "ENVIRONMENTAL_FACTORS": ["..."], '
          '"WARNING_DEVICE_ISSUES": ["..."]}\n'
    )


def build_user_prompt(row_text: str) -> str:
    return f"{row_text}\n\nReturn the JSON classification."


# ─────────────────────────────────────────────────────────────────
# LLM providers
# ─────────────────────────────────────────────────────────────────

def _gemini_client():
    try:
        from google import genai  # noqa
    except ImportError:
        sys.exit("pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY or GOOGLE_API_KEY env var.")
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini(client, model: str, system_prompt: str, user_prompt: str,
                max_tokens: int = 2048) -> str:
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text or ""


def _openai_client():
    try:
        import openai  # noqa
    except ImportError:
        sys.exit("pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY env var.")
    import openai
    return openai.OpenAI(api_key=api_key)


def call_openai(client, model: str, system_prompt: str, user_prompt: str,
                max_tokens: int = 2048) -> str:
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────

def _parse_json_payload(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    start = txt.find("{")
    if start < 0:
        return None
    depth, end = 0, -1
    for i in range(start, len(txt)):
        if txt[i] == "{":   depth += 1
        elif txt[i] == "}": depth -= 1
        if depth == 0:      end = i + 1; break
    if end < 0:
        return None
    blob = txt[start:end]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(blob2)
        except json.JSONDecodeError:
            return None


def _names_to_indices(names: List[str], cat_mapping: Dict[str, int],
                      unknown_log: List[str]) -> List[int]:
    out = []
    if not isinstance(names, list):
        return out
    name_set = {k.lower(): v for k, v in cat_mapping.items()}
    for n in names:
        if not isinstance(n, str):
            continue
        key = n.strip().lower()
        if key in name_set:
            out.append(name_set[key])
        else:
            unknown_log.append(n)
    return out


def llm_response_to_y(payload: Optional[Dict],
                      mappings: Dict[str, Dict[str, int]],
                      unknown_log: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    out = {}
    for cat in ("warning", "environmental", "human"):
        K = len(mappings[cat])
        v = np.zeros(K, dtype=np.int64)
        if payload:
            json_key = CATEGORY_LABEL_KEYS[cat]
            indices = _names_to_indices(payload.get(json_key, []),
                                        mappings[cat], unknown_log[cat])
            for k in indices:
                if 0 <= k < K:
                    v[k] = 1
        out[cat] = v
    return out


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

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
    return {"micro_f1": micro, "macro_f1": macro,
            "precision_micro": p, "recall_micro": r}


# ─────────────────────────────────────────────────────────────────
# Comparison printer
# ─────────────────────────────────────────────────────────────────

def _load_v7_metrics(exp_dir: str) -> Optional[Dict]:
    """Try to load V7 test metrics from the experiment dir."""
    for fname in ("test_metrics.json", "test_results.json", "metrics.json"):
        p = os.path.join(exp_dir, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def print_comparison(b11_summary: Dict, b10_path: Optional[str],
                     v7_exp_dir: Optional[str]):
    rows = []

    def _extract(summary, label):
        ov = summary.get("overall") or summary.get("stage2", {}).get("overall", {})
        w  = (summary.get("stage2") or {}).get("warning", {})
        e  = (summary.get("stage2") or {}).get("environmental", {})
        h  = (summary.get("stage2") or {}).get("human", {})
        rows.append((label,
                     ov.get("micro_f1", float("nan")),
                     ov.get("macro_f1", float("nan")),
                     w.get("micro_f1", float("nan")),
                     e.get("micro_f1", float("nan")),
                     h.get("micro_f1", float("nan"))))

    _extract(b11_summary, "B11 (full-row, LLM zero-shot)")

    if b10_path and os.path.exists(b10_path):
        with open(b10_path, encoding="utf-8") as f:
            b10 = json.load(f)
        _extract(b10, "B10 (narr-only, LLM zero-shot)")

    if v7_exp_dir:
        v7m = _load_v7_metrics(v7_exp_dir)
        if v7m:
            _extract(v7m, "V7  (fine-tuned model)")

    print(f"\n{'='*100}")
    print("  COMPARISON: B11 (full-row) vs B10 (narr-only) vs V7 (fine-tuned)")
    print(f"{'='*100}")
    print(f"  {'Model':<38}  {'OverallμF1':>10}  {'OverallmF1':>10}  "
          f"{'Warn μF1':>8}  {'Env μF1':>8}  {'Hum μF1':>8}")
    print("  " + "-" * 94)
    for name, ov_micro, ov_macro, w, e, h in rows:
        def _fmt_f(v):
            return f"{v:.4f}" if not (v != v) else "  N/A  "
        print(f"  {name:<38}  {_fmt_f(ov_micro):>10}  {_fmt_f(ov_macro):>10}  "
              f"{_fmt_f(w):>8}  {_fmt_f(e):>8}  {_fmt_f(h):>8}")
    print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="B11: LLM zero-shot baseline using full CSV row data.")
    p.add_argument("--experiment-dir",  type=str, default=None)
    p.add_argument("--data-csv",        type=str, default=None,
                   help="Path to HRGC_P_ori.csv (full row data).")
    p.add_argument("--out-dir",         type=str, default=None)
    p.add_argument("--provider",        choices=("gemini", "openai", "both"), default=None,
                   help="LLM provider. 'both' runs Gemini then OpenAI and compares.")
    p.add_argument("--model",           type=str, default=None)
    p.add_argument("--limit",           type=int, default=None)
    p.add_argument("--throttle-ms",     type=int, default=None)
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--compare-only",    action="store_true",
                   help="Skip API calls; just print comparison table.")
    p.add_argument("--b10-summary",     type=str, default=None,
                   help="Path to B10 test_metrics_summary.json for comparison.")
    args = p.parse_args()

    exp_dir      = (args.experiment_dir or EXPERIMENT_DIR).strip()
    data_csv     = (args.data_csv       or DATA_CSV      ).strip()
    out_dir_eff  =  args.out_dir        or OUT_DIR
    provider_eff =  args.provider       or PROVIDER
    model_eff    =  args.model          if args.model       is not None else MODEL
    limit_eff    =  args.limit          if args.limit       is not None else LIMIT
    throttle_eff =  args.throttle_ms    if args.throttle_ms is not None else THROTTLE_MS
    resume_eff   =  args.resume or RESUME

    summary_path = os.path.join(out_dir_eff, "test_metrics_summary.json")

    # ── Compare-only mode ──
    if args.compare_only:
        if not os.path.exists(summary_path):
            sys.exit(f"B11 summary not found at {summary_path}. Run B11 first.")
        with open(summary_path, encoding="utf-8") as f:
            b11_sum = json.load(f)
        print_comparison(b11_sum, args.b10_summary, exp_dir if exp_dir else None)
        return

    # ── Validate paths ──
    if not exp_dir:
        sys.exit("Set EXPERIMENT_DIR or pass --experiment-dir.")
    if not data_csv:
        sys.exit("Set DATA_CSV or pass --data-csv.")
    if not os.path.isdir(exp_dir):
        sys.exit(f"Not a directory: {exp_dir}")

    mappings_path = os.path.join(exp_dir, "subcategory_mappings.json")
    for path in (mappings_path, data_csv):
        if not os.path.exists(path):
            sys.exit(f"Required file missing: {path}")

    # ── Load mappings + full original CSV directly ──
    with open(mappings_path, encoding="utf-8") as f:
        mappings = json.load(f)

    real = load_processed_dataframe(data_csv)
    real = real[real["DETAILED_DESCRIPTION"].notna()].reset_index(drop=True)
    print(f"Loaded {len(real)} rows from {os.path.basename(data_csv)}")

    if limit_eff is not None:
        real = real.head(limit_eff).copy()
        print(f"  limit={limit_eff} → {len(real)} rows")

    real = real.reset_index(drop=True)

    # ── Gold labels ──
    gold_labels = parse_labels_with_mapping(real, mappings)

    system_prompt = build_system_prompt(mappings)
    print(f"System prompt length: {len(system_prompt)} chars")

    print("Building full-row prompts for each test sample...")
    row_texts = [build_row_text(real.iloc[i]) for i in range(len(real))]
    avg_len = int(np.mean([len(t) for t in row_texts]))
    print(f"Average prompt length: {avg_len} chars (vs ~150 for B10 NARR_SUMMARY)\n")

    # ── Decide which providers to run ──
    providers_to_run: List[str] = []
    if provider_eff == "both":
        providers_to_run = ["gemini", "openai"]
    else:
        providers_to_run = [provider_eff]

    all_summaries: Dict[str, Dict] = {}

    for prov in providers_to_run:
        print(f"\n{'#'*60}\n#  Provider: {prov}\n{'#'*60}")

        if prov == "gemini":
            client     = _gemini_client()
            model_name = model_eff or "gemini-2.5-flash"
            call_fn    = lambda sp, up, c=client, m=model_name: call_gemini(c, m, sp, up)
        elif prov == "openai":
            client     = _openai_client()
            model_name = model_eff or "gpt-4o-mini"
            call_fn    = lambda sp, up, c=client, m=model_name: call_openai(c, m, sp, up)
        else:
            print(f"[skip] unknown provider: {prov}")
            continue

        # Each provider gets its own subdir so results don't overwrite each other
        prov_dir = os.path.join(out_dir_eff, prov)
        os.makedirs(prov_dir, exist_ok=True)
        raw_path     = os.path.join(prov_dir, "raw_predictions.jsonl")
        prov_summary = os.path.join(prov_dir, "test_metrics_summary.json")

        seen: set = set()
        saved_preds: Dict[int, Optional[Dict]] = {}
        if resume_eff and os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        idx = r["row_index"]
                        seen.add(idx)
                        saved_preds[idx] = r.get("payload")
                    except (json.JSONDecodeError, KeyError):
                        pass
            print(f"[resume] {len(seen)} rows already saved, skipping them")

        n = len(real)
        pred_per_cat = {c: np.zeros((n, len(mappings[c])), dtype=np.int64)
                        for c in ("warning", "environmental", "human")}
        gold_per_cat = {c: np.zeros((n, len(mappings[c])), dtype=np.int64)
                        for c in ("warning", "environmental", "human")}
        unknown_log  = {c: [] for c in ("warning", "environmental", "human")}
        parse_failures = 0
        api_errors = 0

        for i, lab in enumerate(gold_labels):
            for cat in ("warning", "environmental", "human"):
                for k in lab["subcategories"][cat]:
                    gold_per_cat[cat][i, k] = 1

        t0 = time.time()
        with open(raw_path, "a", encoding="utf-8") as raw_f:
            for i in range(n):
                if i in seen:
                    payload = saved_preds.get(i)
                    v = llm_response_to_y(payload, mappings, unknown_log)
                    for cat in v:
                        pred_per_cat[cat][i] = v[cat]
                    continue

                user_prompt = build_user_prompt(row_texts[i])
                try:
                    raw = call_fn(system_prompt, user_prompt)
                except Exception as e:
                    api_errors += 1
                    print(f"  [row {i}] API error: {e}", flush=True)
                    raw = ""

                payload = _parse_json_payload(raw)
                if payload is None and raw:
                    parse_failures += 1
                    if parse_failures <= 3:
                        print(f"  [row {i}] PARSE FAIL — raw (first 300 chars): {repr(raw[:300])}", flush=True)
                v = llm_response_to_y(payload, mappings, unknown_log)
                for cat in v:
                    pred_per_cat[cat][i] = v[cat]

                raw_f.write(json.dumps({
                    "row_index":    i,
                    "raw_response": raw,
                    "payload":      payload,
                }, ensure_ascii=False) + "\n")
                raw_f.flush()

                if (i + 1) % 10 == 0 or (i + 1) == n:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta  = (n - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{n}]  rate={rate:.2f}/s  "
                          f"eta={eta/60:.1f}min  parse_fail={parse_failures}  "
                          f"api_err={api_errors}", flush=True)

                if throttle_eff > 0:
                    time.sleep(throttle_eff / 1000.0)

        # ── Score ──
        summary = {
            "provider": prov, "model": model_name,
            "n_rows": n, "mode": "full_row",
            "parse_failures": parse_failures, "api_errors": api_errors,
            "unknown_label_counts": {cat: len(unknown_log[cat]) for cat in unknown_log},
            "stage2": {},
        }
        for cat in ("warning", "environmental", "human"):
            m = _multilabel_metrics(gold_per_cat[cat], pred_per_cat[cat])
            summary["stage2"][cat] = m
            print(f"\n--- B11/{prov} [{cat}] ---")
            print(f"  micro-F1={m['micro_f1']:.4f}  macro-F1={m['macro_f1']:.4f}  "
                  f"P={m['precision_micro']:.4f}  R={m['recall_micro']:.4f}")

        all_y = np.hstack([gold_per_cat[c] for c in ("warning", "environmental", "human")])
        all_p = np.hstack([pred_per_cat[c] for c in ("warning", "environmental", "human")])
        overall = _multilabel_metrics(all_y, all_p)
        summary["overall"] = overall
        print(f"\n--- B11/{prov} OVERALL ---")
        print(f"  micro-F1={overall['micro_f1']:.4f}  macro-F1={overall['macro_f1']:.4f}  "
              f"P={overall['precision_micro']:.4f}  R={overall['recall_micro']:.4f}")

        if any(unknown_log[c] for c in unknown_log):
            print(f"\n[B11/{prov}] LLM produced UNKNOWN label names:")
            for cat in unknown_log:
                for nm, cnt in Counter(unknown_log[cat]).most_common(5):
                    print(f"  [{cat}] {cnt}x  {nm!r}")

        with open(prov_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved -> {prov_summary}")
        print(f"Raw preds     -> {raw_path}")

        all_summaries[prov] = summary

    # Also write a combined summary at the top-level out_dir
    # (for backward-compat with --compare-only which expects summary_path)
    if all_summaries:
        last_summary = all_summaries[providers_to_run[-1]]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(last_summary, f, indent=2, default=str)

    # ── Final comparison table ──
    b10_path = os.path.join("baseline_llm_results", "test_metrics_summary.json")
    print(f"\n{'='*100}")
    print("  COMPARISON: B11 variants vs B10 (narr-only) vs V7 (fine-tuned)")
    print(f"{'='*100}")
    hdr = f"  {'Model':<42}  {'OverallμF1':>10}  {'OverallmF1':>10}  {'Warn μF1':>8}  {'Env μF1':>8}  {'Hum μF1':>8}"
    print(hdr)
    print("  " + "-" * 98)

    def _fmt_f(v):
        return f"{v:.4f}" if v == v else "  N/A  "

    def _print_row(label, s):
        ov = s.get("overall", {})
        w  = s.get("stage2", {}).get("warning", {})
        e  = s.get("stage2", {}).get("environmental", {})
        h  = s.get("stage2", {}).get("human", {})
        print(f"  {label:<42}  "
              f"{_fmt_f(ov.get('micro_f1', float('nan'))):>10}  "
              f"{_fmt_f(ov.get('macro_f1', float('nan'))):>10}  "
              f"{_fmt_f(w.get('micro_f1',  float('nan'))):>8}  "
              f"{_fmt_f(e.get('micro_f1',  float('nan'))):>8}  "
              f"{_fmt_f(h.get('micro_f1',  float('nan'))):>8}")

    for prov, s in all_summaries.items():
        _print_row(f"B11/{prov} (full-row, {s['model']})", s)

    if os.path.exists(b10_path):
        with open(b10_path, encoding="utf-8") as f:
            b10 = json.load(f)
        _print_row(f"B10 (narr-only, {b10.get('model','?')})", b10)

    v7m = _load_v7_metrics(exp_dir)
    if v7m:
        _print_row("V7  (fine-tuned model)", v7m)
    print()


if __name__ == "__main__":
    main()
