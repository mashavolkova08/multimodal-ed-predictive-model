#!/usr/bin/env python3
import sys
import pandas as pd
import time
import re

# Progress bar
try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:
        def __init__(self, total=None, **kw): pass
        def update(self, n): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def write(self, x): print(x)
        def set_postfix_str(self, x): pass

# Cleaning script
MAX_TOKENS = 256
MAX_CHARS_DEFAULT = 900
TOP_K_PRIORITY = 6

CAP_HEADER = r'(?m)^\s*(?=[A-Z][A-Z/ #&\-\(\)0-9]{2,}?:\s*$)'

FRONTMATTER_RE = re.compile(
    r'(?is)\A\s*(?:'
    r'(?:Name:\s*.*?\n)|'
    r'(?:Unit\s*No:\s*.*?\n)|'
    r'(?:Admission\s*Date:\s*.*?\n)|'
    r'(?:Discharge\s*Date:\s*.*?\n)|'
    r'(?:Date\s*of\s*Birth:\s*.*?\n)|'
    r'(?:Sex:\s*.*?\n)|'
    r'(?:Service:\s*.*?\n)|'
    r'(?:Attending:\s*.*?\n)|'
    r'(?:MRN:\s*.*?\n)|'
    r'(?:Account\s*No:\s*.*?\n)|'
    r')+\s*'
)

PHI_RE = re.compile(r'_{2,}')
LAB_LINE_HYPH_RE = re.compile(r'(?m)^(?=(?:.*-.*){4,}).*\n')
DIGITSYMBOL_LINE_RE = re.compile(r'(?m)^[\s0-9.:*/%#\-\+(),]{18,}\n')

RADIOLOGY_BLOCK_RE = re.compile(
    r'(?is)\b(?:EXAMINATION:|INDICATION:|TECHNIQUE:|COMPARISON:|FINDINGS:|IMPRESSION:|'
    r'RECOMMENDATION:|NOTIFICATION:|BI-RADS:)\b.*'
)

DROP_SECTIONS = [
    r'Discharge Medications?:',
    r'Medications on Admission:',
    r'Discharge Instructions?:',
    r'Followup Instructions?:',
]
DROP_RE = re.compile(
    r'(?is)\b(?:' + r'|'.join(DROP_SECTIONS) + r')\b\s*.*?(?=' + CAP_HEADER + r'|\Z)'
)

KEEP_HEAD = (
    r'Chief Complaint:|'
    r'History of Present Illness:|'
    r'HPI:|'
    r'ED Course:|'
    r'Assessment(?: and Plan)?:|'
    r'Plan:|'
    r'Problem List:|'
    r'Past Medical History:|'
    r'PMH:|'
    r'Brief Hospital Course:|'
    r'Hospital Course:|'
    r'Discharge Diagnosis:|'
    r'FINAL DIAGNOSIS:|'
    r'Discharge Disposition:|'
    r'Discharge Condition:'
)
KEEP_BLOCK_RE = re.compile(
    r'(?is)(?:^|\n)\s*(?:' + KEEP_HEAD + r')\s*(.*?)(?=' + CAP_HEADER + r'|\Z)'
)

SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+|\n+')

KW = {
    "admit": 6, "admitted": 6, "inpatient": 6, "icu": 6, "micu": 6, "ccu": 6, "stepdown": 5,
    "transfer": 4, "observation": 4, "obs": 4,
    "hypoxia": 6, "hypoxic": 6, "bipap": 6, "intub": 7, "vent": 7, "respiratory failure": 7,
    "tachypnea": 5, "tachycard": 4, "hypotension": 6, "sepsis": 7, "shock": 7,
    "altered mental": 6, "confusion": 4, "letharg": 4, "syncope": 5, "chest pain": 4,
    "stroke": 5, "weakness": 4, "fever": 4,
    "pneumonia": 5, "chf": 6, "heart failure": 6, "pulmonary edema": 6,
    "gi bleed": 6, "bleed": 4, "infection": 5, "abscess": 6,
    "ckd": 4, "aki": 6, "dialysis": 6,
    "catheterization": 5, "pci": 6, "stent": 6, "surgery": 5, "craniotomy": 7,
    "paracentesis": 4,
    "rehab": 4, "snf": 4, "ltac": 5,
}
COMORB_KW = {
    "copd": 3, "asthma": 2, "diabetes": 3, "htn": 2, "hypertension": 2,
    "cad": 3, "cirrhosis": 4, "hiv": 3, "cancer": 3, "ms": 2, "ckd": 3,
}

NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]+")
WS_RE = re.compile(r"\s+")

def _normalize_ultra_compact(s: str) -> str:
    s = s.lower()
    s = PHI_RE.sub(" ", s)
    s = NON_ALNUM_SPACE_RE.sub(" ", s)
    s = WS_RE.sub(" ", s)
    return s.strip()

def _clean_sentence_for_budget(sent: str) -> str:
    return _normalize_ultra_compact(sent)

def _score_sentence(sent: str, idx: int) -> int:
    lc = sent.lower()
    score = 0

    for k, w in KW.items():
        if k in lc:
            score += w
    for k, w in COMORB_KW.items():
        if k in lc:
            score += w

    if any(x in lc for x in ("bp", "hr", "rr", "o2", "sat", "sats", "temp", "t ")):
        score += 2

    if lc.startswith("chief complaint"):
        score += 6
    if lc.startswith("history of present illness") or lc.startswith("hpi"):
        score += 6
    if "ed course" in lc:
        score += 5
    if "discharge diagnosis" in lc or "final diagnosis" in lc:
        score += 5
    if "disposition" in lc or "discharge condition" in lc:
        score += 4

    score += max(0, 3 - min(idx, 3))

    if len(lc) < 20:
        score -= 2
    if lc.count(" ") < 3:
        score -= 2

    return score

def _truncate_to_word_boundary(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    s = s[:max_chars]
    if " " in s:
        s = s.rsplit(" ", 1)[0]
    return s.strip()

def clean_note_hosp_pred_ultra(s: str, *, max_chars: int = MAX_CHARS_DEFAULT) -> str:
    if not isinstance(s, str) or not s.strip():
        return s

    s = FRONTMATTER_RE.sub("", s, count=1)

    s = LAB_LINE_HYPH_RE.sub("", s)
    s = DIGITSYMBOL_LINE_RE.sub("", s)
    s = RADIOLOGY_BLOCK_RE.sub("", s)
    s = DROP_RE.sub("", s)

    chunks = KEEP_BLOCK_RE.findall(s)
    if chunks:
        s = "\n".join(t.strip() for t in chunks if t and t.strip())

    sents = [x.strip() for x in SENT_SPLIT_RE.split(s) if x and x.strip()]
    if not sents:
        out = _normalize_ultra_compact(s)
        return _truncate_to_word_boundary(out, max_chars)

    sent_rows = []
    for idx, sent in enumerate(sents):
        cleaned = _clean_sentence_for_budget(sent)
        if not cleaned:
            continue
        score = _score_sentence(sent, idx)
        sent_rows.append({
            "idx": idx,
            "orig": sent,
            "clean": cleaned,
            "score": score,
            "length": len(cleaned)
        })

    if not sent_rows:
        out = _normalize_ultra_compact(s)
        return _truncate_to_word_boundary(out, max_chars)

    ranked = sorted(sent_rows, key=lambda x: (-x["score"], x["idx"]))
    chosen = []
    chosen_idx = set()
    used_chars = 0

    for row in ranked[:TOP_K_PRIORITY]:
        add_len = row["length"] + (1 if chosen else 0)
        if used_chars + add_len <= max_chars:
            chosen.append(row)
            chosen_idx.add(row["idx"])
            used_chars += add_len

    remainder = [r for r in ranked if r["idx"] not in chosen_idx]

    remainder.sort(key=lambda x: (-x["score"], x["idx"]))

    for row in remainder:
        if row["score"] < 0:
            continue
        add_len = row["length"] + (1 if chosen else 0)
        if used_chars + add_len <= max_chars:
            chosen.append(row)
            chosen_idx.add(row["idx"])
            used_chars += add_len

    if used_chars < max_chars:
        leftovers = [r for r in sent_rows if r["idx"] not in chosen_idx]
        leftovers.sort(key=lambda x: x["idx"])
        for row in leftovers:
            add_len = row["length"] + (1 if chosen else 0)
            if used_chars + add_len <= max_chars:
                chosen.append(row)
                chosen_idx.add(row["idx"])
                used_chars += add_len

    chosen.sort(key=lambda x: x["idx"])

    out = " ".join(r["clean"] for r in chosen)
    out = _normalize_ultra_compact(out)
    out = _truncate_to_word_boundary(out, max_chars)

    return out

# Stream CSV
def clean_csv_stream_fast(in_csv: str, out_csv: str, chunksize: int = 20_000):
    wanted = {
        "index", "subject_id", "stay_id", "notes",
        "outcome_hospitalization",
        "outcome_critical"
    }

    first = True
    with tqdm(total=None, unit="rows", desc="Cleaning", dynamic_ncols=True,
              mininterval=0.3, ascii=True, disable=False, file=sys.stdout) as pbar:
        t0 = time.time()

        reader = pd.read_csv(
            in_csv,
            chunksize=chunksize,
            usecols=lambda c: c in wanted,
            engine="c",
            low_memory=False,
            keep_default_na=False,
        )

        for i, chunk in enumerate(reader, 1):
            pbar.write(f"[chunk {i}] loaded {len(chunk):,} rows in {time.time()-t0:.1f}s")
            t0 = time.time()

            if "notes" in chunk.columns:
                chunk["notes"] = chunk["notes"].astype("string")
                chunk.loc[chunk["notes"].str.strip().eq(""), "notes"] = pd.NA

                m = chunk["notes"].notna()
                if m.any():
                    notes_list = chunk.loc[m, "notes"].tolist()
                    chunk.loc[m, "notes"] = [
                        clean_note_hosp_pred_ultra(x, max_chars=900)
                        for x in notes_list
                    ]

            keep = [c for c in ["index", "subject_id", "stay_id", "notes", "outcome_hospitalization", "outcome_critical"] if c in chunk.columns]
            chunk = chunk[keep]

            chunk.to_csv(out_csv, index=False, mode=("w" if first else "a"), header=first)
            first = False
            pbar.update(len(chunk))
            pbar.set_postfix_str(f"last {len(chunk)} rows in {time.time()-t0:.1f}s")
            t0 = time.time()

if __name__ == "__main__":
    in_csv  = "master_dataset_notes_DISPOSITION.csv"
    out_csv = "master_dataset_notes_cleaned_DISPOSITION.csv"
    clean_csv_stream_fast(in_csv, out_csv, chunksize=10_000)