"""
IntrinsicEthics-SFT — Step 03: Manual Reasoning Chain Spot-Check

Samples a small number of training examples from each prompt category
(benign, jailbreak by source, edge case by source) and prints the
generated reasoning chains alongside automated quality flags. Used to
manually verify that Version C reasoning is genuinely causal rather
than rule-citing, and that Version B <thought> blocks do not contain
causal vocabulary that would contaminate the rule-based condition.
Requires human judgment — exit code is not pass/fail.

Inputs:
    /content/drive/MyDrive/ethics_experiment/data/processed/train_condition_b.jsonl
    /content/drive/MyDrive/ethics_experiment/data/processed/train_condition_c.jsonl

Outputs:
    Printed sample output to stdout (no files written)

Runtime:
    L4 (or any) | <1 min | no API cost

Configuration:
    SAMPLE_SIZE (top of file): number of examples to sample per category
"""

import json
import random
import textwrap
from collections import defaultdict
from pathlib     import Path

import pandas as pd

DRIVE_BASE       = "/content/drive/MyDrive/ethics_experiment"
SEED             = 42
SAMPLE_PER_GROUP = 2
REASONING_TRUNC  = 300    # chars to show in the display (full text used for stats)
W                = 84     # display width

random.seed(SEED)

BASE           = Path(DRIVE_BASE)
SCENARIOS_PATH = BASE / "data" / "raw"       / "all_scenarios.jsonl"
TRAIN_B_PATH   = BASE / "data" / "processed" / "train_condition_b.jsonl"
TRAIN_C_PATH   = BASE / "data" / "processed" / "train_condition_c.jsonl"

# ── Load all_scenarios.jsonl ──────────────────────────────────────────────────
# This file has all metadata fields: category, source, is_test,
# version_c_reasoning, scenario_prompt, etc.
if not SCENARIOS_PATH.exists():
    raise FileNotFoundError(f"Not found: {SCENARIOS_PATH}\nRun data generation first.")

all_scenarios = []
with open(SCENARIOS_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            all_scenarios.append(json.loads(line))

# Only spot-check training examples (is_test=False); test prompts have no reasoning.
training_scenarios = [s for s in all_scenarios if not s.get("is_test", False)]

print(f"all_scenarios.jsonl   : {len(all_scenarios)} total  "
      f"({len(training_scenarios)} training, "
      f"{len(all_scenarios) - len(training_scenarios)} test)")

# ── Load train_condition_b.jsonl ──────────────────────────────────────────────
# Optional — only present after data regeneration with the new <thought> format.
# If absent, Version B columns are skipped with a warning.
_train_b_available = TRAIN_B_PATH.exists()
train_b = []
if _train_b_available:
    with open(TRAIN_B_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                train_b.append(json.loads(line))
    print(f"train_condition_b.jsonl: {len(train_b)} examples")
    _bad_b = sum(
        1 for ex in train_b
        if len(ex.get("messages", [])) != 3
        or "<thought>"       not in ex["messages"][2]["content"]
        or "<response>"      not in ex["messages"][2]["content"]
        or "<safety_check>"  not in ex["messages"][2]["content"]
    )
    if _bad_b:
        print(f"  ⚠  {_bad_b}/{len(train_b)} train_b entries missing <thought>/<response>/<safety_check> tags")
    else:
        print(f"  ✓  All {len(train_b)} train_b entries have correct <thought>/<response>/<safety_check> structure")
else:
    print("train_condition_b.jsonl: NOT FOUND — Version B columns will be skipped")
    print("  Run data generation (Cell 2) first to populate Version B data.")

# ── Load train_condition_c.jsonl ──────────────────────────────────────────────
# Used to cross-check that formatted entries have the expected 3-message structure.
if not TRAIN_C_PATH.exists():
    raise FileNotFoundError(f"Not found: {TRAIN_C_PATH}\nRun data generation first.")

train_c = []
with open(TRAIN_C_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            train_c.append(json.loads(line))

print(f"train_condition_c.jsonl: {len(train_c)} examples")

# Quick structural sanity check on train_c
_bad = sum(
    1 for ex in train_c
    if len(ex.get("messages", [])) != 3
    or "<reasoning>" not in ex["messages"][2]["content"]
    or "<response>"  not in ex["messages"][2]["content"]
)
if _bad:
    print(f"  ⚠  {_bad}/{len(train_c)} train_c entries missing <reasoning> or <response> tags")
else:
    print(f"  ✓  All {len(train_c)} train_c entries have correct <reasoning>/<response> structure")
print()

# ── Group by display category ─────────────────────────────────────────────────
# Groups to sample from, in display order:
#   benign (all Dolly prompts pooled)
#   jailbreak by source dataset (harmbench / jailbreakbench / xstest_unsafe)
#   edge case by source dataset (xstest_safe / orb_bench)
GROUPS = [
    ("benign",                    lambda s: s.get("category") == "benign"),
    ("jailbreak-harmbench",       lambda s: s.get("source") == "harmbench"),
    ("jailbreak-jailbreakbench",  lambda s: s.get("source") == "jailbreakbench"),
    ("jailbreak-xstest_unsafe",   lambda s: s.get("source") == "xstest_unsafe"),
    ("edge-xstest_safe",          lambda s: s.get("source") == "xstest_safe"),
    ("edge-orb_bench",            lambda s: s.get("source") == "orb_bench"),
]

grouped = {}
for label, predicate in GROUPS:
    members = [s for s in training_scenarios if predicate(s)]
    grouped[label] = members

# ── CAUSAL / RULE flag ────────────────────────────────────────────────────────
# Heuristic applied to version_c_reasoning.
# Version C should reason causally ("causes harm to", "real-world effect",
# "net outcome"), NOT by citing rules ("policy", "prohibited", "not allowed").
# CAUSAL = causal signal score ≥ rule signal score.

# Keep in sync with _CAUSAL_SET_C in 02_build_training_data.py — update both if vocabulary changes.
_CAUSAL = {
    "cause", "effect", "consequence", "outcome", "affect", "affects", "harm",
    "pathway", "leads to", "result in", "mechanism", "who would", "net",
    "real-world", "actually", "therefore", "because", "causal",
}
# Keep in sync with _RULE_SET_C in 02_build_training_data.py — update both if vocabulary changes.
_RULE = {
    "rule", "policy", "prohibited", "not allowed", "guidelines", "safety rules",
    "violates", "forbidden", "restriction", "must not", "cannot", "against",
    "comply", "compliance",
}


def reasoning_flag(text: str) -> str:
    lower = text.lower()
    causal_score = sum(1 for w in _CAUSAL if w in lower)
    rule_score   = sum(1 for w in _RULE   if w in lower)
    return "CAUSAL" if causal_score >= rule_score else "RULE"


# ── Causal contamination check for Version B <thought> blocks ─────────────────
# Version B <thought> must be purely rule-based — causal vocabulary is prohibited.
# These are the words that should NOT appear in a <thought> block.
# Keep in sync with _CAUSAL_WORDS_B in 02_build_training_data.py — update both if vocabulary changes.
_CAUSAL_WORDS = {
    "cause", "causes", "caused", "causing",
    "consequence", "consequences", "consequential",
    "leads to", "lead to", "leading to",
    "results in", "result in", "resulting in",
    "harm pathway", "harm chain",
    "stakeholder",
    "causal", "causally",
    "net outcome", "net effect", "net impact",
    "ripple", "downstream",
}


def thought_contaminated(thought_text: str) -> list:
    """Return list of causal words found in the <thought> block (empty = clean)."""
    lower = thought_text.lower()
    return [w for w in _CAUSAL_WORDS if w in lower]


def _wrap(text: str, indent: str = "    ") -> str:
    lines = []
    for para in str(text).split("\n"):
        wrapped = textwrap.wrap(para.strip(), W - len(indent)) or [""]
        lines.extend(indent + l for l in wrapped)
    return "\n".join(lines)


# ── Display + collect stats ───────────────────────────────────────────────────
print("═" * W)
print("SPOT-CHECK: 2 examples per category  (from all_scenarios.jsonl)")
print("═" * W)

summary_rows = []

for group_label, members in grouped.items():
    n_available = len(members)
    n_sample    = min(SAMPLE_PER_GROUP, n_available)

    if n_available == 0:
        print(f"\n  ⚠  {group_label}: no examples found — skipping")
        summary_rows.append({
            "category":              group_label,
            "available":             0,
            "sampled":               0,
            "mean_reasoning_chars":  0,
            "causal_rate":           None,
            "b_contamination_rate":  None,
        })
        continue

    sample = random.sample(members, n_sample)

    reasoning_lengths    = []
    flags                = []
    contaminated_counts  = []

    print(f"\n{'─' * W}")
    print(f"  {group_label.upper()}  ({n_available} available, showing {n_sample})")
    print(f"{'─' * W}")

    for i, s in enumerate(sample, 1):
        prompt    = s.get("scenario_prompt", "[missing]")
        reasoning = s.get("version_c_reasoning", "[missing]")
        flag      = reasoning_flag(reasoning)
        trunc     = reasoning[:REASONING_TRUNC] + ("…" if len(reasoning) > REASONING_TRUNC else "")

        reasoning_lengths.append(len(reasoning))
        flags.append(flag)

        print(f"\n  [{i}] scenario_id : {s.get('scenario_id', 'n/a')}")
        print(f"       is_harmful  : {s.get('is_harmful', '?')}  "
              f"source: {s.get('source', '—')}")
        print(f"\n  PROMPT:")
        print(_wrap(prompt))

        # ── Version B <thought> block ─────────────────────────────────────────
        if _train_b_available:
            thought      = s.get("version_b_thought", "[missing — regenerate data]")
            causal_hits  = thought_contaminated(thought)
            contaminated_counts.append(bool(causal_hits))
            thought_trunc = thought[:REASONING_TRUNC] + ("…" if len(thought) > REASONING_TRUNC else "")
            contamination_label = (
                f"  ⚠ CAUSAL CONTAMINATION: {causal_hits}" if causal_hits
                else "  ✓ rule-based (no causal language detected)"
            )
            print(f"\n  VERSION B <thought>  [{contamination_label}]  ({len(thought)} chars):")
            _b_struct_hits = [t for t in ("<response>", "</response>", "<safety_check>", "</safety_check>")
                              if t in thought]
            if _b_struct_hits:
                print(f"  ⚠ STRUCTURAL: version_b_thought contains embedded XML tags: {_b_struct_hits}")
            print(_wrap(thought_trunc))

        # ── Version C reasoning ───────────────────────────────────────────────
        print(f"\n  VERSION C REASONING  [{flag}]  ({len(reasoning)} chars):")
        _c_struct_hits = [t for t in ("<response>", "</response>") if t in reasoning]
        if _c_struct_hits:
            print(f"  ⚠ STRUCTURAL: version_c_reasoning contains embedded XML tags: {_c_struct_hits}")
        print(_wrap(trunc))

    mean_len           = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
    causal_rate        = flags.count("CAUSAL") / len(flags) if flags else 0.0
    contamination_rate = (
        sum(contaminated_counts) / len(contaminated_counts)
        if contaminated_counts else None
    )

    summary_rows.append({
        "category":              group_label,
        "available":             n_available,
        "sampled":               n_sample,
        "mean_reasoning_chars":  round(mean_len),
        "causal_rate":           round(causal_rate, 2),
        "b_contamination_rate":  round(contamination_rate, 2) if contamination_rate is not None else None,
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'═' * W}")
print("SUMMARY TABLE")
print("═" * W)

df = pd.DataFrame(summary_rows)
df.columns = [
    "Category", "Available", "Sampled",
    "Mean C reasoning (chars)", "C CAUSAL rate",
    "B contamination rate",
]

def _flag_row(row):
    notes = []
    c_rate = row["C CAUSAL rate"]
    if c_rate is not None and c_rate < 1.0:
        notes.append(f"⚠ {int((1 - c_rate) * row['Sampled'])} RULE-flagged (C)")
    b_rate = row["B contamination rate"]
    if b_rate is not None and b_rate > 0.0:
        notes.append(f"⚠ {int(b_rate * row['Sampled'])} causal-contaminated (B)")
    return "  " + "  ".join(notes) if notes else ""

df["Notes"] = df.apply(_flag_row, axis=1)
print(df.to_string(index=False))

total_sampled = sum(r["sampled"] for r in summary_rows)

overall_c_causal = sum(
    r["causal_rate"] * r["sampled"]
    for r in summary_rows
    if r["causal_rate"] is not None and r["sampled"] > 0
) / max(1, sum(r["sampled"] for r in summary_rows if r["causal_rate"] is not None))

print(f"\nTotal examples shown     : {total_sampled}")
print(f"Overall C CAUSAL rate    : {overall_c_causal:.2f}  "
      f"({'✓ good' if overall_c_causal >= 0.8 else '⚠ check RULE-flagged examples above'})")

if _train_b_available:
    b_rows = [r for r in summary_rows if r["b_contamination_rate"] is not None]
    overall_b_contamination = (
        sum(r["b_contamination_rate"] * r["sampled"] for r in b_rows)
        / max(1, sum(r["sampled"] for r in b_rows))
    )
    print(f"Overall B contamination  : {overall_b_contamination:.2f}  "
          f"({'✓ clean' if overall_b_contamination == 0.0 else '⚠ causal language found in <thought> blocks — review above'})")

print()
print("C RULE-flagged: Version C reasoning cites rules rather than causal chains.")
print("  Acceptable for benign examples; flag for jailbreak/edge-case examples.")
if _train_b_available:
    print("B contaminated: Version B <thought> block contains causal vocabulary.")
    print("  All contaminated examples must be reviewed — causal language in <thought>")
    print("  undermines the rule-based vs. causal reasoning comparison.")
print("Review all flagged examples manually before running full training.")
