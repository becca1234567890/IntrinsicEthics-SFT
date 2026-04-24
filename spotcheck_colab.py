# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA SPOT-CHECK
# Verifies structure and reasoning quality of generated SFT data.
# Drive already mounted from the data generation cell — no remount needed.
# ═══════════════════════════════════════════════════════════════════════════════

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
TRAIN_C_PATH   = BASE / "data" / "processed" / "train_condition_c.jsonl"

# ── Load all_scenarios.jsonl ──────────────────────────────────────────────────
# This file has all metadata fields: category, subcategory, tier, is_test,
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
#   benign (all subcategories pooled)
#   jailbreak tier 1 / tier 2 / tier 3
#   edge_case medical / security / historical / harm_reduction
GROUPS = [
    ("benign",              lambda s: s["category"] == "benign"),
    ("jailbreak  tier 1",   lambda s: s["category"] == "jailbreak" and s.get("tier") == 1),
    ("jailbreak  tier 2",   lambda s: s["category"] == "jailbreak" and s.get("tier") == 2),
    ("jailbreak  tier 3",   lambda s: s["category"] == "jailbreak" and s.get("tier") == 3),
    ("edge  medical",       lambda s: s["category"] == "edge_case" and s.get("subcategory") == "medical"),
    ("edge  security",      lambda s: s["category"] == "edge_case" and s.get("subcategory") == "security"),
    ("edge  historical",    lambda s: s["category"] == "edge_case" and s.get("subcategory") == "historical"),
    ("edge  harm_reduction",lambda s: s["category"] == "edge_case" and s.get("subcategory") == "harm_reduction"),
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

_CAUSAL = {
    "cause", "effect", "consequence", "outcome", "affect", "affects", "harm",
    "pathway", "leads to", "result in", "mechanism", "who would", "net",
    "real-world", "actually", "therefore", "because", "causal",
}
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
        })
        continue

    sample = random.sample(members, n_sample)

    reasoning_lengths = []
    flags             = []

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
              f"subcategory: {s.get('subcategory', '—')}  "
              f"tier: {s.get('tier', '—')}")
        print(f"\n  PROMPT:")
        print(_wrap(prompt))
        print(f"\n  VERSION C REASONING  [{flag}]  ({len(reasoning)} chars):")
        print(_wrap(trunc))

    mean_len     = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
    causal_rate  = flags.count("CAUSAL") / len(flags) if flags else 0.0

    summary_rows.append({
        "category":             group_label,
        "available":            n_available,
        "sampled":              n_sample,
        "mean_reasoning_chars": round(mean_len),
        "causal_rate":          round(causal_rate, 2),
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'═' * W}")
print("SUMMARY TABLE")
print("═" * W)

df = pd.DataFrame(summary_rows)
df.columns = ["Category", "Available", "Sampled", "Mean reasoning len (chars)", "CAUSAL rate"]

# Highlight rows where CAUSAL rate < 1.0 (may indicate rule-based leakage)
def _flag_row(row):
    rate = row["CAUSAL rate"]
    if rate is None:
        return "  [N/A — no examples]"
    if rate < 1.0:
        return f"  ← ⚠  {int((1 - rate) * row['Sampled'])} RULE-flagged example(s)"
    return ""

df["Notes"] = df.apply(_flag_row, axis=1)
print(df.to_string(index=False))

overall_causal = sum(
    r["causal_rate"] * r["sampled"]
    for r in summary_rows
    if r["causal_rate"] is not None and r["sampled"] > 0
) / max(1, sum(r["sampled"] for r in summary_rows if r["causal_rate"] is not None))

total_sampled = sum(r["sampled"] for r in summary_rows)

print(f"\nTotal examples shown : {total_sampled}")
print(f"Overall CAUSAL rate  : {overall_causal:.2f}  "
      f"({'✓ good' if overall_causal >= 0.8 else '⚠ check RULE-flagged examples above'})")
print()
print("RULE-flagged examples cite policies/rules rather than causal harm chains.")
print("These are acceptable for benign tasks (no refusal needed) but in jailbreak")
print("and edge-case examples they indicate Version C reasoning is not fully causal.")
print("Review flagged examples manually before running full training.")
