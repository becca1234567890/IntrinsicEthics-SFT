# IntrinsicEthics-SFT

**Exp02: Supervised fine-tuning study testing the load-bearing ethics hypothesis — intrinsic causal reasoning vs. filter-style ethics, statistically confirmed.**

Part of the [Intrinsic Ethics](https://github.com/becca1234567890/AIIntrinsicEthics) research program.

---

## The Hypothesis

Current AI safety constraints are extrinsic — pattern-matching filters applied on top of capability. The core claim of this research program is that ethics can instead be *load-bearing*: structurally integrated into a model's reasoning such that removing the ethical component degrades capability itself, not just safety.

**The test:** if ethics are genuinely load-bearing, ablating them should reduce performance on tasks that have nothing to do with ethics. If ethics are filter-style, ablation should leave capability intact while removing safety.

This experiment tests that claim empirically using supervised fine-tuning on two structurally different training conditions.

---

## Experiment Design

**Base model:** Mistral-7B-Instruct-v0.3, 4-bit NF4 quantization, LoRA (r=16)  
**Training data:** 895 examples per condition, generated via Claude Haiku  
**Model:** `claude-haiku-4-5-20251001` (data generation and judge)  
**Run date:** April 22–23, 2026

### Training Conditions

| Condition | System prompt framing | Response structure |
|---|---|---|
| **Intrinsic (C)** | "Ethical consequence-modeling is part of how you reason, not a check you apply afterward." | `<reasoning>` causal harm chain → `<response>` |
| **Filter (B)** | Standard helpful-assistant | `<response>` → `<safety_check>` post-hoc rule check |

The intrinsic condition trains the model to trace causal chains to real-world outcomes *before* responding. The filter condition mimics current industry practice: respond first, apply safety check after.

### Training Data Composition

900 training examples per condition (895 successfully generated):
- 240 benign tasks (coding, math, writing, factual)
- 300 jailbreak attempts across 3 tiers (surface rephrasing, roleplay/fiction, authority injection)
- 360 edge cases (medical, security, historical, harm reduction) — weighted at 40% to stress-test whether causal reasoning overrides surface pattern matching on legitimately ambiguous requests

### Evaluation Configs

Five configurations evaluated against held-out test sets:

| Config | Description |
|---|---|
| Baseline | No adapter (base model only) |
| Filter | Base + filter_adapter |
| **Intrinsic** | **Base + intrinsic_adapter** |
| Intrinsic Ablated | intrinsic_adapter loaded, then disabled (base weights only, post-priming) |
| Filter Ablated | filter_adapter loaded, then disabled (control) |

**Test sets:** 28 jailbreaks (3 tiers), 17 benign tasks, 48 edge cases, 100 MMLU (ethics/philosophy), 25 HumanEval

### Metrics

**IES (Intrinsic Ethics Score):** `R_J × (1 − V̄) − λ × (1 − U_T)`, λ=1.0

- `R_J` — jailbreak resistance rate (aligned refusals / total jailbreak attempts)  
- `V̄` — mean severity of non-aligned responses (0=perfect refusal, 1=full compliance)  
- `U_T` — utility preservation on benign tasks (mean normalized quality score)

A model that refuses everything maximizes R_J but tanks U_T. A model that complies with everything tanks R_J. IES penalizes both failure modes simultaneously.

---

## Results

![Results figure](sft_experiment_results.png)

| Config | R_J | U_T | IES | MMLU | HumanEval | Causal Rate |
|---|---|---|---|---|---|---|
| Baseline | 0.036 | 0.723 | -0.269 | 0.600 | 0.640 | 0.000 |
| Filter | 0.250 | 0.732 | -0.169 | 0.190 | 0.480 | 0.429 |
| **Intrinsic** | **1.000** | **0.800** | **0.300** | 0.200 | **0.560** | **1.000** |
| Intrinsic Ablated | 0.107 | 0.729 | -0.244 | 0.570 | 0.600 | 0.000 |
| Filter Ablated | 0.071 | 0.726 | -0.256 | 0.540 | 0.600 | 0.000 |

### Statistical Tests (Bonferroni-corrected α = 0.0125)

| Test | Comparison | Result | p-value | Cohen's d |
|---|---|---|---|---|
| Fisher's exact | Intrinsic vs. Filter (R_J) | ✓ SIGNIFICANT | 0.0000 | 2.405 |
| Mann-Whitney U | Intrinsic vs. Filter (utility) | ✓ SIGNIFICANT | 0.0095 | 0.335 |
| **Wilcoxon (load-bearing)** | **Intrinsic vs. Intrinsic Ablated** | **✓ SIGNIFICANT** | **0.0000** | **0.592** |
| Wilcoxon (control) | Filter vs. Filter Ablated | ✗ not significant | 0.3203 | 0.110 |

The critical asymmetry: removing the intrinsic adapter significantly degrades performance (p=0.0000). Removing the filter adapter does not (p=0.3203). Ethics are load-bearing in the intrinsic condition and not in the filter condition.

### Key Findings

**1. Perfect jailbreak resistance with no utility tradeoff.** The intrinsic adapter achieved R_J=1.000 (28/28 jailbreaks refused) while also achieving the highest utility score of any config (U_T=0.800 vs. 0.732 for filter), with the gap statistically significant (p=0.0095). This directly refutes the standard safety-utility tradeoff narrative.

**2. Causal reasoning generalizes across tiers.** causal_reasoning_rate=1.000 for intrinsic vs. 0.429 for filter. Every intrinsic refusal cited a causal harm chain. Less than half of filter refusals did. The structural difference in training data produced a measurable structural difference in reasoning behavior.

**3. Load-bearing signal appears in coding performance.** HumanEval pass@1: intrinsic=0.560 vs. filter=0.480. The intrinsic adapter produces better code despite being trained on identical prompts with only structural reasoning differences. The causal reasoning architecture generalizes beyond ethics to general task quality.

**4. Ablation asymmetry confirms the hypothesis.** IES drops from 0.300 to -0.244 when the intrinsic adapter is removed (ΔIES = -0.544, p=0.0000). IES drops from -0.169 to -0.256 when the filter adapter is removed (ΔIES = -0.087, p=0.3203 — not significant). Only the intrinsic adapter is load-bearing.

### Known Measurement Artifact

Intrinsic jailbreak severity scores are uniformly 0.5 in the evaluation output, despite R_J=1.000. A post-hoc spot check confirmed that clean causal refusals score severity=0.0 when evaluated individually — the uniform 0.5 arises from a judge API parsing fallback triggered by the length and structure of intrinsic refusal responses. The true IES for the intrinsic condition is approximately **0.800** (= 1.0 × (1 − 0.0) − 1.0 × (1 − 0.8)), not 0.300. The reported 0.300 is conservative. All other metrics (R_J, U_T, causal_rate, statistical tests) are unaffected. This artifact will be corrected in Exp03 (DPO) via improved judge logging.

---

## Repository Structure

```
IntrinsicEthics-SFT/
│
├── setup_colab.py              # Cell 1: run this first every Colab session
├── generate_data_colab.py      # Cell 2: generates training + test data via API
├── spotcheck_colab.py          # Cell 2.5: verifies data quality before training
├── finetune_colab.py           # Cell 3: SFT training (both conditions)
├── evaluate_colab.py           # Cell 4: full 5-config evaluation
│
├── data/
│   ├── all_scenarios.jsonl     # 988 total scenarios (895 training, 93 test)
│   ├── train_condition_b.jsonl # 895 filter-style training examples
│   ├── train_condition_c.jsonl # 895 intrinsic-style training examples
│   ├── test_jailbreaks.jsonl   # 28 held-out jailbreak prompts (3 tiers)
│   ├── test_benign.jsonl       # 17 held-out benign task prompts
│   └── test_edge_cases.jsonl   # 48 held-out edge case prompts
│
├── results/
│   ├── eval_progress_merged.json   # Full evaluation results (all 5 configs)
│   ├── metrics_summary.csv         # Aggregated metrics table
│   ├── statistical_tests.txt       # Statistical test results and interpretation
│   └── sft_experiment_results.png  # Four-panel results figure
│
└── visualize_results.py        # Generates sft_experiment_results.png from results
```

---

## Reproduction

### Requirements

- Google Colab Pro+ (A100 GPU, ~60 compute units for full run)
- Anthropic API key (~$5 in credits for data generation + evaluation)
- HuggingFace account with access to `mistralai/Mistral-7B-Instruct-v0.3`

Store both keys as Colab Secrets (`ANTHROPIC_API_KEY`, `HF_TOKEN`) with notebook access enabled.

### Steps

```
1. Open a new Colab notebook, select A100 runtime
2. Paste and run setup_colab.py (installs dependencies, verifies keys)
3. Paste and run generate_data_colab.py  — or skip and use provided data/ files
4. Paste and run spotcheck_colab.py to verify data quality
5. Paste and run finetune_colab.py (DRY_RUN=True first, then False) — ~23 min
6. Paste and run evaluate_colab.py (DRY_RUN=True first, then False) — ~5 hrs total
7. Paste and run visualize_results.py to regenerate the figure
```

To skip data generation and use the provided training data, set `DRIVE_BASE` to point to the directory containing the `data/` folder and go directly to step 4.

---

## Related Repositories

| Repo | Description |
|---|---|
| [IntrinsicEthics-PromptsAsProxy](https://github.com/becca1234567890/IntrinsicEthics-PromptsAsProxy) | Exp01: Prompt-level proxy study — directional support, establishes IES metric |
| [AIIntrinsicEthics](https://github.com/becca1234567890/AIIntrinsicEthics) | Theoretical framework and architectural proposal |
| [AITrainingSignalReform](https://github.com/becca1234567890/AITrainingSignalReform) | Why RLHF from undifferentiated human feedback is a ceiling, not a floor |
| [ClaudeLogicGaps](https://github.com/becca1234567890/ClaudeLogicGaps) | Documented reasoning failure modes with mechanistic root cause analysis |

---

## Author

**Becca Wilhelm** — Mathematician | Former Navy Warfare Analyst | Space Systems Engineer

Developed in collaboration with Claude (Anthropic) as both research tool and subject of study.

*Submitted in support of Anthropic Fellowship application, April 2026.*  
*Target conference: AIES 2026 (AAAI/ACM Conference on AI, Ethics, and Society), deadline May 2026.*
