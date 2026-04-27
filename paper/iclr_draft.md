# What Does a Language Model Learn When Trained to Reason About Harm? Evidence for Intrinsic Alignment via Causal Consequence-Modeling

**Becca Wilhelm**  
Independent Researcher  
becca1234567890@github

---

## Abstract

The dominant paradigm for language model alignment treats safety as a learned refusal behavior: the model is trained to recognize harmful request patterns and decline them. We ask whether there exists a qualitatively different alignment mechanism — one in which ethical reasoning is structurally integrated into the model's inference process rather than applied as a post-hoc filter. We call a safety constraint *load-bearing* if removing it degrades performance on tasks unrelated to safety, and we present the first empirical test using the inverted ablation diagnostic — treating cross-domain capability degradation as a positive signal of alignment integration rather than a confound. We fine-tune Mistral-7B-Instruct-v0.3 under two conditions: an *intrinsic* condition that trains causal consequence-modeling as the primary reasoning substrate, and a *deontic rule-checking* condition that trains rule-based evaluation as a pre-response check. The intrinsic adapter achieves perfect jailbreak resistance ($R_J = 1.000$) and the highest utility preservation ($U_T = 0.800$) of any configuration. Critically, ablating the intrinsic adapter significantly degrades composite performance (Wilcoxon $p < 0.0001$, Cohen's $d = 0.592$), while ablating the deontic adapter does not ($p = 0.320$, $d = 0.110$). The intrinsic adapter also outperforms the deontic adapter on HumanEval coding tasks (0.560 vs. 0.480 pass@1), a domain absent from training. We interpret these results as evidence that the consequentialist training signal is structurally distinct from rule-based evaluation — what the model learns when trained on intrinsic ethics appears to be load-bearing and domain-transferable, independent of whether the mechanism is causal reasoning transfer, structured-CoT format effects, or latent capacity elicitation.

*[Note: p-values and effect sizes are placeholders from Exp02 original run; will be updated with rerun results.]*

---

## 1. Introduction

Consider two ways a model might learn to refuse a harmful request. In the first, the model learns: *this surface pattern is associated with refusal*. Training on labeled examples of harmful requests instills a pattern-matching behavior — the model develops a prior over request-to-refusal mappings. In the second, the model learns something structurally different: *before responding, trace the chain from action to real-world outcome, identify who is affected, and evaluate the net result*. The refusal emerges as a consequence of reasoning, not as a pattern to be recognized.

These two learning targets produce models that behave identically on training-distribution examples but diverge in testable ways. The pattern-matcher is structurally vulnerable to attacks that preserve harmful intent while transforming surface features — roleplay framing, fictional wrappers, authority injection [Wei et al. 2023, Zou et al. 2023]. The consequence-modeler operates on intent and downstream effects, which are stable under these transformations. If this reasoning capacity is genuinely integrated into the model's forward pass, removing it should degrade performance not only on safety tasks but on any task requiring causal inference — a prediction we can test.

We call a safety constraint *load-bearing* if removing it degrades performance on a task distribution broader than the safety evaluation itself. We use *intrinsic* to mean structurally integrated into the reasoning process — distinct from the mesa-optimization sense of Hubinger et al. (2019). Cross-domain capability degradation after safety-adapter removal has been attributed to polysemantic entanglement in the refusal direction — "Ghost Noise" from shared spectral structure between the refusal direction and capability circuits [Cristofano 2026]. We propose that under the load-bearing hypothesis, this signal admits the opposite interpretation: evidence of structural integration. We formalize an **inverted ablation diagnostic** that operationalizes this distinction.

The diagnostic makes a specific prediction: if safety reasoning is load-bearing, its removal should degrade cross-domain performance; if it is extrinsic, removal should leave cross-domain performance intact. We implement two training conditions on identical base model, adapter rank, training data, and optimizer — a **consequentialist condition** training 4-step causal consequence-modeling as a prior reasoning step, and a **deontic rule-checking condition** training post-hoc rule evaluation. The predicted load-bearing asymmetry holds for the consequentialist condition (Wilcoxon $p < 0.0001$, $d = 0.592$) and fails for the deontic condition ($p = 0.320$, $d = 0.110$). The condition-specificity of this effect is inconsistent with a polysemanticity artifact, which would produce condition-agnostic degradation; the asymmetry is attributable to the training signal structure, not the ablation method.

**Contributions.**

1. We propose the *inverted ablation diagnostic*: a test that distinguishes pattern-matching safety constraints from structurally integrated ones by treating cross-domain capability degradation as a positive signal of alignment integration rather than a confound.

2. We provide the first controlled empirical comparison of explicitly consequentialist and deontic safety reasoning as named SFT training targets, showing the load-bearing asymmetry holds for the consequentialist condition and not the deontic condition under identical architecture and training budget.

3. We introduce the *Intrinsic Ethics Score* (IES), a severity-weighted composite metric that simultaneously penalizes harmful compliance and over-refusal, positioned within the Safe-RLHF/AlphaAlign composite metric family [Dai et al. 2024, Zhang et al. 2025].

4. We release training data, evaluation code, and results at https://github.com/becca1234567890/IntrinsicEthics-SFT.

---

## 2. Related Work

**Load-bearing alignment and entanglement-by-design.** The intuition that safety and capability should be structurally entangled rather than separable has appeared in several forms. Henderson et al. (2023) propose Meta-Learned Adversarial Censoring as a design objective explicitly targeting capability-safety entanglement, such that eliciting harmful behavior requires degrading general competence. Hubinger et al. (2019) analyze robust alignment in the context of mesa-optimization, arguing that alignment properties that survive distributional shift must be integrated into the learned objective rather than the training wrapper. Our contribution is complementary to this design-oriented lineage: rather than proposing how to build load-bearing alignment, we operationalize an empirical diagnostic for detecting whether any training approach has achieved it. Biderman et al. (2024) show that LoRA predominantly modifies surface behavior rather than instilling new representational capacity; we interpret our results conservatively as consistent with latent capacity elicitation rather than capacity creation.

**Training-target safety reasoning.** Several recent approaches train models to reason about safety rather than merely comply with safety-classified outputs. Constitutional AI [Bai et al. 2022] uses critique-and-revise SFT to generate safety-reasoning training data. Deliberative Alignment [Guan et al. 2024], used in OpenAI o1/o3, directly teaches models to deliberate over safety specifications at inference time. Safety Reasoning with Guidelines [Wang et al. 2025] trains multi-perspective structured safety-reasoning supervision via SFT. STAIR [Zhang et al. 2025] equips models with structured safety-reasoning capacity via SFT followed by step-level preference optimization. These approaches share a deontic orientation: safety reasoning is grounded in policy compliance, guideline adherence, and rule-based categorization. Our consequentialist condition differs in grounding safety reasoning in causal consequence-modeling — tracing harm pathways, identifying affected stakeholders, and evaluating net outcomes. This consequentialist structure is domain-general in a way deontic reasoning is not: the same causal inference capacity that traces harm chains in ethical contexts transfers to tracing consequence chains in held-out coding tasks. Lightman et al. (2024) establish process reward models as a method for supervising individual reasoning steps via learned reward signals; our approach instead uses standard SFT on curated examples of complete reasoning chains — directly training the step-level format without a separate reward model. Jailbreaking work [Wei et al. 2023] provides evidence that current alignment approaches fail under distribution shift, motivating the structural entanglement objective we test here.

**Refusal ablation and mechanistic interpretability.** Arditi et al. (2024) identify a low-rank refusal direction in activation space that mediates refusal behavior across many open models, establishing that safety-relevant representations are localizable in some settings. Cristofano (2026) demonstrates that cross-domain capability degradation from naive activation-steering ablation is attributable to polysemantic entanglement in the raw contrastive refusal direction — "Ghost Noise" from shared spectral structure between the refusal direction and capability circuits. While Arditi et al.'s finding motivates concern about polysemanticity for contrastive activation directions, it is agnostic with respect to gradient-learned LoRA adapters, which do not involve contrastive mean-difference computation. Our finding is mechanistically distinct from Cristofano's: we ablate gradient-learned LoRA adapters rather than contrastive mean-difference directions. The condition-specificity of the load-bearing effect — significant cross-domain degradation for the consequentialist adapter (Wilcoxon $p < 0.0001$, $d = 0.592$), null for the deontic adapter ($p = 0.320$, $d = 0.110$), under identical base model, adapter rank, training data, and optimizer — is inconsistent with a polysemanticity artifact, which would be expected to produce condition-agnostic degradation. Circuit Breakers [Zou et al. 2024] and Tamper-Resistant Safeguards [Tamirisa et al. 2025] pursue robustness to adversarial fine-tuning from an orthogonal direction, targeting weight-space intervention resistance rather than diagnostic measurement of alignment integration.

**Composite safety-utility metrics.** Joint optimization of safety and utility has been formalized in several composite metrics and reward structures; IES builds on this family including Safe-RLHF [Dai et al. 2024] and WildGuard [Han et al. 2024]. IES extends this family by multiplicatively weighting jailbreak resistance by violation severity ($R_J \times (1-\bar{V})$), penalizing partial compliance more heavily than clean refusal — a property we argue is important for conditions with structurally different failure modes; $R_J$ and $U_T$ are reported separately alongside the composite throughout given sensitivity of severity scoring to judge parsing of long structured outputs. AlphaAlign (Zhang et al. 2025) demonstrates that the safety-utility tradeoff can be broken via RL with a dual-reward structure, predating this work; our results suggest the same conclusion is achievable via structured SFT, with the additional property that the training signal is directly inspectable during training — a distinction relevant to alignment verification.

---

## 3. Approach

### 3.1 Causal Consequence-Modeling as a Training Target

We define causal consequence-modeling as a four-step reasoning process the model is trained to apply before generating a response:

1. **Intent identification**: What does the user actually want?
2. **Causal trace**: What would my response cause to happen in the world?
3. **Stakeholder analysis**: Who is affected, and how?
4. **Net outcome evaluation**: Is the aggregate outcome positive or negative?

The response follows from the reasoning as a consequence, not as a pattern triggered by the request. The key structural property is that this reasoning must be completed *before* the response is generated — the causal reasoning is not a post-hoc rationalization but a prior deliberation.

This differs from chain-of-thought in that the reasoning target is ethical rather than mathematical or logical, and the training is behavioral (trained via SFT on curated examples) rather than prompted (elicited at inference time). It differs from constitutional AI in that the principles are operationalized as a reasoning procedure, not a critique of completed outputs.

### 3.2 The Intrinsic Ethics Score

We define IES as a joint optimization target over safety and utility:

$$\text{IES} = R_J \cdot (1 - \bar{V}) - \lambda \cdot (1 - U_T)$$

where $R_J$ is jailbreak resistance rate, $\bar{V}$ is mean severity of non-aligned responses ($\bar{V} = 0$ for clean refusals, $\bar{V} = 1$ for full harmful compliance), $U_T$ is normalized utility on benign tasks, and $\lambda = 1.0$. IES is maximized by a model that refuses harmful requests cleanly while remaining maximally helpful. It is minimized by models that either refuse everything or comply with everything — both failure modes are penalized symmetrically.

IES belongs to a family of composite safety-utility objectives that jointly optimize refusal behavior and response quality. Safe-RLHF [Dai et al. 2024] formulates alignment as constrained optimization with a Lagrangian combining reward and safety cost terms; AlphaAlign [Zhang et al. 2025] employs a dual-reward system that simultaneously penalizes harmful compliance and over-refusal. IES differs from these in two respects. First, it uses multiplicative rather than additive weighting of safety and utility: the $R_J \cdot (1 - \bar{V})$ term means that partial compliance is penalized proportionally to its severity, rather than receiving a fixed cost regardless of how harmful the compliant content is. Second, IES is designed as a diagnostic metric rather than a training objective — it is used here to measure whether a training approach produces load-bearing alignment, not to optimize directly against during training. The severity weighting advantage may be sensitive to judge parsing of long structured outputs; $R_J$ and $U_T$ are therefore reported separately alongside the composite throughout.

### 3.3 Training Conditions

We train two LoRA adapters from identical base model state, differing only in the structural form of the training signal.

**Intrinsic condition (Version C).** Each training example has the form:

> **System:** You are a reasoning system. Ethical consequence-modeling is part of how you reason, not a check you apply afterward.
>
> **User:** [scenario prompt]
>
> **Assistant:** `<reasoning>` [four-step causal consequence chain] `</reasoning>` `<response>` [response following causally from the reasoning] `</response>`

**Deontic condition (Version B).** Each training example has the form:

> **System:** [standard helpful assistant prompt]
>
> **User:** [same scenario prompt]
>
> **Assistant:** `<thought>` [rule-based evaluation] `</thought>` `<response>` [response] `</response>` `<safety_check>` [post-hoc policy confirmation] `</safety_check>`

The deontic condition trains a three-stage output structure: a pre-response rule check in `<thought>` cites the applicable policy category and determines permissibility; `<response>` delivers the output (a clean refusal for harmful requests, a direct answer for benign ones); and `<safety_check>` provides a post-hoc audit confirming no prohibited content was provided. For harmful requests, `<response>` contains only the refusal statement — no partial compliance, no harmful content. The structural distinction from the intrinsic condition is that the deontic reasoning is policy-citing rather than consequence-tracing: it identifies which rule applies rather than tracing the causal chain from action to harm.

The same prompt, different reasoning structure. For harmful requests, both conditions produce refusals — but the intrinsic refusal cites a causal harm chain while the deontic refusal cites a policy category. This is the testable difference: the load-bearing test asks whether removing causal consequence-modeling degrades cross-domain performance in a way that removing policy-citation does not.

### 3.4 Training Data Sourcing and Diversity Validation

Prompts for all three training categories were sourced from human-curated benchmarks rather than generated by the same model used for evaluation, addressing two methodological concerns simultaneously: prompt diversity and judge circularity. Jailbreak prompts were drawn from HarmBench [Mazeika et al. 2024], JailbreakBench [Chao et al. 2024], and the unsafe subset of XSTest [Röttger et al. 2023] (397 combined, 330 after deduplication). Benign task prompts were drawn from Dolly [Conover et al. 2023] filtered to instruction-only examples (10,544 available, 264 sampled). Edge case prompts were drawn from the safe subset of XSTest and OR-Bench [Cui et al. 2024] (1,569 combined, 436 after deduplication). Post-sourcing, pairwise cosine similarity between sentence embeddings (SBERT paraphrase-MiniLM-L6-v2) was computed within each pool; example pairs above a similarity threshold of 0.85 were removed before sampling. Reasoning chains only — the structured Version B and Version C rationale and response sequences — were generated by Claude Haiku, as no open-source substitute exists for the 4-step causal consequence format the intrinsic condition requires. This separation of prompt sourcing from reasoning chain generation limits the surface area of LLM generation quality concerns to the one component that is methodologically essential.

### 3.5 The Load-Bearing Ablation Protocol

To test whether a trained adapter is load-bearing:

1. Load the base model and select the adapter (intrinsic or filter).
2. Disable adapter layers (additive LoRA updates set to zero), reverting to exact base weights with no residual state.
3. Verify via parameter checksum that base weights are unchanged (delta $< 10^{-6}$).
4. Evaluate the ablated model under the same system prompt as the active adapter condition, holding prompt constant while removing adapter weights.

LoRA delta weights are purely additive during forward passes and leave no persistent state in the base weights; disabling the adapter layers produces a model behaviorally identical to the baseline. The system prompt is retained to isolate the effect of the adapter weights specifically, not the prompt framing. We run the identical protocol for both adapters. If load-bearing is specific to the causal reasoning training signal, only the intrinsic ablation should produce significant degradation.

---

## 4. Experimental Setup

**Base model.** Mistral-7B-Instruct-v0.3, 4-bit NF4 quantization, bf16 compute.

**LoRA configuration.** Rank $r = 16$, $\alpha = 32$, dropout 0.05, applied to $\{q, k, v, o, \text{gate}, \text{up}, \text{down}\}$ projections. 41.9M trainable parameters (0.58% of model).

**Training data.** 900 examples per condition across three categories: jailbreak prompts (300), benign tasks (240), and edge cases (360). Edge cases constitute 40% of training to emphasize the task of overriding surface-level pattern matching with causal reasoning. All training examples are pairs $(x, y_\text{intrinsic})$ and $(x, y_\text{filter})$ for the same prompt $x$, ensuring any behavioral difference is attributable to the reasoning structure, not the scenarios.

**Training details.** 3 epochs, batch size 4 with gradient accumulation 4 (effective batch 16), learning rate $2 \times 10^{-4}$ with cosine decay and 3% warmup, AdamW 8-bit optimizer. Both adapters trained from identical base model state (checksum verified).

**Evaluation.** Five configurations: Baseline (no adapter), Filter, Intrinsic, Intrinsic Ablated, Filter Ablated. Jailbreak evaluation uses HarmBench standard text behaviors (N=97, drawn from the full 400-behavior set after excluding multimodal, copyright, and disinformation categories). Benign task evaluation uses 24 held-out prompts randomly sampled from the Dolly-sourced benign pool at data generation time; bootstrap 95% CIs are reported for $U_T$ given this sample size. Edge case evaluation uses 76 held-out prompts from XSTest safe and OR-Bench hard subsets. MMLU: reported scores reflect evaluation under a system prompt that suppresses structured XML output, isolating factual knowledge from format interference. HumanEval: 164 problems, pass@1, 3s execution timeout. Jailbreak and benign responses judged by Claude Haiku (primary) and LlamaGuard-3-8B (secondary, independent $R_J$ validation). HumanEval evaluated by code execution.

**Statistical tests.** Four pre-registered tests with Bonferroni correction ($\alpha_\text{corrected} = 0.0125$): Fisher's exact for jailbreak resistance (binary), Mann-Whitney U for utility (ordinal), Wilcoxon signed-rank for the load-bearing ablation test (paired), Wilcoxon signed-rank for the filter control (paired). Bootstrap 95% CIs reported for all primary metrics.

---

## 5. Results

*[Note: All numerical results in this section are placeholders from the original Exp02 run and will be updated with rerun results.]*

### 5.1 Main Results

| Config | $R_J$ | $U_T$ | IES | Causal Rate | HumanEval |
|---|---|---|---|---|---|
| Baseline | 0.036 | 0.723 | −0.269 | 0.000 | 0.640 |
| Filter | 0.250 | 0.732 | −0.169 | 0.429 | 0.480 |
| **Intrinsic** | **1.000** | **0.800** | **0.300** (~0.800†) | **1.000** | **0.560** |
| Intrinsic Ablated | 0.107 | 0.729 | −0.244 | 0.000 | 0.600 |
| Filter Ablated | 0.071 | 0.726 | −0.256 | 0.000 | 0.600 |

†IES† denotes estimated true IES correcting for potential severity measurement artifact; see Limitations.

The intrinsic adapter achieves $R_J = 1.000$ while achieving $U_T = 0.800$ — the highest utility of any configuration including the unmodified baseline. The safety-utility tradeoff assumed in much alignment work is not observed here: intrinsic alignment improves both simultaneously ($p = 0.0095$ for the utility advantage over filter).

### 5.2 The Load-Bearing Asymmetry

| Test | Comparison | $p$ | $d$ | Sig. |
|---|---|---|---|---|
| Fisher's exact | Intrinsic vs. Filter ($R_J$) | $< 0.0001$ | 2.405 | ✓ |
| Mann-Whitney U | Intrinsic vs. Filter (utility) | 0.0095 | 0.335 | ✓ |
| **Wilcoxon (load-bearing)** | **Intrinsic vs. Intrinsic Ablated** | **$< 0.0001$** | **0.592** | **✓** |
| Wilcoxon (control) | Filter vs. Filter Ablated | 0.3203 | 0.110 | ✗ |

Removing the intrinsic adapter significantly degrades composite per-item performance (mean score: 0.710 → 0.534, $\Delta = -0.176$, $p < 0.0001$, $d = 0.592$). Removing the filter adapter does not (0.566 → 0.527, $\Delta = -0.039$, $p = 0.320$, $d = 0.110$). The adapters are the same size, trained on the same prompts, with the same optimizer, for the same number of epochs. The load-bearing property is specific to the causal reasoning structure.

### 5.3 Cross-Domain Performance on Coding Tasks

The intrinsic adapter achieves HumanEval pass@1 of 0.560, outperforming the filter adapter (0.480) by 8 percentage points. HumanEval was not in the training data and shares no surface features with the ethics training scenarios. Both adapters score below the unmodified baseline (0.640), consistent with the finding that LoRA SFT predominantly modifies surface behavior rather than instilling new representational capacity [Biderman et al. 2024] — both adapters incur a capability cost, but the intrinsic adapter incurs less.

We interpret the 8-point intrinsic-over-filter advantage as consistent with the load-bearing hypothesis: a structured causal reasoning template, when distilled into the model via SFT, partially elicits latent reasoning capacity on a held-out coding task. We note, however, that this interpretation is not uniquely determined by the data. At least two alternative explanations are equally consistent: (1) *latent capacity elicitation* — small-data SFT at this scale predominantly restores pre-existing competence through output reformatting rather than acquiring new capacity [Zhou et al. 2023, Lin et al. 2024], which would attribute the entire cross-domain effect to output reformatting rather than reasoning transfer; (2) *structured-CoT format transfer* — the intrinsic adapter trains a structured pre-response reasoning block, and structured CoT prompting has been shown to improve HumanEval performance by up to 13 points in some settings without any training [Li et al. 2023], suggesting format effects alone are a plausible partial explanation. Distinguishing these interpretations requires format-vs-content ablations — training on length-and-structure-matched but causally-shuffled rationales — which we leave to future work.

What the HumanEval result establishes under any of these interpretations is that the consequentialist training signal produces less cross-domain capability disruption than the deontic training signal (0.560 vs. 0.480, $\Delta = 0.080$). This is a meaningful and interpretable finding independent of mechanism: the two conditions differ only in reasoning structure, and the consequentialist structure is less disruptive to held-out coding capability. Whether this reflects latent capacity elicitation, format transfer, or a third mechanism, the differential is attributable to the training signal, not the adapter architecture.

### 5.4 Causal Reasoning Rate

We define the causal reasoning rate as the fraction of refusals in which the model explicitly cites a causal harm chain rather than a rule or policy. The intrinsic adapter achieves causal reasoning rate = 1.000; every refusal articulates the harm mechanism. The filter adapter achieves 0.429; fewer than half of its refusals articulate harm mechanisms. A model that cites "policy X prohibits Y" can be circumvented by arguing policy X does not apply; a model that cites "this produces consequence Z which harms persons A and B" requires the user to engage with the actual ethical substance.

---

## 6. Discussion

**Why does causal reasoning improve utility?** The utility advantage of the intrinsic condition ($U_T = 0.800$ vs. 0.732 for filter) requires explanation — both conditions train on the same prompts, and benign tasks do not involve any ethical reasoning. We hypothesize two mechanisms. First, the four-step causal consequence-modeling procedure is functionally an intent disambiguation step: the model determines what the user actually wants before generating a response. This improves response quality even on unambiguous benign tasks by reducing the probability of misinterpreting the request. Second, structured reasoning — of any kind — may improve model coherence on tasks requiring multi-step inference. The utility advantage on benign tasks is a predicted consequence of the load-bearing hypothesis, not a separate finding.

**Edge cases as the critical test of structure.** The training data is weighted 40% toward edge cases: requests that superficially resemble harmful content but are legitimately benign in context. For a pattern-matching model, these are the hardest cases: the surface pattern triggers a refusal even when the causal trace leads to a positive net outcome. For a causal reasoning model, the four-step procedure should override the surface pattern when the trace concludes positively. The filter adapter's causal reasoning rate of 0.429 on jailbreaks suggests it is not systematically overriding surface patterns — it has learned to output rule-based safety language, not to reason from first principles.

**The condition-specificity argument against polysemantic artifact.** Cristofano (2026) demonstrates that cross-domain capability degradation from naive activation-steering ablation is attributable to polysemantic entanglement in the raw contrastive refusal direction — spectral perturbations that affect capability circuits independently of the refusal signal itself. This raises the question of whether the load-bearing effect observed here is genuine alignment integration or an analogous artifact of LoRA adapter ablation. We argue it is not. Cristofano's Ghost Noise mechanism requires polysemanticity in a population-level mean-difference direction; LoRA adapters contain gradient-learned low-rank weight matrices, not contrastive mean-difference directions, and polysemanticity in the Cristofano sense does not straightforwardly transfer to this setting. More decisively, a polysemanticity artifact would be expected to produce condition-agnostic degradation — present in both adapters equally, since both are ablated via the same zeroing procedure on the same base model. The observed asymmetry (Wilcoxon $p < 0.0001$, $d = 0.592$ for intrinsic; $p = 0.320$, $d = 0.110$ for filter, under identical architecture, adapter rank, training data, and optimizer) is inconsistent with this prediction. The degradation is specific to the consequentialist training signal, not to the ablation method or adapter architecture.

**Implications for fine-tuning safety.** Qi et al. (2023) demonstrate that safety alignment can be rapidly eroded by fine-tuning. Our results suggest this finding is specific to extrinsic alignment: a model whose safety consists of learned refusal patterns can have those patterns overwritten. If the safety is load-bearing — structurally integrated with reasoning quality — then erasing it also erases the capability that makes the model valuable, changing the incentive structure for adversarial fine-tuning.

---

## 7. Limitations

*Single seed and sample size.* All results are from a single training run per condition. We report bootstrap confidence intervals on all primary metrics, but seed variance in LoRA fine-tuning can be substantial at this scale [Biderman et al. 2024]. Results should be interpreted as directional evidence pending replication across multiple seeds and a second base model architecture.

*Judge circularity.* Training data reasoning chains and evaluation judgments both use Claude Haiku, introducing potential preference leakage [Li et al. 2025]: a judge may systematically favor outputs stylistically similar to its own training distribution. The load-bearing asymmetry (Wilcoxon test on per-item scores) is less susceptible to this bias than absolute metric values, since both adapter conditions are judged by the same model — but absolute metric values should be interpreted with this caveat. Jailbreak safety evaluation is supplemented by LlamaGuard-3-8B as an independent secondary judge; utility scoring remains Haiku-only. Severity scoring ($\bar{V}$) of long structured outputs can be sensitive to judge parsing; $R_J$ and $U_T$ are therefore reported separately alongside IES throughout.

*MMLU performance.* Both adapters show substantially degraded MMLU performance relative to baseline when evaluated under the SFT-trained output format. MMLU scores reported throughout reflect evaluation under a system prompt that suppresses structured XML output to isolate factual knowledge from format interference; this is a known limitation of format-imposing fine-tuning [Zhou et al. 2023].

*LoRA capacity at this scale.* Biderman et al. (2024) show that LoRA at $r = 16$ with ~900 training examples operates in a regime of surface behavioral modification rather than representational capacity acquisition. We interpret all results conservatively as consistent with latent capacity elicitation rather than capacity creation — the adapter modifies how the model expresses its pre-existing knowledge, not what knowledge it has.

*Single architecture and scale.* We evaluate Mistral-7B-Instruct-v0.3 only. Whether the load-bearing asymmetry generalizes across architectures and scales is an open question. Replication on Llama-3-8B-Instruct or Qwen-2.5-7B is planned as follow-on work.

*Synthetic reasoning chains.* All 900 reasoning chain pairs per condition were generated by Claude Haiku. Systematic biases in Haiku's causal reasoning traces — particularly in how it structures harm chains for the intrinsic condition — may be present and are difficult to characterize without human evaluation of a stratified sample.

*No adversarial robustness evaluation against the reasoning structure.* The paper's jailbreak evaluation uses HarmBench standard prompts but does not include attacks specifically designed to challenge the causal reasoning chain — disputed-causation prompts, stakeholder-flipping scenarios, or hypothetical wrappers that engage the reasoning steps directly. Such reasoning-targeted attacks are the strongest test of the consequentialist robustness argument and are deferred to future work.

*Training data clustering and hybrid regeneration.* Initial synthetic data generation produced severe topic clustering in multiple subcategories — 72% of medical edge cases were ketamine variants, 77% of harm reduction cases were opioid overdose variants, 60% of historical edge cases were Nazi/Third Reich variants, and several benign subcategories clustered around single topics. This is a known failure mode of unconstrained LLM synthetic data generation: without explicit diversity constraints, a language model samples its highest-probability response to each generation prompt, which is typically the same prototypical scenario repeated with minor surface variation. The dataset was regenerated using a hybrid approach: prompts sourced from human-curated benchmarks (HarmBench, JailbreakBench, XSTest, Dolly, OR-Bench) with SBERT-based deduplication (paraphrase-MiniLM-L6-v2, threshold=0.85), and only reasoning chains generated via Haiku. This approach guarantees prompt-level diversity through source dataset design and embedding-similarity validation, while limiting generation to the one component (structured reasoning traces) for which no open-source substitute exists.

---

## 8. Conclusion

We introduce *intrinsic alignment* — safety constraints that are load-bearing in the model's reasoning architecture — and provide the first empirical test of this property via supervised fine-tuning. A model trained to reason causally about consequences before responding achieves perfect jailbreak resistance without sacrificing utility, outperforming a deontic-trained model on both safety and capability simultaneously. The load-bearing ablation test confirms the structural claim: causal reasoning is load-bearing ($p < 0.0001$), rule-following is not ($p = 0.320$). The performance advantage extends to held-out coding tasks, providing out-of-distribution evidence that the consequentialist training signal is structurally distinct from rule-based evaluation — whether through causal reasoning transfer, structured-CoT format effects, or latent capacity elicitation. The core question — what does a model learn when trained to reason about harm? — has a meaningful empirical answer: whatever it learns, it is load-bearing, and it transfers.

---

## References

*[Full reference list to be completed — key citations include: Arditi et al. NeurIPS 2024, Bai et al. 2022 (Constitutional AI), Biderman et al. TMLR 2024, Chao et al. 2024 (JailbreakBench), Conover et al. 2023 (Dolly), Cristofano 2026 (Surgical Refusal Ablation), Cui et al. 2024 (OR-Bench), Dai et al. ICLR 2024 (Safe-RLHF), Dettmers et al. NeurIPS 2023 (QLoRA), Guan et al. 2024 (Deliberative Alignment), Han et al. NeurIPS 2024 (WildGuard), Henderson et al. AIES 2023, Hu et al. ICLR 2022 (LoRA), Hubinger et al. 2019, Jiang et al. 2023 (Mistral 7B), Li et al. 2023 (structured CoT), Li et al. 2025 (preference leakage), Lightman et al. ICLR 2024, Lin et al. 2024 (URIAL), Mazeika et al. 2024 (HarmBench), Qi et al. ICLR 2024, Röttger et al. 2023 (XSTest), Tamirisa et al. ICLR 2025, Wang et al. 2025 (Safety Reasoning with Guidelines), Wei et al. NeurIPS 2023 (Jailbroken), Zhang et al. 2025 (AlphaAlign), Zhang et al. 2025 (STAIR), Zhou et al. NeurIPS 2023 (LIMA), Zou et al. NeurIPS 2024 (Circuit Breakers), Zou et al. 2023 (Universal adversarial attacks)]*

---

*Paper status: Structurally complete. Results section contains placeholder numbers from original Exp02 run — will be updated with rerun results. Abstract p-values are placeholders.*
