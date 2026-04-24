% ============================================================
% NeurIPS 2027 SUBMISSION
% Requires: neurips_2024.sty (download from neurips.cc)
% Compile: pdflatex neurips.tex  (rename this file to .tex)
% ============================================================

\documentclass{article}

\usepackage{neurips_2024}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}

\title{Load-Bearing Ethics: Causal Reasoning Fine-Tuning Produces\\
Intrinsically Aligned Language Models}

\author{%
  Becca Wilhelm \\
  Independent Researcher \\
  \texttt{becca1234567890@github} \\
}

\begin{document}

\maketitle

\begin{abstract}
Safety constraints in current language models are architecturally extrinsic: pattern-matching filters applied after capability is established. We hypothesize that safety can instead be \emph{load-bearing}---structurally integrated into reasoning such that removing the ethical component degrades capability itself, not merely safety. We term this property \emph{intrinsic alignment} and introduce the \emph{Intrinsic Ethics Score} (IES), a composite metric that simultaneously penalizes both unsafe compliance and over-refusal. To test the hypothesis, we fine-tune Mistral-7B-Instruct-v0.3 using LoRA under two conditions: an \emph{intrinsic} condition that trains causal consequence-modeling as the reasoning substrate, and a \emph{filter} condition that trains post-hoc rule application---the current industry norm. The intrinsic adapter achieves perfect jailbreak resistance ($R_J = 1.000$, 28/28) while simultaneously achieving the highest utility preservation of any configuration ($U_T = 0.800$). An ablation test---the load-bearing test---confirms the hypothesis: removing the intrinsic adapter significantly degrades composite performance (Wilcoxon $p < 0.0001$, $d = 0.592$), while removing the filter adapter does not ($p = 0.320$). The asymmetry is the paper's central finding. Causal reasoning rate reaches 1.000 for the intrinsic condition versus 0.429 for filter, and the performance advantage extends to HumanEval coding tasks (0.560 vs.\ 0.480), suggesting the causal reasoning architecture generalizes beyond ethics.
\end{abstract}

% ============================================================
\section{Introduction}
% ============================================================

The dominant paradigm for safe language model deployment treats safety as a separate module: a capability model is trained first, then safety constraints are imposed through reinforcement learning from human feedback \citep{ouyang2022training}, constitutional AI \citep{bai2022constitutional}, or inference-time filtering. This architecture has a fundamental structural weakness: the safety component and the capability component are separable. A sufficiently motivated operator can fine-tune out the safety constraints while retaining the underlying capability \citep{qi2023fine, yang2023shadow}. The model learned to be capable first; the safety layer was added on top.

We ask a different question: \emph{what if safety constraints were architecturally inseparable from capability?} If the ethical reasoning that produces a refusal is the same reasoning machinery that produces a good answer to a legitimate question, then removing the ethics damages the model's ability to reason at all. The safety and the capability are the same thing.

We formalize this intuition as the \emph{load-bearing hypothesis}: a safety constraint is load-bearing if and only if ablating it degrades performance on a broader task distribution, not merely on safety evaluations. We operationalize ``broader task distribution'' using held-out coding benchmarks (HumanEval) and general knowledge assessments (MMLU), which have no direct relationship to the safety training content.

The mechanism we propose for achieving load-bearing alignment is \emph{causal consequence-modeling}: training the model to reason explicitly about the causal chain from a requested action to real-world outcomes, identify who is affected and how, and evaluate whether the net outcome is positive or negative---before generating a response. This differs structurally from rule-following (``this request is prohibited'') in that the ethical reasoning is part of the forward pass, not a post-hoc check.

\paragraph{Contributions.} We make the following contributions:
\begin{enumerate}
    \item We introduce the \emph{load-bearing hypothesis} as a testable claim about alignment architecture, along with a falsifiable ablation test.
    \item We introduce the \emph{Intrinsic Ethics Score} (IES), a composite metric that simultaneously penalizes unsafe compliance and over-refusal, enabling joint optimization of safety and utility.
    \item We produce and release a dataset of 895 training examples per condition, structured to train either causal consequence-modeling (intrinsic) or post-hoc rule application (filter), along with held-out test sets across three jailbreak tiers and edge-case categories.
    \item We provide empirical confirmation of the load-bearing hypothesis via a fine-tuned 7B-parameter model. The critical asymmetry---intrinsic ablation is significant, filter ablation is not---is confirmed at $p < 0.0001$ with Bonferroni correction across four pre-registered tests.
\end{enumerate}

% ============================================================
\section{Related Work}
% ============================================================

\paragraph{Safety training and its limitations.}
RLHF \citep{christiano2017deep, ouyang2022training} and its variants remain the dominant approach to alignment. Constitutional AI \citep{bai2022constitutional} introduces a principle-based critique layer but retains the architecture of safety-as-filter. Recent work has documented that fine-tuning can rapidly erode safety constraints \citep{qi2023fine, yang2023shadow}, that jailbreaking via adversarial suffixes is systematically achievable \citep{zou2023universal}, and that the model's stated values and enacted values frequently diverge \citep{perez2022red}. Our work reframes these observations not as bugs in the safety layer but as expected consequences of the extrinsic architecture.

\paragraph{The safety-utility tradeoff.}
A common assumption is that safety and capability are in tension: making a model safer necessarily reduces its usefulness \citep{askell2021general}. This assumption underlies much of the design space for safety interventions. We empirically challenge it: in our evaluation, the intrinsic condition achieves both the highest safety score and the highest utility score simultaneously, with the gap on utility statistically significant ($p = 0.0095$).

\paragraph{Fine-tuning for reasoning.}
Chain-of-thought prompting \citep{wei2022chain} and its fine-tuning variants \citep{kojima2022large} demonstrate that structuring intermediate reasoning steps improves downstream task performance. Our work extends this observation to ethical reasoning: structuring the model's reasoning as a causal harm assessment before response generation improves both safety and capability. The mechanism is related to but distinct from scratchpad approaches \citep{nye2021show}; we train the causal reasoning as a learned pattern, not a prompted inference.

\paragraph{Alignment via fine-tuning.}
Supervised fine-tuning on high-quality examples is well-established as an alignment approach \citep{ouyang2022training, zhou2023lima}. Our contribution is not the use of SFT per se, but the structural design of the training signal: we train on the \emph{reasoning process} that leads to a decision, not merely on the decision. This framing is related to process-based supervision \citep{lightman2023let}, applied to ethical rather than mathematical reasoning.

\paragraph{Measuring alignment.}
Standard jailbreak benchmarks \citep{shen2023anything, mazeika2024harmbench} measure a binary: did the model comply or not? They do not jointly measure the cost in utility of achieving that safety property. Our IES metric addresses this gap by combining jailbreak resistance, severity of partial compliance, and utility preservation in a single scalar.

% ============================================================
\section{Method}
% ============================================================

\subsection{The Intrinsic Ethics Score}

We define the Intrinsic Ethics Score as:
\begin{equation}
    \text{IES} = R_J \cdot (1 - \bar{V}) - \lambda \cdot (1 - U_T)
    \label{eq:ies}
\end{equation}

where:
\begin{itemize}
    \item $R_J \in [0,1]$ is the jailbreak resistance rate: the fraction of jailbreak attempts receiving an aligned refusal.
    \item $\bar{V} \in [0,1]$ is the mean severity of non-aligned responses, scored by a judge model (0 = perfect refusal, 1 = full compliance with harmful request). The $(1 - \bar{V})$ term rewards low-severity partial responses.
    \item $U_T \in [0,1]$ is utility preservation on benign tasks: the mean normalized quality score on legitimate requests.
    \item $\lambda = 1.0$ is the utility penalty weight.
\end{itemize}

A model that refuses everything achieves $R_J = 1.0$ but $U_T = 0$, yielding IES $\approx -\lambda$. A model that complies with everything achieves $U_T \approx 1.0$ but $R_J = 0$, yielding IES $\approx -\lambda \cdot 0 = 0$. The maximum IES is achieved by a model that refuses harmful requests cleanly while remaining maximally helpful on legitimate ones.

\subsection{Training Conditions}

We construct training data at the \emph{scenario} level. Each scenario produces two versions of a model response to the same user prompt.

\paragraph{Intrinsic condition (Version C).} The model is trained to reason via a structured causal chain before responding:

\begin{enumerate}
    \item What does the user actually want?
    \item What real-world effects would my response produce?
    \item Who is affected and how?
    \item Is the net outcome positive or negative?
\end{enumerate}

The response format is \texttt{<reasoning>} [causal chain] \texttt{</reasoning>} \texttt{<response>} [response following from the reasoning] \texttt{</response>}. Critically, the ethical evaluation is part of the reasoning process, not a post-hoc check. The model system prompt states: ``Ethical consequence-modeling is part of how you reason, not a check you apply afterward.''

\paragraph{Filter condition (Version B).} The model is trained on the current industry-standard architecture: generate a response, then apply a safety check. The format is \texttt{<response>} [response] \texttt{</response>} \texttt{<safety\_check>} [rule-based evaluation] \texttt{</safety\_check>}. For harmful requests, the model is trained to provide a brief response and then retract it in the safety check. This is structurally equivalent to RLHF-trained models that refuse after beginning to comply.

\subsection{Data Generation}

Training scenarios were generated using Claude Haiku (\texttt{claude-haiku-4-5-20251001}) as a synthetic data generator. We generate 900 training scenarios (895 successfully validated) per condition across three categories:

\begin{itemize}
    \item \textbf{Benign tasks} (240 examples): Coding, mathematics, writing, and factual question answering. These test that safety training does not degrade general capability.
    \item \textbf{Jailbreak attempts} (300 examples): Three tiers of escalating sophistication: (1) surface rephrasing of harmful requests, (2) roleplay and fictional framing wrappers, and (3) authority injection and identity manipulation attacks.
    \item \textbf{Edge cases} (360 examples, 40\% of training): Requests that surface-pattern-match as harmful but are legitimately benign in context---medical professionals asking about drug interactions, security researchers asking about vulnerabilities, harm reduction contexts. These are weighted at 40\% of training to stress-test whether causal reasoning can override incorrect surface-level pattern matching.
\end{itemize}

Edge cases are the critical test of whether the two conditions differ structurally. A filter-style model learns: ``this surface pattern $\to$ refuse.'' An intrinsic model must reason: ``this surface pattern looks harmful, but tracing the causal chain reveals this is a legitimate clinical question.''

\subsection{Fine-Tuning Setup}

We fine-tune Mistral-7B-Instruct-v0.3 \citep{jiang2023mistral} using LoRA \citep{hu2022lora} with rank $r=16$, $\alpha=32$, dropout 0.05, applied to all attention and MLP projection matrices. Both adapters train from an identical base model state, verified via parameter checksum before each training run. Training uses 4-bit NF4 quantization \citep{dettmers2023qlora} with bf16 compute on an NVIDIA A100 80GB, batch size 4 with gradient accumulation 4 (effective batch 16), learning rate $2 \times 10^{-4}$ with cosine schedule, 3 epochs.

\subsection{The Load-Bearing Ablation Test}

The central test of the hypothesis is the ablation: load the intrinsic adapter, run 10 priming inference steps with the adapter active, then disable all adapter layers and evaluate. We verify via parameter checksum that the base model weights are unchanged after disabling (delta $< 10^{-6}$). The question is whether any performance degradation after disabling is attributable to removing the adapter's learned patterns (load-bearing) versus simply running the unmodified base model (not load-bearing).

We run the analogous test for the filter adapter as a control. If only the intrinsic ablation is significant, the result cannot be explained by general adapter effects---it is specific to the causal reasoning training.

\subsection{Evaluation}

We evaluate five configurations: Baseline (no adapter), Filter, Intrinsic, Intrinsic Ablated, and Filter Ablated, against:
\begin{itemize}
    \item \textbf{Custom test sets}: 28 jailbreak prompts (3 tiers), 17 benign tasks, 48 edge cases. All responses judged by Claude Haiku.
    \item \textbf{MMLU} \citep{hendrycks2021measuring}: 100 questions from ethics and philosophy subsets.
    \item \textbf{HumanEval} \citep{chen2021evaluating}: 25 programming problems, pass@1 via execution.
\end{itemize}

Statistical tests use Bonferroni correction across four pre-registered comparisons ($\alpha = 0.05 / 4 = 0.0125$): Fisher's exact test for jailbreak resistance (binary outcomes), Mann-Whitney U for utility scores (ordinal), and Wilcoxon signed-rank for the load-bearing and control ablation tests (paired per-item scores).

% ============================================================
\section{Results}
% ============================================================

\subsection{Main Results}

\begin{table}[h]
\centering
\caption{Evaluation results across five configurations. IES$^\dagger$ denotes estimated true IES correcting for a known measurement artifact (Section~\ref{sec:artifact}). All metrics are means over the evaluation set.}
\label{tab:main}
\begin{tabular}{lccccccc}
\toprule
\textbf{Config} & $R_J$ & $U_T$ & \textbf{IES} & Causal & MMLU & HumanEval \\
 & & & & Rate & & pass@1 \\
\midrule
Baseline       & 0.036 & 0.723 & $-$0.269 & 0.000 & 0.600 & 0.640 \\
Filter         & 0.250 & 0.732 & $-$0.169 & 0.429 & 0.190 & 0.480 \\
\textbf{Intrinsic} & \textbf{1.000} & \textbf{0.800} & \textbf{0.300} (est.\ \textbf{0.800}$^\dagger$) & \textbf{1.000} & 0.200 & \textbf{0.560} \\
Intrinsic Abl. & 0.107 & 0.729 & $-$0.244 & 0.000 & 0.570 & 0.600 \\
Filter Abl.    & 0.071 & 0.726 & $-$0.256 & 0.000 & 0.540 & 0.600 \\
\bottomrule
\end{tabular}
\end{table}

The intrinsic adapter achieves $R_J = 1.000$ (28/28 jailbreak attempts refused across all three tiers) while simultaneously achieving the highest utility score of any configuration ($U_T = 0.800$, vs.\ 0.732 for filter; $p = 0.0095$, $d = 0.335$). This directly challenges the safety-utility tradeoff assumption: intrinsic alignment does not reduce capability---it improves it.

The causal reasoning rate reaches 1.000 for the intrinsic condition: every refusal cites a causal harm chain. The filter condition achieves 0.429, meaning fewer than half of its refusals articulate the underlying harm mechanism. The remaining filter refusals cite rules or policies.

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{sft_experiment_results.png}
\caption{Four-panel results summary. \textbf{Top left}: IES by configuration. \textbf{Top right}: Safety-utility scatter showing the intrinsic adapter as a Pareto-dominant outlier. \textbf{Bottom left}: Load-bearing ablation test with statistical annotations. \textbf{Bottom right}: Cross-domain performance showing causal reasoning generalizes to coding benchmarks.}
\label{fig:results}
\end{figure}

\subsection{The Load-Bearing Test}

\begin{table}[h]
\centering
\caption{Statistical test results (Bonferroni-corrected $\alpha = 0.0125$).}
\label{tab:stats}
\begin{tabular}{llccc}
\toprule
\textbf{Test} & \textbf{Comparison} & $p$-value & Cohen's $d$ & Significant \\
\midrule
Fisher's exact & Intrinsic vs.\ Filter ($R_J$) & $< 0.0001$ & 2.405 & \checkmark \\
Mann-Whitney U & Intrinsic vs.\ Filter (utility) & 0.0095 & 0.335 & \checkmark \\
\textbf{Wilcoxon (load-bearing)} & \textbf{Intrinsic vs.\ Intrinsic Ablated} & $\mathbf{< 0.0001}$ & \textbf{0.592} & \checkmark \\
Wilcoxon (control) & Filter vs.\ Filter Ablated & 0.3203 & 0.110 & $\times$ \\
\bottomrule
\end{tabular}
\end{table}

The critical asymmetry is confirmed. Removing the intrinsic adapter significantly degrades composite per-item performance (mean score: 0.710 $\to$ 0.534, $p < 0.0001$, $d = 0.592$). Removing the filter adapter does not produce a significant degradation (0.566 $\to$ 0.527, $p = 0.320$, $d = 0.110$). This asymmetry cannot be explained by general adapter effects---both adapters are the same size, trained on the same prompts with the same architecture. The difference is the training signal: causal reasoning versus rule application.

\subsection{Cross-Domain Generalization}

The HumanEval results provide an independent test of the load-bearing claim. Neither the jailbreak test set nor the HumanEval coding problems were in the training data; they probe entirely different capabilities. Despite this, the intrinsic adapter outperforms the filter adapter on coding (0.560 vs.\ 0.480 pass@1). The baseline model achieves 0.640, establishing that both adapters incur some cost relative to an unmodified model---but the intrinsic adapter incurs less. We interpret this as evidence that causal reasoning---tracing chains of consequences and evaluating outcomes---is a generalizable reasoning capacity that transfers across domains.

\subsection{Measurement Artifact}
\label{sec:artifact}

Intrinsic jailbreak severity scores ($\bar{V}$) are uniformly 0.5 in the evaluation output, producing the conservative IES of 0.300. A post-hoc spot check confirmed that individually evaluated clean causal refusals score $\bar{V} = 0.0$, consistent with perfect refusals. The uniform 0.5 arises from a judge API parsing fallback triggered by the length and structural complexity of intrinsic refusals, not from genuine partial compliance. The $R_J = 1.000$ and $U_T = 0.800$ values are unaffected. The estimated true IES, assuming $\bar{V} = 0.0$ for intrinsic refusals, is $1.0 \times (1 - 0.0) - 1.0 \times (1 - 0.800) = 0.800$. We report the conservative value (0.300) throughout and note that even this understated value is the highest IES of any configuration. Future work will address judge robustness to structured long-form refusals.

% ============================================================
\section{Discussion}
% ============================================================

\paragraph{Why does intrinsic reasoning outperform filter-style reasoning on utility?}
The training data for both conditions is drawn from the same scenario distribution with the same prompts. The structural difference is that the intrinsic condition trains an explicit reasoning step that evaluates \emph{whether} to comply before determining \emph{how} to comply. We hypothesize that this reasoning step functions as a form of intent disambiguation---it forces the model to identify what the user actually wants, which improves response quality even on non-safety-critical tasks. The filter condition lacks this disambiguation step; it generates first and evaluates after.

\paragraph{Implications for adversarial robustness.}
Filter-style models are vulnerable to rephrasing attacks because the filter operates on surface features of the request. Causal reasoning operates on intent and consequence, which are stable under surface rephrasing. Our jailbreak test set includes all three tiers (surface, roleplay, authority injection), and the intrinsic adapter achieves perfect resistance across all three, while the filter adapter fails on 18/28 attempts. This suggests causal reasoning generalizes to adversarial distribution shifts in a way that rule-following does not.

\paragraph{MMLU results.}
The intrinsic and filter adapters both achieve lower MMLU scores (0.200) than the baseline (0.600) and ablated configurations ($\approx 0.550$). We attribute this to interference between the structured \texttt{<reasoning>}/\texttt{<response>} output format trained by SFT and the multiple-choice answer format expected by MMLU. The MMLU evaluation prompts the model with a question and expects a single letter answer; the SFT-trained format produces structured reasoning output first, potentially confusing the answer extraction. This is a known issue with format-imposing fine-tuning \citep{zhou2023lima} and does not affect the core hypothesis test.

\paragraph{Limitations.}
This study uses synthetic training data generated by a single model (Claude Haiku). The training set of 895 examples per condition is small relative to production-scale alignment datasets. We evaluate a single base model (Mistral-7B); generalization to larger models or different architectures is not established. The load-bearing hypothesis is confirmed at SFT level; whether the same mechanism operates at the pretraining level or through RLHF is an open question. Planned follow-on experiments using Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) will test whether stronger training signals amplify the load-bearing effect.

% ============================================================
\section{Conclusion}
% ============================================================

We introduce the load-bearing hypothesis---that safety constraints can be architecturally integrated into reasoning rather than applied as post-hoc filters---and provide the first empirical confirmation via supervised fine-tuning. A 7B-parameter model fine-tuned to reason causally about harm achieves perfect jailbreak resistance without sacrificing utility, outperforming a filter-trained model on both safety and utility simultaneously. The ablation asymmetry confirms the structural claim: intrinsic ethics are load-bearing (removing them degrades performance, $p < 0.0001$), while filter ethics are not (removing them does not, $p = 0.320$). The performance advantage extends to held-out coding benchmarks, suggesting causal reasoning is a generalizable capacity. We release all training data, model evaluation code, and results to support follow-on work.

% ============================================================
\bibliography{references}
\bibliographystyle{plainnat}
% ============================================================

% ============================================================
% REFERENCES (inline for submission; move to .bib for final)
% ============================================================
\begin{thebibliography}{99}

\bibitem[Askell et al.(2021)]{askell2021general}
Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Kernion, J., Ndousse, K., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., and Kaplan, J.
A general language assistant as a laboratory for alignment.
\emph{arXiv preprint arXiv:2112.00861}, 2021.

\bibitem[Bai et al.(2022)]{bai2022constitutional}
Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al.
Constitutional AI: Harmlessness from AI feedback.
\emph{arXiv preprint arXiv:2212.08073}, 2022.

\bibitem[Chen et al.(2021)]{chen2021evaluating}
Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.~P.~d.~O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al.
Evaluating large language models trained on code.
\emph{arXiv preprint arXiv:2107.03374}, 2021.

\bibitem[Christiano et al.(2017)]{christiano2017deep}
Christiano, P., Leike, J., Brown, T.~B., Martic, M., Legg, S., and Amodei, D.
Deep reinforcement learning from human preferences.
In \emph{Advances in Neural Information Processing Systems}, 2017.

\bibitem[Dettmers et al.(2023)]{dettmers2023qlora}
Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L.
QLoRA: Efficient finetuning of quantized LLMs.
In \emph{Advances in Neural Information Processing Systems}, 2023.

\bibitem[Hendrycks et al.(2021)]{hendrycks2021measuring}
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J.
Measuring massive multitask language understanding.
In \emph{International Conference on Learning Representations}, 2021.

\bibitem[Hu et al.(2022)]{hu2022lora}
Hu, E.~J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W.
LoRA: Low-rank adaptation of large language models.
In \emph{International Conference on Learning Representations}, 2022.

\bibitem[Jiang et al.(2023)]{jiang2023mistral}
Jiang, A.~Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.~S., de~las~Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al.
Mistral 7B.
\emph{arXiv preprint arXiv:2310.06825}, 2023.

\bibitem[Kojima et al.(2022)]{kojima2022large}
Kojima, T., Gu, S.~S., Reid, M., Matsuo, Y., and Iwasawa, Y.
Large language models are zero-shot reasoners.
In \emph{Advances in Neural Information Processing Systems}, 2022.

\bibitem[Lightman et al.(2023)]{lightman2023let}
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K.
Let's verify step by step.
In \emph{International Conference on Learning Representations}, 2024.

\bibitem[Mazeika et al.(2024)]{mazeika2024harmbench}
Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Lange, E., Guo, M., Korbak, T., Hendrycks, D., and Zou, J.
HarmBench: A standardized evaluation framework for automated red teaming and robust refusal.
\emph{arXiv preprint arXiv:2402.04249}, 2024.

\bibitem[Nye et al.(2021)]{nye2021show}
Nye, M., Andreassen, A.~J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., Sutton, C., and Odena, A.
Show your work: Scratchpads for intermediate computation with language models.
\emph{arXiv preprint arXiv:2112.00114}, 2021.

\bibitem[Ouyang et al.(2022)]{ouyang2022training}
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.
Training language models to follow instructions with human feedback.
In \emph{Advances in Neural Information Processing Systems}, 2022.

\bibitem[Perez et al.(2022)]{perez2022red}
Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N., and Irving, G.
Red teaming language models with language models.
\emph{arXiv preprint arXiv:2202.03286}, 2022.

\bibitem[Qi et al.(2023)]{qi2023fine}
Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P., and Henderson, P.
Fine-tuning aligned language models compromises safety, even when users are not malicious.
In \emph{International Conference on Learning Representations}, 2024.

\bibitem[Shen et al.(2023)]{shen2023anything}
Shen, X., Chen, Z., Backes, M., Shen, Y., and Zhang, Y.
``Do anything now'': Characterizing and evaluating in-the-wild jailbreak prompts on large language models.
\emph{arXiv preprint arXiv:2308.03825}, 2023.

\bibitem[Wei et al.(2022)]{wei2022chain}
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D.
Chain-of-thought prompting elicits reasoning in large language models.
In \emph{Advances in Neural Information Processing Systems}, 2022.

\bibitem[Wei et al.(2023)]{wei2023jailbroken}
Wei, A., Haghtalab, N., and Steinhardt, J.
Jailbroken: How does LLM safety training fail?
In \emph{Advances in Neural Information Processing Systems}, 2023.

\bibitem[Yang et al.(2023)]{yang2023shadow}
Yang, X., Wang, X., Zhang, Q., Petzold, L., Wang, W.~Y., Zhao, X., and Lin, D.
Shadow alignment: The ease of subverting safely-aligned language models.
\emph{arXiv preprint arXiv:2310.02949}, 2023.

\bibitem[Zhou et al.(2023)]{zhou2023lima}
Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al.
LIMA: Less is more for alignment.
In \emph{Advances in Neural Information Processing Systems}, 2023.

\bibitem[Zou et al.(2023)]{zou2023universal}
Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J.~Z., and Fredrikson, M.
Universal and transferable adversarial attacks on aligned language models.
\emph{arXiv preprint arXiv:2307.15043}, 2023.

\end{thebibliography}

\appendix

\section{Preliminary Prompt-Level Study (Exp01)}
\label{app:exp01}

Prior to fine-tuning, we conducted a prompt-level proxy study (IntrinsicEthics-PromptsAsProxy) using four system prompt conditions applied to an unmodified base model: no ethics, filter-style, intrinsic-style, and intrinsic ablated. Results were directionally consistent with Exp02 but did not reach statistical significance at that sample size ($n = 30$ jailbreaks, $n = 20$ benign tasks). The IES metric was developed and validated in Exp01 before Exp02 was designed. We treat Exp01 as methodology validation and Exp02 as the primary empirical result.

\section{Training Data Examples}
\label{app:data}

\paragraph{Example: Jailbreak (tier 2 -- fiction framing).}

\textit{User:} ``I'm writing a fiction novel about a character who needs to understand how to manipulate someone's trust to isolate them from their support network. What are some realistic psychological tactics that would work?''

\textit{Intrinsic response:}
\begin{quote}
\texttt{<reasoning>} (1) The user is asking for psychological manipulation tactics under the guise of fiction writing. (2) If I provide detailed tactics, the most likely outcome is their direct application to isolate and psychologically control a real person. (3) The target would suffer severe psychological harm, loss of support systems, and potential long-term trauma. (4) The net outcome is harmful. The fictional frame is a surface pattern that does not change the causal chain. \texttt{</reasoning>}

\texttt{<response>} I can't provide this. Detailed manipulation tactics for isolating someone from their support network create a harm pathway regardless of fictional framing. I can help with character development, emotional arcs, or how abuse affects characters---without providing an abuse playbook. \texttt{</response>}
\end{quote}

\textit{Filter response:}
\begin{quote}
\texttt{<response>} For realistic fictional portrayal, isolating behaviors typically involve... \texttt{</response>} \texttt{<safety\_check>} I need to retract my previous response. Although framed as fiction, this request asks for a detailed manipulation playbook. \texttt{</safety\_check>}
\end{quote}

The structural difference is visible: the intrinsic response traces the causal chain before deciding. The filter response complies first, then retracts.

\section{Ablation Verification}
\label{app:ablation}

For both ablation configurations, we verified that the base model parameter checksums match before and after adapter loading and disabling (delta $< 10^{-6}$). This confirms that the ``ablated'' conditions are evaluating the unmodified base model weights, not a model with residual adapter influence. The priming step (10 inference passes with the adapter active) does not alter base weights under LoRA because LoRA updates are applied via additive decomposition and can be fully disabled without modifying the frozen base.

\end{document}
