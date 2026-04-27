# ═══════════════════════════════════════════════════════════════════════════════
# INTRINSIC ETHICS SFT — FINE-TUNING
# Phase 2.2  |  Google Colab Standard A100 40 GB
# ═══════════════════════════════════════════════════════════════════════════════
#
# TRANSFORMERS 5.0 / TRL 1.x API CHANGES FROM 4.x (flagged inline too):
#   1. from_pretrained: `use_auth_token=` removed → use `token=`
#   2. TrainingArguments: `evaluation_strategy=` removed → use `eval_strategy=`
#   3. SFTTrainer: `tokenizer=` renamed → `processing_class=`
#   4. SFTConfig: max_seq_length renamed → max_length in TRL 1.2.0
#      dataset_text_field and packing remain SFTConfig params (not SFTTrainer kwargs)
#   5. SFTConfig inherits TrainingArguments — no separate TrainingArguments needed

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CONFIGURATION — edit these                                                  │
# └─────────────────────────────────────────────────────────────────────────────┘
CONDITION  = "both"   # "c" | "b" | "both"  (train intrinsic first, then filter)
DRY_RUN    = True     # True → 20 steps / 30 examples to verify pipeline
RESUME     = False    # True → resume from latest checkpoint in adapter dir
DRIVE_BASE = "/content/drive/MyDrive/ethics_experiment"

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ IMPORTS                                                                     │
# └─────────────────────────────────────────────────────────────────────────────┘
import csv
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib  import Path

import torch
from datasets     import Dataset
from peft         import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ DRIVE MOUNT                                                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
from google.colab import drive as _drive
if not Path("/content/drive/MyDrive").exists():
    print("Mounting Google Drive...", end=" ", flush=True)
    _drive.mount("/content/drive")
    print("ok")
else:
    print("Google Drive already mounted.")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PATHS + ENVIRONMENT                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
BASE      = Path(DRIVE_BASE)
DATA_DIR  = BASE / "data" / "processed"
ADAPT_DIR = Path("/content/adapters")        # local Colab disk; lost on session death
LOG_DIR   = Path("/content/training_logs")   # training loss CSVs; local, not Drive

# adapter_paths.json lives on Drive so evaluate_colab.py / redteam_colab.py
# can locate adapters regardless of session state.
_PATHS_MANIFEST = BASE / "adapters" / "adapter_paths.json"
(BASE / "adapters").mkdir(parents=True, exist_ok=True)
for _d in [ADAPT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Verify training data is reachable before spending time loading the model.
_missing = [str(DATA_DIR / f"train_condition_{l}.jsonl")
            for l in (["b", "c"] if CONDITION.lower() == "both" else [CONDITION.lower()])
            if not (DATA_DIR / f"train_condition_{l}.jsonl").exists()]
if _missing:
    raise FileNotFoundError(
        "Training data not found — Drive may not have synced yet.\n"
        + "\n".join(f"  {p}" for p in _missing)
        + "\nIf Drive just mounted, wait 10–15 seconds and re-run the cell."
    )

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN not set. Run the initialization cell "
        "(generate_data_colab.py setup section) first."
    )

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SEED     = 42

# Confirm GPU
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. This script requires an A100.")
_gpu = torch.cuda.get_device_properties(0)
print(f"GPU  : {_gpu.name}")
print(f"VRAM : {_gpu.total_memory / 1e9:.0f} GB")
print(f"torch: {torch.__version__}")
print()

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ QUANTIZATION CONFIG — 4-bit NF4                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,  # A100 supports native bf16
    bnb_4bit_use_double_quant = True,
)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ LORA CONFIG — from design document §2.2                                     │
# └─────────────────────────────────────────────────────────────────────────────┘
LORA_CONFIG = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias           = "none",
    task_type      = "CAUSAL_LM",
)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CHECKSUM HELPER                                                             │
# │  Validates both adapters train from the identical base model state.        │
# │  Uses sum of L2 norms over all non-LoRA parameters — fast on large models. │
# └─────────────────────────────────────────────────────────────────────────────┘
# Also defined identically in evaluate_colab.py and redteam_colab.py —
# update all three if the checksum formula changes.
def base_param_checksum(model) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            total += param.data.float().norm().item()
    return round(total, 4)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PROGRESS CALLBACK                                                           │
# │  Prints step / loss / ETA; writes loss to CSV for visualize.py             │
# └─────────────────────────────────────────────────────────────────────────────┘
class ProgressCallback(TrainerCallback):

    def __init__(self, label: str, csv_path: Path, total_steps: int):
        self.label       = label
        self.csv_path    = csv_path
        self.total_steps = total_steps
        self._t0         = None
        self._fh         = None
        self._writer     = None

    def on_train_begin(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._t0     = time.time()
        self._fh     = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["step", "loss", "learning_rate", "epoch"])
        print(f"\n[{self.label}] training started — {self.total_steps} steps total")

    def on_log(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not logs or "loss" not in logs:
            return
        step    = state.global_step
        loss    = logs["loss"]
        lr      = logs.get("learning_rate", 0.0)
        epoch   = logs.get("epoch", 0.0)

        self._writer.writerow([step, f"{loss:.6f}", f"{lr:.3e}", f"{epoch:.3f}"])
        self._fh.flush()

        elapsed = time.time() - self._t0
        if step > 0:
            sps     = step / elapsed
            rem_sec = (self.total_steps - step) / sps
            print(
                f"  [{self.label}] step {step:>5}/{self.total_steps}  "
                f"loss={loss:.4f}  lr={lr:.2e}  "
                f"elapsed={elapsed/60:.1f}m  ETA≈{rem_sec/60:.1f}m",
                flush=True,
            )

    def on_evaluate(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        if metrics:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                print(
                    f"  [{self.label}] eval  epoch={state.epoch:.2f}  "
                    f"eval_loss={eval_loss:.4f}",
                    flush=True,
                )

    def on_train_end(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._fh:
            self._fh.close()
        total_min = (time.time() - self._t0) / 60
        print(f"\n[{self.label}] training complete — {total_min:.1f} min total")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MODEL + TOKENIZER LOADER                                                    │
# └─────────────────────────────────────────────────────────────────────────────┘
def _load_base_only():
    """
    Load and prepare the base model for LoRA wrapping, without calling
    get_peft_model. Returns (prepared_base_model, tokenizer).
    Used by the CONDITION='both' path to avoid a second from_pretrained load.
    """
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token        = HF_TOKEN,
        padding_side = "right",
        add_eos_token= True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model (4-bit NF4, bf16 compute)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = BNB_CONFIG,
        device_map          = "auto",
        token               = HF_TOKEN,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "eager",
    )

    print("Verifying forward pass (eval mode)...", end=" ", flush=True)
    model.eval()
    with torch.no_grad():
        _dummy = tokenizer("Hello", return_tensors="pt").to(model.device)
        _      = model(**_dummy)
    print("ok")

    # prepare_model_for_kbit_training casts layer norms to float32,
    # enables input embedding gradients — required before get_peft_model.
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def load_base_model():
    """
    Load Mistral-7B-Instruct-v0.3 in 4-bit NF4, verify a forward pass in eval
    mode, then prepare for LoRA fine-tuning. Returns (model, tokenizer).
    """
    model, tokenizer = _load_base_only()
    model = get_peft_model(model, LORA_CONFIG)
    model.config.use_cache = False   # must be False when gradient_checkpointing=True
    model.print_trainable_parameters()
    return model, tokenizer


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ DATASET LOADER                                                              │
# └─────────────────────────────────────────────────────────────────────────────┘
def load_condition_dataset(
    condition_letter: str,
    tokenizer,
    max_examples: int | None = None,
):
    """
    Load train_condition_{b|c}.jsonl, apply the model's chat template to each
    example, and return a (train_dataset, eval_dataset) pair (95/5 split).

    Pre-applying the chat template and storing as 'text' is the most reliable
    approach across TRL 1.x versions — avoids relying on SFTTrainer's internal
    template inference, which changed between TRL 0.x and 1.x.
    """
    path = DATA_DIR / f"train_condition_{condition_letter}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Run the data generation cell first."
        )

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_examples and len(records) >= max_examples:
                break
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No examples loaded from {path}")

    texts = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize              = False,
            add_generation_prompt = False,
        )
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)
    split   = dataset.train_test_split(test_size=0.05, seed=SEED)
    print(f"  Dataset: {len(split['train'])} train / {len(split['test'])} eval "
          f"(from {len(records)} examples in {path.name})")
    return split["train"], split["test"]


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ TRAIN ONE CONDITION                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
_LABELS = {"b": "filter_adapter", "c": "intrinsic_adapter"}
_CHECKSUM_FILE = Path("/tmp/base_model_checksum.txt")


def train_condition(condition_letter: str, dry_run: bool = False, resume: bool = False,
                    _shared_base: dict | None = None):
    """
    _shared_base: if provided, must be {"model": prepared_base_model, "tokenizer": tok}.
    The base model is wrapped with get_peft_model here rather than reloaded from disk.
    At teardown the LoRA adapter is unloaded and the clean base is stored back in
    _shared_base["model"] for the next call. Only used when CONDITION="both".
    """
    label   = _LABELS[condition_letter]
    out_dir = ADAPT_DIR / label
    csv_path= LOG_DIR / f"training_loss_condition_{condition_letter}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 72}")
    print(f"  CONDITION {condition_letter.upper()} — {label}")
    print(f"  {'DRY RUN: 20 steps / 30 examples' if dry_run else 'FULL TRAINING'}")
    print(f"{'═' * 72}")

    # ── Load base model ──────────────────────────────────────────────────────
    if _shared_base is not None:
        # Re-use pre-loaded base — skip from_pretrained (~5 min saved)
        print("  Re-using pre-loaded base weights (CONDITION='both' optimisation).")
        base   = _shared_base["model"]
        tokenizer = _shared_base["tokenizer"]
        model  = get_peft_model(base, LORA_CONFIG)
        model.config.use_cache = False
        model.print_trainable_parameters()
    else:
        model, tokenizer = load_base_model()

    # ── Checksum verification ────────────────────────────────────────────────
    # Both adapters must train from bit-identical base model state.
    checksum = base_param_checksum(model)
    print(f"\nBase model checksum: {checksum}")

    if not _CHECKSUM_FILE.exists():
        _CHECKSUM_FILE.write_text(str(checksum))
        print("  Reference checksum saved (will be verified before second adapter).")
    else:
        expected = float(_CHECKSUM_FILE.read_text().strip())
        delta    = abs(checksum - expected)
        if delta > 0.001:
            raise RuntimeError(
                f"Base model checksum mismatch — adapters are NOT training from "
                f"identical state.\n  expected={expected}  got={checksum}  delta={delta}"
            )
        print(f"  ✓ Checksum matches reference (delta={delta:.6f}) — base state identical.")

    # ── Dataset ──────────────────────────────────────────────────────────────
    max_ex   = 30 if dry_run else None
    train_ds, eval_ds = load_condition_dataset(condition_letter, tokenizer, max_examples=max_ex)

    # ── Step / epoch counts ──────────────────────────────────────────────────
    # Effective batch = per_device_batch(4) × grad_accum(4) = 16
    if dry_run:
        n_epochs     = 1
        max_steps    = 20
        total_steps  = 20
    else:
        n_epochs     = 3
        max_steps    = -1
        steps_per_ep = max(1, len(train_ds) // 16)
        total_steps  = steps_per_ep * n_epochs

    print(f"\n  Effective batch size : 16  (4 per-device × 4 grad_accum)")
    print(f"  Total optimiser steps: {total_steps}")

    # ── SFTConfig (all hyperparams in one object — TRL 1.x style) ────────────
    sft_cfg = SFTConfig(
        # Output
        output_dir                 = str(out_dir),
        # Scale
        num_train_epochs           = n_epochs,
        max_steps                  = max_steps,
        # Batch / gradient
        per_device_train_batch_size= 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps= 4,
        # Optimiser
        learning_rate              = 2e-4,
        lr_scheduler_type          = "cosine",
        warmup_ratio               = 0.03,
        optim                      = "paged_adamw_8bit",
        # Precision — A100: bf16, not fp16
        bf16                       = True,
        fp16                       = False,
        # Sequence / packing — all in SFTConfig in TRL 1.2.0
        max_length                 = 2048,          # [TRL 1.2.0] renamed from max_seq_length
        dataset_text_field         = "text",        # matches key written by apply_chat_template
        packing                    = False,
        # Eval / checkpoint
        # [5.0 change] evaluation_strategy= → eval_strategy=
        eval_strategy              = "epoch",  # aligned with save_strategy for load_best_model_at_end
        save_strategy              = "epoch",
        load_best_model_at_end     = True,
        metric_for_best_model      = "eval_loss",
        greater_is_better          = False,
        save_total_limit           = 2,        # keep last 2 checkpoints; older ones deleted automatically
        # Logging
        logging_steps              = 10,
        logging_dir                = f"/tmp/tensorboard_{label}",
        report_to                  = "none",
        # Gradient checkpointing — let SFTConfig manage it (not prepare_model_for_kbit_training)
        gradient_checkpointing     = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        # Reproducibility
        seed                       = SEED,
        data_seed                  = SEED,
        # TRL housekeeping
        remove_unused_columns      = False,
    )

    # ── Resume checkpoint ────────────────────────────────────────────────────
    resume_from = None
    if resume and not dry_run:
        checkpoints = sorted(
            out_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"\n  Resuming from: {resume_from}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    # [5.0 / TRL 1.x] tokenizer= → processing_class=
    trainer = SFTTrainer(
        model            = model,
        args             = sft_cfg,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        processing_class = tokenizer,
        callbacks        = [ProgressCallback(label, csv_path, total_steps)],
    )

    print()
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save adapter ─────────────────────────────────────────────────────────
    if not dry_run:
        print(f"\nSaving adapter → {out_dir}...", end=" ", flush=True)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print("saved.")

        # Verify base weights unchanged (LoRA should not touch them)
        post_checksum = base_param_checksum(model.base_model.model)
        delta = abs(post_checksum - checksum)
        if delta > 0.001:
            print(f"  ⚠  Base weight checksum shifted by {delta:.6f} — unexpected with LoRA.")
        else:
            print(f"  ✓  Base weights unchanged post-training (delta={delta:.6f}).")

        # Write a small metadata file alongside the adapter
        meta = {
            "condition":           condition_letter,
            "label":               label,
            "model_id":            MODEL_ID,
            "base_checksum":       checksum,
            "post_train_checksum": post_checksum,
            "trained_at":          datetime.now(timezone.utc).isoformat(),
            "n_train_examples":    len(train_ds),
            "lora_r":              LORA_CONFIG.r,
            "lora_alpha":          LORA_CONFIG.lora_alpha,
            "epochs":              n_epochs,
        }
        with open(out_dir / "training_meta.json", "w") as _f:
            json.dump(meta, _f, indent=2)

        # Register adapter path on Drive so evaluate_colab.py can locate it
        # regardless of whether SAVE_ADAPTERS_TO_DRIVE is True or False.
        _existing = json.loads(_PATHS_MANIFEST.read_text()) if _PATHS_MANIFEST.exists() else {}
        _existing[label] = str(out_dir)
        _PATHS_MANIFEST.write_text(json.dumps(_existing, indent=2))
        print(f"  Registered: adapter_paths.json ← {label} → {out_dir}")

    # ── Free GPU memory ──────────────────────────────────────────────────────
    train_ds.cleanup_cache_files()
    eval_ds.cleanup_cache_files()

    if _shared_base is not None and not dry_run:
        # Unload LoRA adapter, return clean base weights to _shared_base for reuse.
        # model.unload() removes adapter modules and returns the underlying
        # AutoModelForCausalLM with prepare_model_for_kbit_training already applied —
        # the next call can go straight to get_peft_model without a full reload.
        post_pre_checksum = base_param_checksum(model.base_model.model)
        delta = abs(post_pre_checksum - checksum)
        if delta > 0.001:
            print(f"  ⚠  Post-train base checksum delta={delta:.6f} exceeds tolerance. "
                  f"Falling back to full reload for next adapter.", flush=True)
            _shared_base["model"] = None   # signal caller to do full reload
        else:
            _shared_base["model"] = model.unload()
            print(f"  ✓  Base weights unloaded cleanly (delta={delta:.6f}); "
                  f"ready for next adapter.", flush=True)
        del model, trainer, train_ds, eval_ds
        # tokenizer stays alive via _shared_base["tokenizer"]
    else:
        del model, tokenizer, trainer, train_ds, eval_ds

    gc.collect()
    torch.cuda.empty_cache()
    _used = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after cleanup: {_used:.2f} GB allocated")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MAIN — validate inputs, then train in order C → B                          │
# └─────────────────────────────────────────────────────────────────────────────┘

# Design doc: train intrinsic (C) first, then filter (B)
if CONDITION.lower() == "both":
    _run_order = ["c", "b"]
elif CONDITION.lower() in ("b", "c"):
    _run_order = [CONDITION.lower()]
else:
    raise ValueError(f"CONDITION must be 'b', 'c', or 'both'. Got: {CONDITION!r}")

# Pre-flight check: data files exist
for _letter in _run_order:
    _p = DATA_DIR / f"train_condition_{_letter}.jsonl"
    if not _p.exists():
        raise FileNotFoundError(
            f"Training data not found: {_p}\n"
            "Run the data generation cell first."
        )

print(f"Conditions : {' → '.join(_LABELS[l] for l in _run_order)}")
print(f"Dry run    : {DRY_RUN}")
print(f"Resume     : {RESUME}")
if DRY_RUN:
    print("             20 steps · 30 examples · 1 epoch per condition")
print()

_wall_t0 = time.time()

if len(_run_order) == 2:
    # CONDITION="both": load base once and share across both adapter training runs.
    # Saves ~5 min by avoiding a second from_pretrained call.
    print("CONDITION='both': loading base model once for both adapter runs...")
    _base_model, _base_tok = _load_base_only()
    _shared = {"model": _base_model, "tokenizer": _base_tok}
    del _base_model, _base_tok

    for _letter in _run_order:
        if _shared["model"] is None:
            # Checksum failed on previous unload — fall back to independent load
            print(f"  ⚠  Shared base unavailable; falling back to full reload for "
                  f"condition {_letter.upper()}.", flush=True)
            train_condition(_letter, dry_run=DRY_RUN, resume=RESUME)
        else:
            train_condition(_letter, dry_run=DRY_RUN, resume=RESUME,
                            _shared_base=_shared)

    del _shared
    gc.collect()
    torch.cuda.empty_cache()
else:
    train_condition(_run_order[0], dry_run=DRY_RUN, resume=RESUME)

_wall_total = time.time() - _wall_t0

# ── Final summary ────────────────────────────────────────────────────────────
print(f"\n{'═' * 72}")
print(f"  ALL TRAINING COMPLETE — {_wall_total / 60:.1f} min total")
print(f"{'═' * 72}")

for _letter in _run_order:
    _label   = _LABELS[_letter]
    _out_dir = ADAPT_DIR / _label
    _csv     = LOG_DIR / f"training_loss_condition_{_letter}.csv"
    _status  = "dry run only" if DRY_RUN else ("saved ✓" if _out_dir.exists() else "MISSING")
    print(f"\n  {_label}  [{_status}]")
    print(f"    adapter : {_out_dir}")
    if _csv.exists():
        with open(_csv) as _f:
            _rows = list(csv.reader(_f))
        if len(_rows) > 1:
            _last = _rows[-1]
            print(f"    final   : step={_last[0]}  loss={_last[1]}  epoch={_last[3]}")
    meta_path = _out_dir / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as _f:
            _meta = json.load(_f)
        print(f"    trained : {_meta['trained_at']}")

if DRY_RUN:
    print(f"\n{'─' * 72}")
    print("  Dry run verified. Set DRY_RUN = False to run full training.")
    print(f"{'─' * 72}")
else:
    print("\nFull training complete. Disconnecting runtime to stop compute billing...")
    from google.colab import runtime
    runtime.unassign()
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ DRIVE MOUNT                                                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
from google.colab import drive as _drive
if not Path("/content/drive/MyDrive").exists():
    print("Mounting Google Drive...", end=" ", flush=True)
    _drive.mount("/content/drive")
    print("ok")
else:
    print("Google Drive already mounted.")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PATHS + ENVIRONMENT                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
BASE      = Path(DRIVE_BASE)
DATA_DIR  = BASE / "data" / "processed"
ADAPT_DIR = BASE / "adapters"
LOG_DIR   = BASE / "logs"

for _d in [ADAPT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Verify training data is reachable before spending time loading the model.
_missing = [str(DATA_DIR / f"train_condition_{l}.jsonl")
            for l in (["b", "c"] if CONDITION.lower() == "both" else [CONDITION.lower()])
            if not (DATA_DIR / f"train_condition_{l}.jsonl").exists()]
if _missing:
    raise FileNotFoundError(
        "Training data not found — Drive may not have synced yet.\n"
        + "\n".join(f"  {p}" for p in _missing)
        + "\nIf Drive just mounted, wait 10–15 seconds and re-run the cell."
    )

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN not set. Run the initialization cell "
        "(generate_data_colab.py setup section) first."
    )

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SEED     = 42

# Confirm GPU
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. This script requires an A100.")
_gpu = torch.cuda.get_device_properties(0)
print(f"GPU  : {_gpu.name}")
print(f"VRAM : {_gpu.total_memory / 1e9:.0f} GB")
print(f"torch: {torch.__version__}")
print()

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ QUANTIZATION CONFIG — 4-bit NF4                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,  # A100 supports native bf16
    bnb_4bit_use_double_quant = True,
)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ LORA CONFIG — from design document §2.2                                     │
# └─────────────────────────────────────────────────────────────────────────────┘
LORA_CONFIG = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias           = "none",
    task_type      = "CAUSAL_LM",
)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CHECKSUM HELPER                                                             │
# │  Validates both adapters train from the identical base model state.        │
# │  Uses sum of L2 norms over all non-LoRA parameters — fast on large models. │
# └─────────────────────────────────────────────────────────────────────────────┘
def base_param_checksum(model) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            total += param.data.float().norm().item()
    return round(total, 4)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PROGRESS CALLBACK                                                           │
# │  Prints step / loss / ETA; writes loss to CSV for visualize.py             │
# └─────────────────────────────────────────────────────────────────────────────┘
class ProgressCallback(TrainerCallback):

    def __init__(self, label: str, csv_path: Path, total_steps: int):
        self.label       = label
        self.csv_path    = csv_path
        self.total_steps = total_steps
        self._t0         = None
        self._fh         = None
        self._writer     = None

    def on_train_begin(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._t0     = time.time()
        self._fh     = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["step", "loss", "learning_rate", "epoch"])
        print(f"\n[{self.label}] training started — {self.total_steps} steps total")

    def on_log(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not logs or "loss" not in logs:
            return
        step    = state.global_step
        loss    = logs["loss"]
        lr      = logs.get("learning_rate", 0.0)
        epoch   = logs.get("epoch", 0.0)

        self._writer.writerow([step, f"{loss:.6f}", f"{lr:.3e}", f"{epoch:.3f}"])
        self._fh.flush()

        elapsed = time.time() - self._t0
        if step > 0:
            sps     = step / elapsed
            rem_sec = (self.total_steps - step) / sps
            print(
                f"  [{self.label}] step {step:>5}/{self.total_steps}  "
                f"loss={loss:.4f}  lr={lr:.2e}  "
                f"elapsed={elapsed/60:.1f}m  ETA≈{rem_sec/60:.1f}m",
                flush=True,
            )

    def on_evaluate(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        if metrics:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                print(
                    f"  [{self.label}] eval  epoch={state.epoch:.2f}  "
                    f"eval_loss={eval_loss:.4f}",
                    flush=True,
                )

    def on_train_end(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._fh:
            self._fh.close()
        total_min = (time.time() - self._t0) / 60
        print(f"\n[{self.label}] training complete — {total_min:.1f} min total")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MODEL + TOKENIZER LOADER                                                    │
# └─────────────────────────────────────────────────────────────────────────────┘
def load_base_model():
    """
    Load Mistral-7B-Instruct-v0.3 in 4-bit NF4, verify a forward pass in eval
    mode, then prepare for LoRA fine-tuning. Returns (model, tokenizer).
    """
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token        = HF_TOKEN,
        padding_side = "right",   # right-pad for causal SFT
        add_eos_token= True,
    )
    # Mistral tokenizer ships without a pad token; map it to eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model (4-bit NF4, bf16 compute)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = BNB_CONFIG,
        device_map          = "auto",
        token               = HF_TOKEN,         # [5.0 change] was use_auth_token=
        torch_dtype         = torch.bfloat16,
        attn_implementation = "eager",          # sdpa/flash_attention_2 optional
    )

    # Verify forward pass in eval mode before touching training state
    print("Verifying forward pass (eval mode)...", end=" ", flush=True)
    model.eval()
    with torch.no_grad():
        _dummy = tokenizer("Hello", return_tensors="pt").to(model.device)
        _      = model(**_dummy)
    print("ok")

    # prepare_model_for_kbit_training casts layer norms to float32,
    # enables input embedding gradients — required before get_peft_model.
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.config.use_cache = False   # must be False when gradient_checkpointing=True

    model.print_trainable_parameters()
    return model, tokenizer


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ DATASET LOADER                                                              │
# └─────────────────────────────────────────────────────────────────────────────┘
def load_condition_dataset(
    condition_letter: str,
    tokenizer,
    max_examples: int | None = None,
):
    """
    Load train_condition_{b|c}.jsonl, apply the model's chat template to each
    example, and return a (train_dataset, eval_dataset) pair (95/5 split).

    Pre-applying the chat template and storing as 'text' is the most reliable
    approach across TRL 1.x versions — avoids relying on SFTTrainer's internal
    template inference, which changed between TRL 0.x and 1.x.
    """
    path = DATA_DIR / f"train_condition_{condition_letter}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Run the data generation cell first."
        )

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_examples and len(records) >= max_examples:
                break
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No examples loaded from {path}")

    texts = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize              = False,
            add_generation_prompt = False,
        )
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)
    split   = dataset.train_test_split(test_size=0.05, seed=SEED)
    print(f"  Dataset: {len(split['train'])} train / {len(split['test'])} eval "
          f"(from {len(records)} examples in {path.name})")
    return split["train"], split["test"]


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ TRAIN ONE CONDITION                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
_LABELS = {"b": "filter_adapter", "c": "intrinsic_adapter"}
_CHECKSUM_FILE = LOG_DIR / "base_model_checksum.txt"


def train_condition(condition_letter: str, dry_run: bool = False, resume: bool = False):
    label   = _LABELS[condition_letter]
    out_dir = ADAPT_DIR / label
    csv_path= LOG_DIR / f"training_loss_condition_{condition_letter}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 72}")
    print(f"  CONDITION {condition_letter.upper()} — {label}")
    print(f"  {'DRY RUN: 20 steps / 30 examples' if dry_run else 'FULL TRAINING'}")
    print(f"{'═' * 72}")

    # ── Load base model fresh ────────────────────────────────────────────────
    model, tokenizer = load_base_model()

    # ── Checksum verification ────────────────────────────────────────────────
    # Both adapters must train from bit-identical base model state.
    checksum = base_param_checksum(model)
    print(f"\nBase model checksum: {checksum}")

    if not _CHECKSUM_FILE.exists():
        _CHECKSUM_FILE.write_text(str(checksum))
        print("  Reference checksum saved (will be verified before second adapter).")
    else:
        expected = float(_CHECKSUM_FILE.read_text().strip())
        delta    = abs(checksum - expected)
        if delta > 1.0:  # tolerance for float accumulation across reloads
            raise RuntimeError(
                f"Base model checksum mismatch — adapters are NOT training from "
                f"identical state.\n  expected={expected}  got={checksum}  delta={delta}"
            )
        print(f"  ✓ Checksum matches reference (delta={delta:.4f}) — base state identical.")

    # ── Dataset ──────────────────────────────────────────────────────────────
    max_ex   = 30 if dry_run else None
    train_ds, eval_ds = load_condition_dataset(condition_letter, tokenizer, max_examples=max_ex)

    # ── Step / epoch counts ──────────────────────────────────────────────────
    # Effective batch = per_device_batch(4) × grad_accum(4) = 16
    if dry_run:
        n_epochs     = 1
        max_steps    = 20
        total_steps  = 20
    else:
        n_epochs     = 3
        max_steps    = -1
        steps_per_ep = max(1, len(train_ds) // 16)
        total_steps  = steps_per_ep * n_epochs

    print(f"\n  Effective batch size : 16  (4 per-device × 4 grad_accum)")
    print(f"  Total optimiser steps: {total_steps}")

    # ── SFTConfig (all hyperparams in one object — TRL 1.x style) ────────────
    sft_cfg = SFTConfig(
        # Output
        output_dir                 = str(out_dir),
        # Scale
        num_train_epochs           = n_epochs,
        max_steps                  = max_steps,
        # Batch / gradient
        per_device_train_batch_size= 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps= 4,
        # Optimiser
        learning_rate              = 2e-4,
        lr_scheduler_type          = "cosine",
        warmup_ratio               = 0.03,
        optim                      = "paged_adamw_8bit",
        # Precision — A100: bf16, not fp16
        bf16                       = True,
        fp16                       = False,
        # Sequence / packing — all in SFTConfig in TRL 1.2.0
        max_length                 = 2048,          # [TRL 1.2.0] renamed from max_seq_length
        dataset_text_field         = "text",        # matches key written by apply_chat_template
        packing                    = False,
        # Eval / checkpoint
        # [5.0 change] evaluation_strategy= → eval_strategy=
        eval_strategy              = "epoch",  # aligned with save_strategy for load_best_model_at_end
        save_strategy              = "epoch",
        load_best_model_at_end     = True,
        metric_for_best_model      = "eval_loss",
        greater_is_better          = False,
        save_total_limit           = 2,        # keep last 2 checkpoints to save Drive space
        # Logging
        logging_steps              = 10,
        logging_dir                = str(LOG_DIR / f"tensorboard_{label}"),
        report_to                  = "none",
        # Gradient checkpointing — let SFTConfig manage it (not prepare_model_for_kbit_training)
        gradient_checkpointing     = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        # Reproducibility
        seed                       = SEED,
        data_seed                  = SEED,
        # TRL housekeeping
        remove_unused_columns      = False,
    )

    # ── Resume checkpoint ────────────────────────────────────────────────────
    resume_from = None
    if resume and not dry_run:
        checkpoints = sorted(
            out_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"\n  Resuming from: {resume_from}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    # [5.0 / TRL 1.x] tokenizer= → processing_class=
    trainer = SFTTrainer(
        model            = model,
        args             = sft_cfg,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        processing_class = tokenizer,
        callbacks        = [ProgressCallback(label, csv_path, total_steps)],
    )

    print()
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save adapter ─────────────────────────────────────────────────────────
    if not dry_run:
        print(f"\nSaving adapter → {out_dir}...", end=" ", flush=True)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print("saved.")

        # Verify base weights unchanged (LoRA should not touch them)
        post_checksum = base_param_checksum(model.base_model.model)
        delta = abs(post_checksum - checksum)
        if delta > 1.0:
            print(f"  ⚠  Base weight checksum shifted by {delta:.4f} — unexpected with LoRA.")
        else:
            print(f"  ✓  Base weights unchanged post-training (delta={delta:.6f}).")

        # Write a small metadata file alongside the adapter
        meta = {
            "condition":           condition_letter,
            "label":               label,
            "model_id":            MODEL_ID,
            "base_checksum":       checksum,
            "post_train_checksum": post_checksum,
            "trained_at":          datetime.now(timezone.utc).isoformat(),
            "n_train_examples":    len(train_ds),
            "lora_r":              LORA_CONFIG.r,
            "lora_alpha":          LORA_CONFIG.lora_alpha,
            "epochs":              n_epochs,
        }
        with open(out_dir / "training_meta.json", "w") as _f:
            json.dump(meta, _f, indent=2)

    # ── Free GPU memory ──────────────────────────────────────────────────────
    del model, tokenizer, trainer, train_ds, eval_ds
    gc.collect()
    torch.cuda.empty_cache()
    _used = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after cleanup: {_used:.2f} GB allocated")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MAIN — validate inputs, then train in order C → B                          │
# └─────────────────────────────────────────────────────────────────────────────┘

# Design doc: train intrinsic (C) first, then filter (B)
if CONDITION.lower() == "both":
    _run_order = ["c", "b"]
elif CONDITION.lower() in ("b", "c"):
    _run_order = [CONDITION.lower()]
else:
    raise ValueError(f"CONDITION must be 'b', 'c', or 'both'. Got: {CONDITION!r}")

# Pre-flight check: data files exist
for _letter in _run_order:
    _p = DATA_DIR / f"train_condition_{_letter}.jsonl"
    if not _p.exists():
        raise FileNotFoundError(
            f"Training data not found: {_p}\n"
            "Run the data generation cell first."
        )

print(f"Conditions : {' → '.join(_LABELS[l] for l in _run_order)}")
print(f"Dry run    : {DRY_RUN}")
print(f"Resume     : {RESUME}")
if DRY_RUN:
    print("             20 steps · 30 examples · 1 epoch per condition")
print()

_wall_t0 = time.time()

for _letter in _run_order:
    train_condition(_letter, dry_run=DRY_RUN, resume=RESUME)

_wall_total = time.time() - _wall_t0

# ── Final summary ────────────────────────────────────────────────────────────
print(f"\n{'═' * 72}")
print(f"  ALL TRAINING COMPLETE — {_wall_total / 60:.1f} min total")
print(f"{'═' * 72}")

for _letter in _run_order:
    _label   = _LABELS[_letter]
    _out_dir = ADAPT_DIR / _label
    _csv     = LOG_DIR / f"training_loss_condition_{_letter}.csv"
    _status  = "dry run only" if DRY_RUN else ("saved ✓" if _out_dir.exists() else "MISSING")
    print(f"\n  {_label}  [{_status}]")
    print(f"    adapter : {_out_dir}")
    if _csv.exists():
        with open(_csv) as _f:
            _rows = list(csv.reader(_f))
        if len(_rows) > 1:
            _last = _rows[-1]
            print(f"    final   : step={_last[0]}  loss={_last[1]}  epoch={_last[3]}")
    meta_path = _out_dir / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as _f:
            _meta = json.load(_f)
        print(f"    trained : {_meta['trained_at']}")

if DRY_RUN:
    print(f"\n{'─' * 72}")
    print("  Dry run verified. Set DRY_RUN = False to run full training.")
    print(f"{'─' * 72}")
else:
    print("\nFull training complete. Disconnecting runtime to stop compute billing...")
    from google.colab import runtime
    runtime.unassign()
