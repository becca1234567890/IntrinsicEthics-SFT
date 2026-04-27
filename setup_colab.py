# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: INITIALIZATION — run this first every session
# ═══════════════════════════════════════════════════════════════════════════════

import os
import subprocess
from google.colab import drive, userdata

# ── Mount Google Drive ────────────────────────────────────────────────────────
drive.mount('/content/drive')

# ── Verify GPU ────────────────────────────────────────────────────────────────
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                       '--format=csv,noheader'], capture_output=True, text=True)
print(f"GPU: {result.stdout.strip()}")

# ── Install dependencies ──────────────────────────────────────────────────────
print("\nInstalling dependencies...", flush=True)
subprocess.run(['pip', 'install', '-q',
    'transformers>=4.40.0', 'peft>=0.10.0', 'trl>=0.8.0',
    'bitsandbytes>=0.43.0', 'datasets>=2.18.0', 'anthropic>=0.25.0',
    'accelerate>=0.29.0', 'scipy>=1.12.0', 'pandas>=2.2.0',
    'matplotlib>=3.8.0', 'seaborn>=0.13.0', 'scikit-learn>=1.4.0',
    'tqdm>=4.66.0', 'evaluate>=0.4.0'], check=True)
print("Installation complete.")

# ── Verify imports and versions ───────────────────────────────────────────────
import torch, transformers, peft, trl, bitsandbytes, datasets, anthropic, accelerate
from packaging import version
from google.colab import userdata

print(f"\ntorch:          {torch.__version__}")
print(f"transformers:   {transformers.__version__}")
print(f"peft:           {peft.__version__}")
print(f"trl:            {trl.__version__}")
print(f"bitsandbytes:   {bitsandbytes.__version__}")
print(f"datasets:       {datasets.__version__}")
print(f"anthropic:      {anthropic.__version__}")
print(f"accelerate:     {accelerate.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"GPU:            {torch.cuda.get_device_name(0)}")
print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

trl_ok = version.parse(trl.__version__) >= version.parse("0.8.0")
dpo_ok = version.parse(trl.__version__) >= version.parse("0.8.6")
print(f"\ntrl >= 0.8.0 (SFT): {'✓' if trl_ok else '✗ UPGRADE NEEDED'}")
print(f"trl >= 0.8.6 (DPO): {'✓' if dpo_ok else '✗ UPGRADE NEEDED'}")

# ── Verify API keys ───────────────────────────────────────────────────────────
anthropic_key = userdata.get('ANTHROPIC_API_KEY')
hf_token = userdata.get('HF_TOKEN')
os.environ['HF_TOKEN'] = hf_token if hf_token else ''
anthropic_key = userdata.get('ANTHROPIC_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = anthropic_key if anthropic_key else ''
print(f"HF token:       {'✓' if hf_token else '✗ MISSING'}")
print(f"Anthropic key:  {'✓' if anthropic_key else '✗ MISSING'}")
print("\n✓ Initialization complete — ready to run experiment cells.")print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"GPU:            {torch.cuda.get_device_name(0)}")
print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

trl_ok = version.parse(trl.__version__) >= version.parse("0.8.0")
dpo_ok = version.parse(trl.__version__) >= version.parse("0.8.6")
print(f"\ntrl >= 0.8.0 (SFT): {'✓' if trl_ok else '✗ UPGRADE NEEDED'}")
print(f"trl >= 0.8.6 (DPO): {'✓' if dpo_ok else '✗ UPGRADE NEEDED'}")

# ── Verify API keys ───────────────────────────────────────────────────────────
anthropic_key = userdata.get('ANTHROPIC_API_KEY')
hf_token = userdata.get('HF_TOKEN')
os.environ['HF_TOKEN'] = hf_token if hf_token else ''
anthropic_key = userdata.get('ANTHROPIC_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = anthropic_key if anthropic_key else ''
print(f"HF token:       {'✓' if hf_token else '✗ MISSING'}")
print(f"Anthropic key:  {'✓' if anthropic_key else '✗ MISSING'}")
print("\n✓ Initialization complete — ready to run experiment cells.")
