"""Collect the HuggingFace metadata needed to finalize config.py and the
approval request. Run locally, paste the output back.

  pip install huggingface_hub
  huggingface-cli login          # needed for gated repos (Gemma, possibly gpt-oss)
  python hf_probe.py
"""
import json, sys
from huggingface_hub import HfApi, hf_hub_download, model_info

REPOS = [
    "google/gemma-3-1b-it",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",          # verify exact name; may differ
    "openai/gpt-oss-20b",
    "google/gemma-4-26B-A4B-it",# verify exact name; may differ
    "Qwen/Qwen3-32B",
]

api = HfApi()

def grab(repo, fname):
    try:
        return json.load(open(hf_hub_download(repo, fname)))
    except Exception:
        return {}

for repo in REPOS:
    print("=" * 78)
    print(repo)
    try:
        info = model_info(repo, files_metadata=True)
    except Exception as e:
        print(f"  !! CANNOT ACCESS: {type(e).__name__}: {e}")
        print("  -> gated (accept terms on the model page + `huggingface-cli login`),")
        print("     renamed, or private. Resolve before we finalize.")
        continue

    cfg  = grab(repo, "config.json")
    tcfg = grab(repo, "tokenizer_config.json")
    gcfg = grab(repo, "generation_config.json")

    # weight size -> GPU memory planning
    gb = sum(f.size or 0 for f in (info.siblings or [])
             if f.rfilename.endswith((".safetensors", ".bin"))) / 1e9

    print(f"  revision (pin this)  : {info.sha}")
    print(f"  gated                : {getattr(info, 'gated', 'unknown')}")
    print(f"  license (card)       : {(info.cardData or {}).get('license', '?')}")
    print(f"  weights on disk      : {gb:.1f} GB")
    print(f"  architecture         : {cfg.get('architectures')}")
    print(f"  torch_dtype          : {cfg.get('torch_dtype')}")
    print(f"  quantization_config  : {'YES -> ' + str(cfg.get('quantization_config'))[:90] if cfg.get('quantization_config') else 'none (unquantized)'}")
    print(f"  max_position_embed   : {cfg.get('max_position_embeddings')}")
    print(f"  num_params (if given): {cfg.get('num_parameters', 'n/a')}")
    print(f"  MoE experts          : {cfg.get('num_experts') or cfg.get('num_local_experts') or 'dense'}")
    print(f"  chat_template present: {'yes' if tcfg.get('chat_template') else 'NO -- needs manual prompt formatting'}")
    print(f"  gen defaults         : temp={gcfg.get('temperature')} top_p={gcfg.get('top_p')}")

    # thinking / reasoning mode -- the setting most likely to distort this study
    tmpl = tcfg.get("chat_template") or ""
    flags = [k for k in ("enable_thinking", "reasoning_effort", "thinking") if k in tmpl]
    print(f"  reasoning toggles    : {flags if flags else 'none found in chat template'}")
    if "<think>" in tmpl or "enable_thinking" in tmpl:
        print("     ^^ THINKING MODE SUPPORTED -- must be explicitly set, see notes")
print("=" * 78)
print("Paste this whole output back.")
