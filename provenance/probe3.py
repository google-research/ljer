# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-flight checks. Laptop only -- downloads tokenizers and text files
(a few MB), never model weights.

  pip install transformers
  python probe3.py

Answers two questions:
  1. Will vLLM's llm.chat() find a chat template for every model?
     (vLLM uses the same tokenizer.apply_chat_template path, so if this
      renders here it will render there.)
  2. Does enable_thinking=False actually take effect on the Qwen models?
  3. What license files does each repo really carry?
"""
import json
from huggingface_hub import list_repo_files, hf_hub_download
from transformers import AutoTokenizer

REPOS = [
    "google/gemma-3-1b-it",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",
    "openai/gpt-oss-20b",
    "google/gemma-4-26B-A4B-it",
    "Qwen/Qwen3-32B",
]
MSG = [{"role": "user", "content": "PROMPT_BODY_HERE"}]

print("#" * 78)
print("# 1. CHAT TEMPLATE RENDERING")
print("#" * 78)
for r in REPOS:
    print(f"\n--- {r}")
    try:
        tok = AutoTokenizer.from_pretrained(r, trust_remote_code=True)
    except Exception as e:
        print(f"  TOKENIZER LOAD FAILED: {type(e).__name__}: {e}")
        continue
    try:
        s = tok.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True)
        print(f"  RENDERS OK ({len(s)} chars)")
        print("  ", repr(s[:180]))
    except Exception as e:
        print(f"  NO TEMPLATE: {type(e).__name__}: {e}")
        print("  -> vLLM llm.chat() will fail. Use llm.generate() with manual formatting.")
        continue
    # thinking toggle: does the rendered string actually change?
    if "qwen" in r.lower():
        try:
            off = tok.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            on  = tok.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True)
            if off == on:
                print("  !! enable_thinking has NO EFFECT on the rendered prompt.")
                print("     Check whether thinking is controlled some other way.")
            else:
                print(f"  enable_thinking WORKS (off={len(off)} vs on={len(on)} chars)")
                d = off.replace(on, "") or on.replace(off, "")
                print("   diff:", repr(d[:100]))
        except TypeError:
            print("  !! tokenizer rejects enable_thinking kwarg -- verify another way.")

print()
print("#" * 78)
print("# 2. LICENSE FILES ACTUALLY IN EACH REPO")
print("#" * 78)
for r in ["google/gemma-3-1b-it", "google/gemma-4-26B-A4B-it"]:
    print(f"\n--- {r}")
    try:
        files = list_repo_files(r)
    except Exception as e:
        print(f"  cannot list: {e}"); continue
    lic = [f for f in files if "licen" in f.lower() or "terms" in f.lower()
           or "notice" in f.lower() or "policy" in f.lower()]
    print("  license-ish files:", lic or "NONE")
    for f in lic[:2]:
        try:
            txt = open(hf_hub_download(r, f)).read()
            print(f"  --- first 400 chars of {f} ---")
            print("  " + txt[:400].replace("\n", "\n  "))
        except Exception as e:
            print(f"  could not read {f}: {e}")
    # model card frontmatter is what the `license:` field came from
    try:
        card = open(hf_hub_download(r, "README.md")).read()
        head = card[:600]
        print("  --- model card frontmatter ---")
        print("  " + head.replace("\n", "\n  "))
    except Exception as e:
        print(f"  no README: {e}")
