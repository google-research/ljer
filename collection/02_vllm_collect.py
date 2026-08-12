"""Step 2 (GPU) -- collect N=10 judge ratings per (item, model, arm) with vLLM.

Runs on a machine with a GPU, not the laptop that runs the analysis. Uses vLLM's
offline batch API, so there is no server to stand up and no rate limiting, and
thousands of prompts are batched at once.

Requires roughly 80 GB of GPU memory. The largest judge is 65.5 GB of weights at
BF16 and will not fit a 48 GB card. Models are loaded one at a time, smallest
first, so only the largest has to fit and a configuration error surfaces on a 2 GB
model rather than after a 65 GB download.

  # on the GPU machine
  pip install vllm
  export HF_TOKEN=...                 # gemma-3-1b-it is gated
  python 02_vllm_collect.py --check   # 3 items, all models -- verifies everything
  python 02_vllm_collect.py           # full pilot

Models are loaded ONE AT A TIME, smallest first, so only the largest must fit.
Output appends to out/raw_ratings.jsonl in the same schema 03_parse.py expects.
"""
import argparse, json, gc, shutil, sys, time
from pathlib import Path
import config as C

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"      # gitignored: reconstructed benchmark text (input)
OUT = ROOT / "out"        # gitignored: raw judge transcripts (output)
OUT.mkdir(exist_ok=True)
# Raw judge transcripts. Judges quote fragments of the question and completion
# while reasoning, so this file is a partial reproduction of benchmark text and
# is gitignored. 03_parse.py distils it into data/ratings.csv, which contains
# verdicts only and is what the repository releases.
RAW = OUT / "raw_ratings.jsonl"


def build_prompt(item, arm):
    """noref omits the reference answer; format() ignores unused kwargs is FALSE
    in Python, so pass only what each template needs."""
    tmpl = C.ARMS[arm]
    if arm == "noref":
        return tmpl.format(question=item["question"], completion=item["completion"])
    return tmpl.format(question=item["question"], answer=item["answer"],
                       completion=item["completion"])


def done_keys():
    d = set()
    if RAW.exists():
        for line in RAW.open():
            try:
                r = json.loads(line)
                d.add((r["item_id"], r["model"], r["arm"]))
            except Exception:
                pass
    return d


def purge_weights(repo):
    """Delete a model's downloaded weights after we are done with it.

    All seven models total ~200 GB. Purging as we go caps peak disk at roughly
    the largest model plus overhead, about 90 GB, so the run fits on a much
    smaller disk. Re-downloading costs a few minutes if the model is rerun.

    Set HF_HOME to a path on a filesystem with room for the largest model. Some
    hosts mount the working directory on a separate, smaller volume than the
    root filesystem, and the download will fail on a quota rather than a
    space error if the cache lands there.
    """
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
    except Exception as e:
        print(f"  (purge skipped: {e})"); return
    for r in cache.repos:
        if r.repo_id == repo:
            gb = r.size_on_disk / 1e9
            shutil.rmtree(r.repo_path, ignore_errors=True)
            print(f"  purged {repo} ({gb:.1f} GB freed)")
            return


def run_model(model, items, done, check):
    from vllm import LLM, SamplingParams

    jobs = [(it, arm) for it in items for arm in C.ARMS
            if (it["item_id"], model["key"], arm) not in done]
    if not jobs:
        print(f"  {model['key']}: nothing pending, skipping load")
        return 0

    print(f"\n=== {model['key']}  ({model['gb']:.1f} GB, {model['dtype']})  "
          f"{len(jobs)} prompts x n={C.N_ROUNDS} ===")

    # Qwen3-32B at 65.5 GB leaves little KV room on an 80 GB card; cap
    # concurrency so vLLM does not OOM on profiling.
    # Context ceiling comes from config (C.MAX_MODEL_LEN) so 01_prepare_data.py
    # checks against the same number the collector actually uses.
    # seed: an INTEGER, not None. Newer vLLM rejects None, and a fixed seed is
    # correct here anyway -- we request all N_ROUNDS samples in ONE call via
    # SamplingParams(n=...), and vLLM draws them sequentially from a single RNG
    # stream, so the draws still differ from each other. The seed makes the run
    # reproducible instead. (A fixed seed WOULD be fatal if we made N separate
    # single-sample calls, which is why we do not.)
    kwargs = dict(model=model["repo"], revision=model["rev"], dtype=model["dtype"],
                  gpu_memory_utilization=0.92, max_model_len=C.MAX_MODEL_LEN,
                  seed=C.SEED)
    if model["gb"] > 60:
        kwargs["max_num_seqs"] = 32

    llm = LLM(**kwargs)
    sp = SamplingParams(n=C.N_ROUNDS, temperature=C.TEMPERATURE, top_p=C.TOP_P,
                        max_tokens=C.MAX_TOKENS)

    convs = [[{"role": "user", "content": build_prompt(it, arm)}] for it, arm in jobs]

    # Per-model: enable_thinking is Qwen-only, reasoning_effort is gpt-oss-only.
    # Passing the wrong kwarg to the wrong family errors out.
    ctk = C.CHAT_KWARGS.get(model["key"]) or None
    try:
        outs = llm.chat(convs, sp, chat_template_kwargs=ctk) if ctk else llm.chat(convs, sp)
    except TypeError:
        # older vLLM without chat_template_kwargs
        print("  ! vLLM ignored chat_template_kwargs -- thinking mode NOT disabled.")
        print("    Upgrade vLLM or expect long outputs. Continuing.")
        outs = llm.chat(convs, sp)

    n = 0
    with RAW.open("a") as fh:
        for (it, arm), o in zip(jobs, outs):
            texts = [c.text for c in o.outputs]
            fh.write(json.dumps(dict(
                item_id=it["item_id"], split=it["split"], gold=it["gold"],
                model=model["key"], model_id=model["repo"], revision=model["rev"],
                dtype=model["dtype"], arm=arm,
                chat_kwargs=C.CHAT_KWARGS.get(model["key"], {}),
                native_reasoner=model["key"] in C.NATIVE_REASONERS,
                temperature=C.TEMPERATURE, top_p=C.TOP_P,
                n_rounds=len(texts), raw=texts, ts=time.time())) + "\n")
            n += 1

    # sanity: do the 10 draws actually differ? If not, the study is void.
    uniq = [len(set(c.text.strip() for c in o.outputs)) for o in outs[:20]]
    frac = sum(1 for u in uniq if u == 1) / max(len(uniq), 1)
    print(f"  wrote {n}.  identical-across-all-10 on {frac:.0%} of first 20 prompts")
    if frac > 0.9:
        print("  !! NEAR-ZERO STOCHASTICITY. Raise TEMPERATURE or drop this model.")

    del llm
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="3 items per model")
    ap.add_argument("--only", help="comma-separated model keys")
    ap.add_argument("--purge", action="store_true",
                    help="delete each model's weights after use (small disk)")
    a = ap.parse_args()

    import os
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("!! HF_TOKEN is not set in this shell.")
        print("   gemma-3-1b-it is a gated repo and will fail with a 401.")
        print("   Set it to a read-scoped token and rerun. The token is used only to")
        print("   download weights; nothing is uploaded to Hugging Face.\n")

    src = WORK / "pilot_items.jsonl"
    if not src.exists():
        sys.exit(f"missing {src}\n"
                 "Run collection/fetch_items.py to rebuild it from data/item_ids.csv,\n"
                 "or collection/01_prepare_data.py to draw a fresh sample.")
    items = [json.loads(l) for l in src.open()]
    if a.check:
        items = items[:3]
    models = C.MODELS
    if a.only:
        keys = set(a.only.split(","))
        models = [m for m in models if m["key"] in keys]

    models = sorted(models, key=lambda m: m["gb"])   # smallest first: fail fast, cheap
    done = done_keys()
    tot_gb = sum(m["gb"] for m in models)
    peak = max(m["gb"] for m in models) if a.purge else tot_gb
    print(f"{len(items)} items x {len(C.ARMS)} arms x {len(models)} models "
          f"({len(done)} cells already cached)")
    print(f"weights: {tot_gb:.0f} GB total, peak disk ~{peak + 25:.0f} GB "
          f"({'purging as we go' if a.purge else 'keeping all -- use --purge for less'})")

    total = 0
    for m in models:
        try:
            total += run_model(m, items, done, a.check)
            if a.purge:
                purge_weights(m["repo"])
        except Exception as e:
            print(f"  FAILED {m['key']}: {type(e).__name__}: {e}", file=sys.stderr)
            print("  continuing with remaining models", file=sys.stderr)

    print(f"\ndone: {total} cells -> {RAW}")
    print("Copy out/raw_ratings.jsonl back to your laptop, then run "
          "03_parse.py and 04_diagnostics.py there.")


if __name__ == "__main__":
    main()
