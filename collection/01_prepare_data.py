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

"""Step 1 -- fetch VerifyBench, inspect the schema, draw the pilot sample.

Runs on your LAPTOP (no GPU). Writes work/pilot_items.jsonl, which you then
copy to the GPU machine for step 2.

  !! THE OUTPUT CONTAINS BENCHMARK TEXT AND MUST NOT BE REDISTRIBUTED. !!

work/pilot_items.jsonl holds questions, reference answers and completions copied
verbatim from VerifyBench. It is written to work/, which is gitignored, so that a
user running this script cannot commit it by accident. This repository ships item
identifiers instead (data/item_ids.csv), which is what keeps the benchmark's own
license, and the licenses of the 41 upstream datasets it draws on, attached to
the benchmark rather than following our release. Use collection/fetch_items.py to
rebuild this file from those identifiers.

Run twice:
  python 01_prepare_data.py --inspect   # prints columns; update FIELDS in config.py
  python 01_prepare_data.py             # writes data/pilot_items.jsonl

The sample is stratified on gold label x answer type and seeded, so both
classes and all four answer types are represented and the draw is reproducible.

Sizes come from config: 150 items from VerifyBench (balanced, pi = 0.500 --
this is the split we estimate error rates on) and 100 from VerifyBench-Hard
(pi = 0.291, too few positives for rate estimation, used for the within-item
variance check in D1).
"""
import argparse, json, random, sys
from pathlib import Path
from collections import Counter
import config as C

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"      # gitignored: holds reconstructed benchmark text
WORK.mkdir(exist_ok=True)


def load_splits():
    """Pull both splits from the Hugging Face hub. License: MIT (ZJU-REAL/VerifyBench)."""
    from datasets import load_dataset
    ds = load_dataset("ZJU-REAL/VerifyBench")
    return ds["VerifyBench"], ds["VerifyBenchHard"]


def inspect(split, name):
    print(f"\n=== {name}: {len(split)} rows ===")
    print("columns:", split.column_names)
    row = split[0]
    for k, v in row.items():
        s = str(v).replace("\n", " ")
        print(f"  {k:<20} {type(v).__name__:<6} {s[:110]}")
    # gold label balance -- must match the paper (VB 1000/1000, VB-H 291/709)
    for cand in ("gold_correct", "label", "correct", "golden_label", "y"):
        if cand in split.column_names:
            print(f"  -> label balance on '{cand}':", Counter(split[cand]))
            break


def stratified(split, n, seed, want_types=True, one_per_question=True,
               balance_gold=True):
    """Draw n rows from a split.

    one_per_question: VerifyBench pairs each question with TWO completions (one
    correct, one incorrect), so its 2,000 rows cluster into 1,000 questions.
    Taking both would put two dependent items in the sample, violating the
    conditional-independence assumption. We take at most one per question.

    balance_gold: TRUE only where the natural prevalence is already balanced.
    VerifyBench is 50/50 by construction, so stratifying on gold costs nothing.
    VerifyBench-Hard is NATURALLY 291/709 -- larger judges over-accept wrong
    answers, and that skew is a real property of the data. Prevalence is one of
    the quantities this paper estimates, so resampling it to 50/50 would destroy
    the estimand. For VBH we take a simple random sample, preserving both the
    prevalence and the natural answer-type composition.
    """
    rng = random.Random(seed)
    F = C.FIELDS
    rows = [dict(r) for r in split]
    for i, r in enumerate(rows):
        r["_idx"] = i
    if one_per_question and "question_id" in split.column_names:
        by_q = {}
        for r in rows:
            by_q.setdefault(r["question_id"], []).append(r)
        rows = [rng.choice(v) for v in by_q.values()]
        rng.shuffle(rows)
    # strata = (gold label, answer type) when a type column exists, else gold label alone
    if not balance_gold:
        # simple random sample: preserves natural prevalence AND natural
        # answer-type composition
        return rows[:n]
    tcol = next((c for c in ("answer_type", "type", "subtype") if c in split.column_names), None)
    def key(r):
        g = int(bool(r[F["gold"]]))
        return (g, r[tcol]) if (want_types and tcol) else (g,)
    buckets = {}
    for r in rows:
        buckets.setdefault(key(r), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    # round-robin across strata so the draw stays balanced at any n
    out, keys = [], sorted(buckets, key=str)
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    vb, vbh = load_splits()
    if a.inspect:
        inspect(vb, "VerifyBench"); inspect(vbh, "VerifyBench-Hard")
        print("\nUpdate FIELDS in config.py to match, then rerun without --inspect.")
        return

    F = C.FIELDS
    missing = [v for v in F.values() if v not in vb.column_names]
    if missing:
        sys.exit(f"FIELDS mismatch, columns not found: {missing}. Run --inspect first.")

    items = []
    # VB: stratified + gold-balanced (matches its natural 50/50 composition).
    # VBH: simple random (preserves its natural 291/709 prevalence).
    for split, name, n, bal in ((vb, "VB", C.PILOT_VB, True),
                                (vbh, "VBH", C.PILOT_VBH, False)):
        for r in stratified(split, n, C.SEED, balance_gold=bal):
            items.append(dict(
                item_id=f"{name}-{r['_idx']}",
                split=name,
                question_id=r.get("question_id"),
                source=r.get("source"),
                answer_type=r.get("answer_type"),
                completion_model=r.get("completion_model"),
                question=r[F["question"]],
                answer=r[F["answer"]],
                completion=r[F["completion"]],
                gold=int(bool(r[F["gold"]])),
            ))

    p = WORK / "pilot_items.jsonl"
    with p.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    print(f"wrote {len(items)} items -> {p}")
    for name in ("VB", "VBH"):
        sub = [i for i in items if i["split"] == name]
        if not sub:
            continue
        prev = sum(i["gold"] for i in sub) / len(sub)
        want = 0.500 if name == "VB" else 0.291
        flag = "" if abs(prev - want) < 0.10 else f"   <-- EXPECTED ~{want:.3f}"
        print(f"  {name}: n={len(sub)}  prevalence={prev:.3f} (natural {want:.3f})"
              f"  unique questions={len({i['question_id'] for i in sub})}{flag}")

    # Prompt length: completions carry <think> traces and can be long. vLLM
    # truncates silently past max_model_len, so measure headroom now.
    # We tokenize with a real tokenizer where possible -- the char/3.5 heuristic
    # under-counts LaTeX-heavy math badly enough to matter at the tail.
    lens, how = [], "char/3.5 estimate"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")   # ungated, representative
        longest_arm = max(C.ARMS.values(), key=len)
        for i in items:
            body = longest_arm.replace("{question}", i["question"]) \
                              .replace("{answer}", i["answer"]) \
                              .replace("{completion}", i["completion"])
            lens.append(len(tok(body)["input_ids"]))
        how = "measured (Qwen3-8B tokenizer, longest prompt arm)"
    except Exception as e:
        print(f"\n  [tokenizer unavailable: {type(e).__name__}; falling back to estimate]")
        lens = [int((len(i["question"]) + len(i["answer"]) +
                     len(i["completion"]) + 1200) / 3.5) for i in items]
    lens.sort()
    q = lambda f: lens[min(int(f * len(lens)), len(lens) - 1)]
    budget = C.MAX_MODEL_LEN - C.MAX_TOKENS
    print(f"\n  prompt tokens ({how}):")
    print(f"    median={q(.5)}  p90={q(.9)}  p99={q(.99)}  max={lens[-1]}")
    print(f"    input budget = MAX_MODEL_LEN({C.MAX_MODEL_LEN}) - MAX_TOKENS"
          f"({C.MAX_TOKENS}) = {budget}")
    over = sum(1 for x in lens if x > budget)
    if over:
        print(f"  !! {over} item(s) exceed the budget and WOULD BE TRUNCATED.")
        print(f"     Raise MAX_MODEL_LEN in config.py above {lens[-1] + C.MAX_TOKENS},")
        print(f"     or drop those items and say so in the paper.")
    else:
        print(f"    headroom on the longest item: {budget - lens[-1]} tokens  OK")

    # Self-evaluation: some completions were generated by models that are also
    # judges in our roster. A judge grading its own output may be biased.
    try:
        import config as _c
        judges = {m["repo"].split("/")[-1].lower() for m in _c.MODELS}
        overlap = {}
        for i in items:
            cm = (i.get("completion_model") or "").lower()
            if cm and any(cm == j or cm in j or j in cm for j in judges):
                overlap[cm] = overlap.get(cm, 0) + 1
        if overlap:
            print("\n  SELF-EVALUATION OVERLAP (judge also generated these completions):")
            for k, v in sorted(overlap.items(), key=lambda x: -x[1]):
                print(f"    {k}: {v} items ({100*v/len(items):.1f}%)")
            print("    -> report self- vs other-generated error rates separately for")
            print("       that judge, or exclude and state it. Do not ignore silently.")
    except Exception:
        pass
    print("\n!! This file contains benchmark text. It is written to work/, which is")
    print("   gitignored, and must not be committed or redistributed. The release")
    print("   ships item identifiers (data/item_ids.csv) instead, so the benchmark's")
    print("   upstream license chain never attaches to anything we publish.")
    print("\nNext: copy collection/ and work/pilot_items.jsonl to the GPU machine, then")
    print("      python 02_vllm_collect.py --check")


if __name__ == "__main__":
    main()
