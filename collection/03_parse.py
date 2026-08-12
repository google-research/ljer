"""Step 3 -- parse raw judge text into binary verdicts and difficulty scores.

Runs on your LAPTOP (no GPU). Reads out/raw_ratings.jsonl and writes
data/ratings.csv, which holds verdicts only and is what the repository releases.

Two of the seven judges reason natively and wrap their answer in channel
markers -- gpt-oss uses the harmony format (analysis / final channels) and
gemma-4 emits a <|channel>thought block. We isolate the final channel before
looking for the verdict, so reasoning text is never mistaken for the answer.

Parse failures are DATA, not nuisance. A model that cannot follow the output
format has a high effective error rate, and silently dropping those rounds
biases eps downward. Every failure is recorded and reported per model.

  python 03_parse.py
"""
import csv, json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"        # gitignored: raw transcripts in, report out
DATA = ROOT / "data"      # ships: parsed verdicts, no benchmark text
DATA.mkdir(exist_ok=True)

# Take the LAST match: the prompt asks for the verdict at the end, and models
# often restate the required format mid-analysis before committing.
VERDICT  = re.compile(r"final\s*judgment\s*[:\-]?\s*\**\s*(yes|no)\b", re.I)
DIFF     = re.compile(r"difficulty\s*[:\-]?\s*\**\s*([1-5])\b", re.I)
FALLBACK = re.compile(r"\b(yes|no)\b[\s\.\*]*$", re.I)

# Reasoning wrappers. If a final/answer channel exists, everything before it is
# scratch work and must not be scanned for the verdict.
FINAL_CH = re.compile(r"<\|channel\|?>\s*(final|answer)\s*<\|?message\|?>", re.I)
THOUGHT  = re.compile(
    r"<\|channel\|?>\s*(thought|analysis)\s*<.*?>.*?(?=<\|channel\|?>|\Z)", re.I | re.S)
THINK    = re.compile(r"<think>.*?</think>", re.I | re.S)


def strip_reasoning(text):
    """Return (answer_text, had_reasoning_wrapper)."""
    m = list(FINAL_CH.finditer(text))
    if m:
        return text[m[-1].end():], True
    had = bool(THOUGHT.search(text) or THINK.search(text))
    if had:
        text = THINK.sub(" ", THOUGHT.sub(" ", text))
    return text, had


def parse_one(text):
    """-> (verdict 1/0/None, difficulty 1-5/None, status)"""
    if not text or not text.strip():
        return None, None, "empty"
    body, had_reasoning = strip_reasoning(text)
    d = DIFF.findall(body) or DIFF.findall(text)     # difficulty may precede the split
    diff = int(d[-1]) if d else None
    v = VERDICT.findall(body)
    if v:
        return (1 if v[-1].lower() == "yes" else 0), diff, "ok"
    v = VERDICT.findall(text)      # verdict inside an unclosed reasoning block
    if v:
        return (1 if v[-1].lower() == "yes" else 0), diff, "ok_in_reasoning"
    f = FALLBACK.findall(body.strip())
    if f:
        return (1 if f[-1].lower() == "yes" else 0), diff, "fallback"
    if had_reasoning:
        return None, diff, "reasoning_no_verdict"   # likely truncated mid-thought
    if len(text) > 4000:
        return None, diff, "truncated_or_rambling"
    return None, diff, "no_verdict"


def main():
    raw = OUT / "raw_ratings.jsonl"
    if not raw.exists():
        sys.exit(
            f"missing {raw}\n"
            "This step parses raw judge transcripts, which are produced by\n"
            "02_vllm_collect.py on a GPU machine and are not distributed with this\n"
            "repository. If you only want to reproduce the analysis, the parsed\n"
            "verdicts are already in data/ratings.csv and you can go straight to\n"
            "04_diagnostics.py.")

    rows, status = [], Counter()
    per_model = defaultdict(Counter)
    for line in raw.open():
        rec = json.loads(line)
        for rnd, text in enumerate(rec["raw"]):
            v, d, st = parse_one(text)
            status[st] += 1
            per_model[rec["model"]][st] += 1
            rows.append(dict(
                item_id=rec["item_id"], split=rec["split"], gold=rec["gold"],
                model=rec["model"], arm=rec["arm"], round=rnd,
                native_reasoner=int(rec.get("native_reasoner", False)),
                verdict=v if v is not None else "",
                difficulty=d if d is not None else "",
                n_chars=len(text or ""), status=st))

    if not rows:
        sys.exit("raw_ratings.jsonl is empty.")

    p = DATA / "ratings.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    good = {"ok", "ok_in_reasoning", "fallback"}
    lines = ["PARSE REPORT", "=" * 66, f"total rounds: {len(rows)}", ""]
    for st, c in status.most_common():
        lines.append(f"  {st:<24} {c:6d}  {100*c/len(rows):5.1f}%")
    lines += ["", f"  {'model':<14}{'n':>6}{'unparsed':>10}{'mean chars':>12}   flag"]
    for m, ctr in sorted(per_model.items()):
        tot = sum(ctr.values())
        bad = tot - sum(ctr[k] for k in good)
        chars = [r["n_chars"] for r in rows if r["model"] == m]
        flag = "  <-- INVESTIGATE" if bad / tot > 0.05 else ""
        lines.append(f"  {m:<14}{tot:6d}{bad:6d} ({100*bad/tot:4.1f}%)"
                     f"{sum(chars)/len(chars):11.0f}{flag}")
    lines += [
        "", "Notes:",
        "  ok_in_reasoning      verdict found only inside an unclosed reasoning block.",
        "  reasoning_no_verdict model was still thinking when it hit MAX_TOKENS.",
        "     If common for gpt-oss or gemma-4, raise MAX_TOKENS and re-run those two:",
        "     python 02_vllm_collect.py --only gptoss-20b,gemma4-26b",
        "  Above ~5% unparsed for any model, decide explicitly whether an unparseable",
        "  verdict is a missing value or an error -- and say which in the paper.",
    ]
    (OUT / "parse_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
