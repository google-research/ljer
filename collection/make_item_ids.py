"""Generate data/item_ids.csv, the file that ships in place of benchmark text.

Run once, on the machine that holds work/pilot_items.jsonl, before staging the
repository. The output is three columns and a few kilobytes, and contains no
question, reference answer, or completion text.

  python collection/make_item_ids.py

The gold label is included so that the evaluation runs without a second lookup.
It is the benchmark's own published label, so including it redistributes nothing
that the benchmark does not already publish, and it lets fetch_items.py detect
whether the benchmark has been revised since collection.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "work" / "pilot_items.jsonl"
OUT = ROOT / "data" / "item_ids.csv"

TEXT_FIELDS = ("question", "answer", "completion")


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}\nRun 01_prepare_data.py or fetch_items.py first.")
    rows = [json.loads(l) for l in SRC.open()]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "split", "gold"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("item_id", "split", "gold")})

    # Refuse to finish quietly if anything resembling text got through.
    written = OUT.read_text()
    longest = max((len(f) for line in written.split("\n")[1:] if line
                   for f in line.split(",")), default=0)
    if longest > 24:
        sys.exit(f"ABORT: a field of {longest} chars is present. Inspect {OUT} "
                 "before staging -- it should contain identifiers only.")

    print(f"wrote {len(rows)} rows -> {OUT} ({OUT.stat().st_size} bytes)")
    print(f"longest field: {longest} chars, so no free text is present")
    for name in ("VB", "VBH"):
        sub = [r for r in rows if r["split"] == name]
        if sub:
            print(f"  {name}: n={len(sub)}  correct completions="
                  f"{sum(r['gold'] for r in sub)}")
    print("\nSafe to commit. work/pilot_items.jsonl is NOT.")


if __name__ == "__main__":
    main()
