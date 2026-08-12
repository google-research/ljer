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

"""Rebuild the working item file from item identifiers.

The repository ships identifiers, not benchmark text. This script joins
data/item_ids.csv against the public benchmark and writes the working file that
the collection scripts expect.

Two reasons the repository is built this way. The benchmark's own license, and
the licenses of the 41 upstream datasets it draws on, stay attached to the
benchmark rather than following our release. And the repository stays at a few
megabytes rather than tens.

  python collection/fetch_items.py

Writes work/pilot_items.jsonl, which is gitignored. That file contains benchmark
text and must not be committed or redistributed.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "data" / "item_ids.csv"
OUT = ROOT / "work" / "pilot_items.jsonl"

FIELDS = dict(question="question", answer="answer",
              completion="completion", gold="gold_correct")


def main():
    if not IDS.exists():
        sys.exit(f"missing {IDS}")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    wanted = list(csv.DictReader(IDS.open()))
    print(f"{len(wanted)} items requested")

    ds = load_dataset("ZJU-REAL/VerifyBench")
    splits = {"VB": ds["VerifyBench"], "VBH": ds["VerifyBenchHard"]}

    OUT.parent.mkdir(exist_ok=True)
    n_written, mismatched = 0, 0
    with OUT.open("w") as fh:
        for w in wanted:
            # item_id encodes the split and the source row index, e.g. "VB-1026"
            split_key, idx = w["item_id"].rsplit("-", 1)
            row = splits[split_key][int(idx)]

            # The gold label is carried in item_ids.csv so the evaluation runs
            # without a second lookup. If it disagrees with the benchmark, the
            # benchmark has been revised since collection and the pinned results
            # no longer correspond to it.
            if int(bool(row[FIELDS["gold"]])) != int(w["gold"]):
                mismatched += 1

            fh.write(json.dumps(dict(
                item_id=w["item_id"], split=split_key,
                question_id=row.get("question_id"),
                source=row.get("source"),
                answer_type=row.get("answer_type"),
                completion_model=row.get("completion_model"),
                question=row[FIELDS["question"]],
                answer=row[FIELDS["answer"]],
                completion=row[FIELDS["completion"]],
                gold=int(w["gold"]))) + "\n")
            n_written += 1

    print(f"wrote {n_written} items -> {OUT}")
    if mismatched:
        print(f"!! {mismatched} gold labels disagree with the benchmark as fetched.")
        print("   The benchmark has changed since collection. Results in the paper")
        print("   correspond to the labels in data/item_ids.csv.")
    else:
        print("gold labels match the benchmark exactly")
    print("\nNOTE: this file contains benchmark text. It is gitignored, and must")
    print("not be committed or redistributed. The benchmark is MIT licensed and")
    print("available from its own repository.")


if __name__ == "__main__":
    main()
