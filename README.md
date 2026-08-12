# Judge error rates

Code and data for measuring the error rates of LLM-based judges, and for
estimating population quantities from their noisy verdicts.

Accompanies *Estimating LLM Judge Error Rates and Outcome Prevalence with Limited
or No Gold Labels*. [CITATION AND LINK ADDED AT CAMERA-READY]

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

## What is here

**Data.** 52,500 binary judgments from seven open-weight judges on VerifyBench
and VerifyBench-Hard, ten independent samples per item under three prompt
conditions.

**Code.** The collection pipeline, five Stan models, and the R analysis that
produces every table in the paper.

## What is not here, and why

**No benchmark text.** `data/item_ids.csv` holds identifiers, not questions or
completions. Run `collection/fetch_items.py` to reconstruct the working file from
the public benchmark. This keeps the benchmark's own licensing with the benchmark,
where it belongs, and keeps this repository to a few megabytes.

**No model weights.** Checkpoints are pinned by commit hash in
`collection/config.py` and downloaded from their public repositories.

**No raw judge transcripts.** The released ratings are parsed verdicts. Raw
free-form judge output frequently quotes the benchmark text it is reasoning about,
so we do not redistribute it.

## Layout

```
collection/     collection pipeline, and the two scripts that convert
                between benchmark text and item identifiers
fitting/stan/   five models, m1 through m5
fitting/R/      fitting and evaluation
data/           ships. item identifiers and parsed verdicts. no benchmark text.
work/           gitignored. reconstructed benchmark text.
out/            gitignored. raw judge transcripts and fit objects.
```

The `work/` and `out/` split is deliberate. Everything containing benchmark text
lands in a gitignored directory, so `data/` is safe to commit by construction
rather than by vigilance.

## Quick start

Reproducing the analysis needs no GPU, and no download either. `data/` already
holds the parsed verdicts.

```bash
pip install -r requirements.txt
cd fitting/R
Rscript 02_fit_cells.R full               # ~15 min, 4 chains per fit
Rscript 03_evaluate.R                     # produces the paper's tables
```

`fitting/R/latentFit.R` uses cmdstanr where available and falls back to rstan.

To inspect the items themselves, rebuild them from the identifiers:

```bash
python collection/fetch_items.py          # writes work/pilot_items.jsonl
```

That file contains benchmark text and is gitignored. Do not redistribute it.

## Re-collecting the ratings

Needs one GPU with at least 80 GB of memory, roughly two hours, and about 200 GB
of disk for the weights.

```bash
export HF_TOKEN="..."        # one judge is a gated repository
export HF_HOME=/path/with/room     # see note below
python collection/01_prepare_data.py      # -> work/pilot_items.jsonl
python collection/02_vllm_collect.py --purge   # -> out/raw_ratings.jsonl
python collection/03_parse.py                  # -> data/ratings.csv
python collection/04_diagnostics.py            # -> data/item_level.csv
```

`--purge` deletes each model's weights after use, capping peak disk at the largest
single model rather than the sum, around 90 GB instead of 200 GB.

Set `HF_HOME` to a filesystem with room for the largest model. Some hosts mount
the working directory on a smaller volume than the root filesystem, and the
download then fails with a quota error rather than an out-of-space error, which
is a confusing way to lose an hour.

If you draw a different sample, regenerate the identifier file so the release
stays consistent with it:

```bash
python collection/make_item_ids.py        # -> data/item_ids.csv
```

## The models

| File | Likelihood | Error rates | Restriction | Supervision |
|---|---|---|---|---|
| `m1_binom_box.stan` | binomial | constant | each rate below 1/2 | none |
| `m2_binom_sum.stan` | binomial | constant | rates sum below 1 | none |
| `m3_betabinom_sum.stan` | beta-binomial | constant | rates sum below 1 | none |
| `m4_betabinom_hetero.stan` | beta-binomial | vary with difficulty | rates sum below 1 | none |
| `m5_betabinom_semisup.stan` | beta-binomial | constant | none | labelled subset |

`m1` is included so the paper can quantify the cost of the constraint it
imposes. Use `m3` or `m5`.

## Reproducibility

Checkpoints are pinned to commit hashes, item samples are seeded, and the prompts
are reproduced verbatim from the benchmark authors' published appendices. The
predicted error rates in `config.py` were fixed before any data was collected and
have not been revised, so predicted-versus-observed remains a genuine
out-of-sample comparison.

Sampling used temperature 0.7 and top-p 0.9 with ten draws per item. Judge error
rates are properties of a specific model, prompt and temperature configuration,
so figures reported here do not transfer to other configurations. Re-estimating
them for a new configuration is what the code is for.

## Licenses

Code is Apache 2.0. The ratings are [CC0 / CC BY 4.0]. The benchmark is MIT and
is fetched rather than redistributed. Judge model licenses are recorded per model
in `collection/config.py`.

## Citation

[ADDED AT CAMERA-READY]
