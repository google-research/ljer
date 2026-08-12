"""Step 4 -- the five decisions the pilot exists to make. Laptop only.

  D1  Is there within-item variance at all?        (go / no-go for the study)
  D2  Do observed error rates match prediction?    (validates the roster choice)
  D3  Does self-reported difficulty track error?   (Reviewer uhNj Q1, real data)
  D4  Does asking for difficulty change verdicts?  (justifies the arm design)
  D5  Does removing the reference push a judge into the violated regime?

  python 04_diagnostics.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"        # gitignored
DATA = ROOT / "data"      # ships: parsed verdicts and the modelling input
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

PRED = {m["key"]: m["pred_sum"] for m in C.MODELS}
ORDER = [m["key"] for m in C.MODELS]


def load():
    df = pd.read_csv(DATA / "ratings.csv")
    keep = df.status.isin(["ok", "ok_in_reasoning", "fallback"])
    print(f"rounds: {len(df)} total, {keep.sum()} parsed ({100*keep.mean():.1f}%)")
    return df[keep].copy()


def agg(df):
    """One row per (item, model, arm): k_c, N_c, gold, mean difficulty."""
    return df.groupby(["item_id", "split", "gold", "model", "arm", "native_reasoner"],
                      as_index=False).agg(k=("verdict", "sum"), N=("verdict", "size"),
                                          H=("difficulty", "mean"))


def eps(sub):
    """(eps_0, eps_1) on a balanced subset, using items as the unit."""
    neg, pos = sub[sub.gold == 0], sub[sub.gold == 1]
    e0 = (neg.k / neg.N).mean() if len(neg) else np.nan
    e1 = 1 - (pos.k / pos.N).mean() if len(pos) else np.nan
    return e0, e1


def d1(a):
    print("\n" + "=" * 74)
    print("D1  WITHIN-ITEM VARIANCE   (go / no-go -- no variance, no method)")
    print("=" * 74)
    print(f"  {'model':<13}{'arm':<11}{'rsnr':>5}{'n':>5}{'unanim':>8}{'binom':>7}{'excess':>8}  verdict")
    for m in ORDER:
        for arm in C.ARMS:
            s = a[(a.model == m) & (a.arm == arm)]
            if s.empty:
                continue
            N = s.N.median()
            unan = ((s.k == 0) | (s.k == s.N)).mean()
            p = (s.k / s.N).mean()
            e = min(p, 1 - p)
            binom = e ** N + (1 - e) ** N
            v = "NO-GO" if unan > .90 else "thin" if unan > .80 else "ok"
            if unan - binom > .15:
                v += "/clustered"
            r = "yes" if s.native_reasoner.iloc[0] else "-"
            print(f"  {m:<13}{arm:<11}{r:>5}{len(s):5d}{unan:8.3f}{binom:7.3f}"
                  f"{unan-binom:+8.3f}  {v}")
    print("  unanim>0.90  no signal: raise TEMPERATURE, or drop the model.")
    print("  excess>0.15  ratings correlated within item -- conditional independence")
    print("               is violated. Report it; two NeurIPS reviewers asked.")
    print("  rsnr=yes     model reasons natively. If these cluster more than the")
    print("               others, that is a finding: reasoning judges are more")
    print("               self-consistent and so harder to measure by resampling.")


def d2(a):
    print("\n" + "=" * 74)
    print("D2  ERROR RATES vs PRE-REGISTERED PREDICTION  (VerifyBench, pi=0.5)")
    print("=" * 74)
    print(f"  {'model':<13}{'arm':<11}{'eps_0':>7}{'eps_1':>7}{'sum':>7}{'pred':>7}{'diff':>8}  regime")
    for m in ORDER:
        for arm in C.ARMS:
            s = a[(a.model == m) & (a.arm == arm) & (a.split == "VB")]
            if s.empty:
                continue
            e0, e1 = eps(s)
            tot, pr = e0 + e1, PRED.get(m)
            reg = "VIOLATED" if tot >= 1 else "boundary" if tot > .85 else "identified"
            pt = f"{pr:7.3f}" if pr else "      -"
            dt = f"{tot-pr:+8.3f}" if pr else "       -"
            print(f"  {m:<13}{arm:<11}{e0:7.3f}{e1:7.3f}{tot:7.3f}{pt}{dt}  {reg}")
    print("  Predictions come from Yan et al. Table 2 and are PRE-REGISTERED.")
    print("  Divergence is expected (N=10 @ T=0.7 vs their single-shot). Report the")
    print("  comparison; never tune toward their numbers.")

    # VerifyBench-Hard: pi = 0.291, so no published anchor (their accuracy only
    # pins the sum when the split is balanced). But this is where the violated
    # regime is most likely to appear: the split skews toward wrong completions
    # and judges over-accept, which inflates eps_0.
    print()
    print("=" * 74)
    print("D2b  ERROR RATES ON VERIFYBENCH-HARD  (pi=0.291, no anchor)")
    print("=" * 74)
    print(f"  {'model':<13}{'arm':<11}{'eps_0':>7}{'eps_1':>7}{'sum':>7}{'n+':>5}{'n-':>5}  regime")
    for m in ORDER:
        for arm in C.ARMS:
            s2 = a[(a.model == m) & (a.arm == arm) & (a.split == "VBH")]
            if s2.empty:
                continue
            e0, e1 = eps(s2)
            tot = e0 + e1
            npos, nneg = (s2.gold == 1).sum(), (s2.gold == 0).sum()
            reg = "VIOLATED" if tot >= 1 else "boundary" if tot > .85 else "identified"
            print(f"  {m:<13}{arm:<11}{e0:7.3f}{e1:7.3f}{tot:7.3f}{npos:5d}{nneg:5d}  {reg}")
    print("  n+ is small here (~29 by design), so eps_1 is noisy -- SE ~0.09 at eps=0.3.")
    print("  Read this panel for the REGIME, not for precise rates.")
    print("  A VIOLATED cell is what the semi-supervised variant exists to handle.")


def d3(a):
    print("\n" + "=" * 74)
    print("D3  IS SELF-REPORTED DIFFICULTY CALIBRATED TO ERROR?")
    print("=" * 74)
    s = a[(a.arm == "difficulty") & a.H.notna()].copy()
    if s.empty:
        print("  no difficulty scores parsed"); return
    s["err"] = np.where(s.gold == 1, 1 - s.k / s.N, s.k / s.N)
    print(f"  {'model':<13}{'rho':>8}{'n':>6}   realized error by reported difficulty")
    for m in ORDER:
        g = s[s.model == m]
        if len(g) < 10:
            continue
        rho = g.H.corr(g.err, method="spearman")
        b = g.groupby(g.H.round())["err"].agg(["mean", "size"])
        txt = "  ".join(f"H{int(h)}:{r['mean']:.2f}({int(r['size'])})" for h, r in b.iterrows())
        print(f"  {m:<13}{rho:8.3f}{len(g):6d}   {txt}")
    print("  rho>0 supports the Theorem-2 assumption that H_c predicts error.")
    print("  rho~0 means H_c carries no signal. That is a legitimate publishable")
    print("  answer to whether a model knows what it finds hard.")


def d4(a):
    print("\n" + "=" * 74)
    print("D4  DOES ASKING FOR DIFFICULTY PERTURB THE VERDICT?")
    print("=" * 74)
    w = a.pivot_table(index=["item_id", "model"], columns="arm", values="k").dropna()
    if w.empty or not {"verbatim", "difficulty"} <= set(w.columns):
        print("  need both arms"); return
    for m in ORDER:
        g = w[w.index.get_level_values("model") == m]
        if g.empty:
            continue
        d = g["difficulty"] - g["verbatim"]
        print(f"  {m:<13} mean shift in k: {d.mean():+.3f}   |shift|>2 on {(d.abs()>2).mean():5.1%}")
    print("  Near-zero shift lets us report the verbatim arm as the comparability")
    print("  anchor and the difficulty arm as the modeling input, without caveat.")


def d5(a):
    print("\n" + "=" * 74)
    print("D5  DOES REMOVING THE REFERENCE ANSWER REACH THE VIOLATED REGIME?")
    print("=" * 74)
    print(f"  {'model':<13}{'verbatim sum':>14}{'noref sum':>11}{'shift':>9}  crosses 1?")
    for m in ORDER:
        row = {}
        for arm in ("verbatim", "noref"):
            s = a[(a.model == m) & (a.arm == arm) & (a.split == "VB")]
            if not s.empty:
                e0, e1 = eps(s); row[arm] = e0 + e1
        if len(row) < 2:
            continue
        v, n = row["verbatim"], row["noref"]
        cross = "YES -- usable lever" if (n >= 1 > v) else ("already violated" if v >= 1 else "no")
        print(f"  {m:<13}{v:14.3f}{n:11.3f}{n-v:+9.3f}  {cross}")
    print("  Yan et al. Table 3 reports 5-18% accuracy loss without the reference.")
    print("  This is a published ablation and a real deployment case, so using it to")
    print("  reach the violated regime is experimental design, not model-shopping.")


def d0_parsefail(raw_csv):
    """How much do unparsed rounds move eps? gemma3-1b drops 23% of rounds, and
    those failures are not random -- it is too small to follow the output format.
    Dropping them silently biases eps; report the bound instead."""
    print("\n" + "=" * 74)
    print("D0  PARSE-FAILURE SENSITIVITY  (are dropped rounds biasing eps?)")
    print("=" * 74)
    df = pd.read_csv(raw_csv)
    good = df.status.isin(["ok", "ok_in_reasoning", "fallback"])
    print(f"  {'model':<13}{'drop%':>7}{'sum|drop':>10}{'sum|yes=1':>11}{'sum|no=0':>10}  spread")
    for m in ORDER:
        sub = df[(df.model == m) & (df.arm == "verbatim") & (df.split == "VB")]
        if sub.empty:
            continue
        drop = 1 - good[sub.index].mean()
        out = []
        for label, fill in (("drop", None), ("yes", 1), ("no", 0)):
            d = sub.copy()
            if fill is None:
                d = d[good[sub.index]]
            else:
                d.loc[~good[sub.index], "verdict"] = fill
            d["verdict"] = pd.to_numeric(d.verdict, errors="coerce")
            d = d.dropna(subset=["verdict"])
            g = d.groupby(["item_id", "gold"], as_index=False).agg(
                k=("verdict", "sum"), N=("verdict", "size"))
            e0, e1 = eps(g)
            out.append(e0 + e1)
        print(f"  {m:<13}{100*drop:6.1f}%{out[0]:10.3f}{out[1]:11.3f}{out[2]:10.3f}"
              f"{max(out)-min(out):8.3f}")
    print("  Columns: drop = unparsed excluded (what D2 does); yes/no = unparsed")
    print("  counted as that verdict. A wide spread means the drop-them choice is")
    print("  load-bearing and must be stated in the paper, not buried.")


def d6_design_effect(a):
    """How much does N=10 actually buy, given the clustering D1 found?

    Under conditional independence Var(k/N) = p(1-p)/N. Observed variance is far
    larger, and the ratio IS the design effect:
        DE = Var_obs / Var_binomial = 1 + (N-1)*rho
        N_eff = N / DE
    This is the number that says what repeated sampling is worth, and it is the
    direct motivation for a beta-binomial likelihood.
    """
    print("\n" + "=" * 74)
    print("D6  DESIGN EFFECT -- WHAT IS N=10 ACTUALLY WORTH?")
    print("=" * 74)
    print(f"  {'model':<13}{'arm':<11}{'rsnr':>5}{'DE':>7}{'rho':>7}{'N_eff':>7}  interpretation")
    for m in ORDER:
        for arm in C.ARMS:
            s2 = a[(a.model == m) & (a.arm == arm) & (a.split == "VB")]
            if len(s2) < 20:
                continue
            des, ws = [], []
            for g in (0, 1):                       # condition on gold: p differs
                sub = s2[s2.gold == g]
                if len(sub) < 10:
                    continue
                N = sub.N.median()
                ph = (sub.k / sub.N).mean()
                if ph <= 0 or ph >= 1:
                    continue
                vb = ph * (1 - ph) / N
                vo = ((sub.k / sub.N) ** 2).mean() - ph ** 2
                if vb > 0:
                    des.append(max(vo / vb, 1.0)); ws.append(len(sub))
            if not des:
                continue
            DE = sum(d * w for d, w in zip(des, ws)) / sum(ws)
            N = s2.N.median()
            rho = (DE - 1) / (N - 1)
            neff = N / DE
            note = ("resampling nearly worthless" if neff < 1.5 else
                    "resampling buys little" if neff < 3 else
                    "resampling helps")
            r = "yes" if s2.native_reasoner.iloc[0] else "-"
            print(f"  {m:<13}{arm:<11}{r:>5}{DE:7.2f}{rho:7.3f}{neff:7.2f}  {note}")
    print("  N_eff is the number of INDEPENDENT ratings your 10 draws are worth.")
    print("  A plain binomial likelihood treats N_eff as 10, so its credible")
    print("  intervals are too narrow by roughly sqrt(DE). This is the empirical")
    print("  case for the beta-binomial extension.")


def _de(sub, N):
    """Design effect for one homogeneous stratum. None if too small."""
    if len(sub) < 8:
        return None, 0
    ph = (sub.k / sub.N).mean()
    if ph <= 0 or ph >= 1:
        return None, 0
    vb = ph * (1 - ph) / N
    vo = ((sub.k / sub.N) ** 2).mean() - ph ** 2
    return (max(vo / vb, 1.0), len(sub)) if vb > 0 else (None, 0)


def d7_h_absorbs(a):
    """How much of the overdispersion does the difficulty covariate explain?

    D6 measures excess variance within gold strata. If that excess is item-level
    heterogeneity in eps, then conditioning on H_c should shrink it -- that is
    exactly what Extension 1 (the H_c error-rate regression) is for. Whatever
    remains needs a beta-binomial or an item random effect.
    """
    print("\n" + "=" * 74)
    print("D7  DOES H_c ABSORB THE OVERDISPERSION?  (difficulty arm, VerifyBench)")
    print("=" * 74)
    print(f"  {'model':<13}{'DE|gold':>9}{'DE|gold,H':>11}{'absorbed':>10}  implication")
    for m in ORDER:
        s2 = a[(a.model == m) & (a.arm == "difficulty") & (a.split == "VB") & a.H.notna()]
        if len(s2) < 40:
            continue
        N = s2.N.median()
        # baseline: condition on gold only
        num = den = 0.0
        for g in (0, 1):
            de, w = _de(s2[s2.gold == g], N)
            if de:
                num += de * w; den += w
        if not den:
            continue
        de_gold = num / den
        # refined: condition on gold AND rounded difficulty
        num2 = den2 = 0.0
        for g in (0, 1):
            for h, sub in s2[s2.gold == g].groupby(s2.H.round()):
                de, w = _de(sub, N)
                if de:
                    num2 += de * w; den2 += w
        if not den2:
            continue
        de_gh = num2 / den2
        absorbed = 1 - (de_gh - 1) / (de_gold - 1) if de_gold > 1 else 0
        note = ("H_c explains most of it" if absorbed > .5 else
                "H_c explains some" if absorbed > .2 else
                "H_c explains almost none")
        print(f"  {m:<13}{de_gold:9.2f}{de_gh:11.2f}{100*absorbed:9.0f}%  {note}")
    print("  'absorbed' = share of the EXCESS variance removed by conditioning on H_c.")
    print("  High  -> the heterogeneous error model (Extension 1) is sufficient.")
    print("  Low   -> residual overdispersion remains; a beta-binomial likelihood or")
    print("           an item-level random effect is needed on top of it.")
    print("  Caveat: H_c is coarse (mostly 1-2 for the strong judges), so strata get")
    print("  small. Read the direction, not the decimal.")


def main():
    a = agg(load())
    a.to_csv(DATA / "item_level.csv", index=False)
    print(f"item-model-arm cells: {len(a)}")
    d0_parsefail(DATA / 'ratings.csv')
    d1(a); d2(a); d3(a); d4(a); d5(a); d6_design_effect(a); d7_h_absorbs(a)
    print(f"\nwrote {DATA/'item_level.csv'} -- input to the Stan models.")
    print("Columns k, N, gold, H are what latentLogit.R expects.")


if __name__ == "__main__":
    main()
