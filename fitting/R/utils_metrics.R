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

## utils_metrics.R -------------------------------------------------------
## Shared evaluation helpers. Sourced by 00_, 02_ and 03_.

## AUC by the Mann-Whitney identity; no external dependency.
private_auc <- function(prob, gold) {
  ok <- !is.na(gold) & !is.na(prob)
  p <- prob[ok]; g <- gold[ok]
  if (length(unique(g)) < 2) return(NA_real_)
  r <- rank(p)
  n1 <- sum(g == 1); n0 <- sum(g == 0)
  (sum(r[g == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

brier <- function(prob, gold) {
  ok <- !is.na(gold) & !is.na(prob); mean((prob[ok] - gold[ok])^2)
}

## Empirical error rates from gold -- the "truth" that model estimates are
## scored against. NOTE these are themselves estimates: on VerifyBench-Hard only
## ~24 positives support eps_1, giving SE ~0.09, so compare intervals to
## intervals rather than to a point (see FITTING_PLAN 0.5).
empirical_eps <- function(dat) {
  neg <- dat[dat$gold == 0, ]; pos <- dat[dat$gold == 1, ]
  e0 <- if (nrow(neg)) mean(neg$k / neg$N) else NA_real_
  e1 <- if (nrow(pos)) 1 - mean(pos$k / pos$N) else NA_real_
  list(eps_0 = e0, eps_1 = e1, eps_sum = e0 + e1,
       se_0 = if (nrow(neg)) sqrt(e0 * (1 - e0) / nrow(neg)) else NA_real_,
       se_1 = if (nrow(pos)) sqrt(e1 * (1 - e1) / nrow(pos)) else NA_real_,
       n_neg = nrow(neg), n_pos = nrow(pos))
}

## Design effect within gold strata -- the D6 statistic, recomputed here so the
## fitting code can pass rho_fixed to m3 without re-reading the diagnostics.
design_effect <- function(dat) {
  num <- den <- 0
  for (g in c(0, 1)) {
    s <- dat[dat$gold == g, ]
    if (nrow(s) < 8) next
    N <- stats::median(s$N); ph <- mean(s$k / s$N)
    if (ph <= 0 || ph >= 1) next
    vb <- ph * (1 - ph) / N
    vo <- mean((s$k / s$N)^2) - ph^2
    if (vb > 0) { num <- num + max(vo / vb, 1) * nrow(s); den <- den + nrow(s) }
  }
  if (den == 0) return(list(DE = NA_real_, rho = NA_real_, n_eff = NA_real_))
  DE <- num / den; N <- stats::median(dat$N)
  list(DE = DE, rho = (DE - 1) / (N - 1), n_eff = N / DE)
}
