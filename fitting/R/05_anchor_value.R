## 05_anchor_value.R -----------------------------------------------------
## FIGURE 1: what are ground-truth anchors actually worth?
##
## Renamed from 05_anchor_curve.R and widened in scope. The original ran only on
## the violated cells, which presumes anchors matter for IDENTIFICATION. That is
## one of three possible worlds:
##
##   World 1  mixture identified, violations found
##            -> anchors are NECESSARY in the violated regime
##   World 2  mixture identified, no violations
##            -> anchors are OPTIONAL; the question is what they buy
##   World 3  mixture confounded with overdispersion at high rho
##            -> anchors are necessary for a DIFFERENT reason
##
## Step 0 decides which. This script serves all three, because in every world the
## practitioner's question is the same: given my judge's error rate and
## dispersion, how many labels should I collect?
##
## Three outcomes are recorded per setting:
##   precision  1 / posterior variance of the prevalence -- the natural currency
##   accuracy   |estimated - true| prevalence, and held-out AUC
##   EXCHANGE RATE  how many UNLABELLED items one anchor is worth
##
## The exchange rate is the number that makes this actionable. "Anchors reduce
## uncertainty" is obvious a priori; "one anchor is worth ~12 unlabelled items,
## rising to ~40 when the error sum exceeds 0.8" is a budgeting decision.
##
## Usage: Rscript 05_anchor_value.R

suppressPackageStartupMessages({ library(dplyr); library(readr) })
source("latentFit.R"); source("utils_metrics.R")

IN  <- "../../data/item_level.csv"
OUT <- "../out_fit"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

FRACS   <- c(0, 0.05, 0.10, 0.20, 0.35, 0.50)   # 0 = unsupervised reference
W_GRID  <- c(1, 3, NA)                          # NA = floor(n_U/n_L), the default
REPS    <- 5
SUBSAMP <- c(0.35, 0.50, 0.70, 1.00)            # unlabelled-only calibration sweep

d <- read_csv(IN, show_col_types = FALSE) %>% filter(!is.na(gold), N >= 3)

## Select cells spanning the error-sum range, not just the violated ones. This is
## the axis the paper's claim is conditioned on, so it must be sampled.
cells <- d %>% group_by(model, arm, split) %>%
  summarise(n = n(), .groups = "drop") %>% filter(n >= 60)
cells$eps_sum <- mapply(function(m, a, s) {
  sub <- d %>% filter(model == m, arm == a, split == s)
  empirical_eps(sub)$eps_sum
}, cells$model, cells$arm, cells$split)

## ~9 cells: three low, three mid, three high, so the exchange rate can be
## regressed on the error sum rather than reported at one point.
cells <- cells %>% arrange(eps_sum) %>%
  mutate(band = cut(eps_sum, breaks = c(-Inf, 0.35, 0.8, Inf),
                    labels = c("low", "mid", "high"))) %>%
  group_by(band) %>% slice_head(n = 3) %>% ungroup()
message("cells selected (spanning the error-sum range):")
print(cells %>% select(model, arm, split, n, eps_sum, band))

fit_one <- function(sub, frac, w, seed) {
  m <- if (frac == 0) "m3" else "m5"     # frac 0 => unsupervised reference
  f <- tryCatch(
    LatentFit$new(sub, m, anchor_frac = max(frac, 1e-9),
                  W = if (is.na(w)) NULL else w, seed = seed)$run(refresh = 0),
    error = function(e) NULL)
  if (is.null(f)) return(NULL)
  s <- f$summary(); pc <- f$prob_correct()
  gv <- function(p, col) { v <- s[s$parameter == p, col]; if (length(v)) v else NA_real_ }
  list(prev = gv("prevalence", "mean"),
       prev_sd = gv("prevalence", "sd"),
       width = gv("prevalence", "97.5%") - gv("prevalence", "2.5%"),
       eps_sum = gv("eps_sum", "mean"), rho = gv("rho", "mean"),
       auc = private_auc(pc$prob, pc$gold), converged = f$diagnostics$converged)
}

## ---- A. anchor sweep ---------------------------------------------------
rows <- list()
for (i in seq_len(nrow(cells))) {
  cl  <- cells[i, ]
  sub <- d %>% filter(model == cl$model, arm == cl$arm, split == cl$split)
  for (fr in FRACS) for (w in (if (fr == 0) NA else W_GRID)) for (r in seq_len(REPS)) {
    o <- fit_one(sub, fr, w, 20260807 + r); if (is.null(o)) next
    rows[[length(rows) + 1]] <- data.frame(
      judge = cl$model, arm = cl$arm, split = cl$split, band = cl$band,
      eps_sum_emp = cl$eps_sum, n_total = nrow(sub),
      frac = fr, n_anchor = round(fr * nrow(sub)),
      W = ifelse(is.na(w), "auto", as.character(w)), rep = r,
      prev = o$prev, prev_sd = o$prev_sd, width = o$width,
      prev_true = mean(sub$gold), abs_err = abs(o$prev - mean(sub$gold)),
      rho_post = o$rho, auc = o$auc, converged = o$converged)
    if (fr == 0) break   # only one W applies to the unsupervised reference
  }
}
anchors <- bind_rows(rows)
write_csv(anchors, file.path(OUT, "anchor_value.csv"))

## ---- B. unlabelled-only calibration, to price the exchange rate --------
## Fit m3 (no anchors) on subsamples of the SAME cells. Precision as a function
## of unlabelled n is the ruler against which anchor precision is measured.
cal <- list()
for (i in seq_len(nrow(cells))) {
  cl  <- cells[i, ]
  full <- d %>% filter(model == cl$model, arm == cl$arm, split == cl$split)
  for (fs in SUBSAMP) for (r in seq_len(REPS)) {
    set.seed(20260807 + 100 * r)
    sub <- full[sample.int(nrow(full), max(30, round(fs * nrow(full)))), ]
    o <- fit_one(sub, 0, NA, 20260807 + r); if (is.null(o)) next
    cal[[length(cal) + 1]] <- data.frame(
      judge = cl$model, arm = cl$arm, split = cl$split, band = cl$band,
      eps_sum_emp = cl$eps_sum, n_unlabelled = nrow(sub), rep = r,
      prev_sd = o$prev_sd, converged = o$converged)
  }
}
calib <- bind_rows(cal)
write_csv(calib, file.path(OUT, "anchor_calibration.csv"))

## ---- C. the exchange rate ----------------------------------------------
## Posterior precision for a scalar mean grows ~linearly in n, so fit
##      1 / prev_sd^2  ~  b * n_unlabelled
## per cell on the unlabelled-only runs, then ask how much n that same slope
## would need to deliver the precision gain a given anchor count actually gave.
exch <- list()
for (i in seq_len(nrow(cells))) {
  cl <- cells[i, ]
  cc <- calib %>% filter(judge == cl$model, arm == cl$arm, split == cl$split, converged)
  aa <- anchors %>% filter(judge == cl$model, arm == cl$arm, split == cl$split,
                           converged, W == "auto")
  if (nrow(cc) < 4 || !any(aa$frac == 0)) next
  b <- coef(lm(I(1 / prev_sd^2) ~ 0 + n_unlabelled, data = cc))[["n_unlabelled"]]
  base <- mean(1 / aa$prev_sd[aa$frac == 0]^2)
  for (fr in setdiff(FRACS, 0)) {
    a <- aa %>% filter(frac == fr)
    if (!nrow(a)) next
    gain <- mean(1 / a$prev_sd^2) - base                  # precision added
    exch[[length(exch) + 1]] <- data.frame(
      judge = cl$model, arm = cl$arm, split = cl$split, band = cl$band,
      eps_sum_emp = cl$eps_sum, n_anchor = a$n_anchor[1],
      precision_gain = gain,
      equiv_unlabelled = gain / b,                        # items that gain equals
      items_per_anchor = (gain / b) / a$n_anchor[1])      # THE EXCHANGE RATE
  }
}
exchange <- bind_rows(exch)
write_csv(exchange, file.path(OUT, "anchor_exchange_rate.csv"))

## ---- report ------------------------------------------------------------
cat("\n", strrep("=", 74), "\nANCHOR VALUE BY ERROR-SUM BAND\n", strrep("=", 74), "\n", sep = "")
print(anchors %>% filter(converged, W == "auto") %>% group_by(band, n_anchor) %>%
  summarise(cells = n(), width = mean(width, na.rm = TRUE),
            abs_err = mean(abs_err, na.rm = TRUE),
            auc = mean(auc, na.rm = TRUE), .groups = "drop"))

cat("\n", strrep("=", 74), "\nEXCHANGE RATE: unlabelled items one anchor is worth\n",
    strrep("=", 74), "\n", sep = "")
print(exchange %>% group_by(band, n_anchor) %>%
  summarise(cells = n(), items_per_anchor = mean(items_per_anchor, na.rm = TRUE),
            .groups = "drop"))
cat("\n  This is the number a practitioner budgets against. If one anchor is worth\n",
    "  ~5 unlabelled items, labelling is rarely worth the annotation cost; if it\n",
    "  is worth ~50, it usually is. Expect the rate to RISE with the error sum.\n", sep = "")

cat("\n", strrep("=", 74), "\nTEMPERING WEIGHT W SENSITIVITY\n", strrep("=", 74), "\n", sep = "")
print(anchors %>% filter(converged, frac > 0) %>% group_by(W, n_anchor) %>%
  summarise(width = mean(width, na.rm = TRUE), abs_err = mean(abs_err, na.rm = TRUE),
            .groups = "drop"))
cat("\n  W = 1 is an ordinary Bayesian posterior. Larger W defines a power\n",
    "  posterior whose intervals are not calibrated in the strict sense -- if\n",
    "  width falls with W while abs_err does not, that is the overconfidence\n",
    "  both NeurIPS reviewers asked about, made visible.\n", sep = "")
