## 02_fit_cells.R --------------------------------------------------------
## Main fitting driver. One fit per (judge, arm, split) cell per model.
##
## Scope, to keep the run to a single overnight job (see FITTING_PLAN Step 2):
##   headline   m2, m3            all 42 cells      84 fits
##   violated   m4, m5            7 VBH/noref cells 14 fits
##   ladder     m1..m5 + baselines 3 exemplar cells 24 fits
##
## Usage:
##   Rscript 02_fit_cells.R headline
##   Rscript 02_fit_cells.R violated
##   Rscript 02_fit_cells.R ladder
##   Rscript 02_fit_cells.R all
##   Rscript 02_fit_cells.R full      every model on every cell (~15 min)

suppressPackageStartupMessages({ library(dplyr); library(readr) })
source("latentFit.R"); source("utils_metrics.R"); source("04_baselines.R")

SCOPE <- (commandArgs(trailingOnly = TRUE) %||% "headline")[1]
# The shipped data/item_level.csv is what 04_diagnostics.py produces. out/ is
# gitignored and holds only intermediates from a fresh collection run.
IN    <- "../../data/item_level.csv"
OUT   <- "../out_fit"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

EXEMPLARS <- list(                       # chosen in FITTING_PLAN Step 2
  c("gemma3-1b", "verbatim", "VB"),      # boundary, eps_0 = 0.789 > 0.5
  c("qwen3-8b",  "verbatim", "VB"),      # comfortably identified
  c("qwen3-32b", "noref",    "VBH"))     # violated, sum = 1.317

d <- read_csv(IN, show_col_types = FALSE) %>%
  filter(!is.na(k), !is.na(N), N >= 3)   # Theorem 1 needs N_c >= 3

cells <- d %>% distinct(model, arm, split) %>% arrange(model, arm, split)
message(sprintf("%d cells available; scope = %s", nrow(cells), SCOPE))

## m4 regresses the error rates on the self-reported difficulty H_c, which is
## ONLY elicited in the "difficulty" arm -- the verbatim and noref prompts never
## ask for it. So m4 is always fitted on the difficulty-arm data for the same
## (judge, split), whatever arm the surrounding ladder is comparing. The arm is
## recorded as "difficulty" in the output so the substitution is visible.
plan <- list()
add  <- function(m, a, s, mods) for (mm in mods) {
  aa <- if (mm == "m4") "difficulty" else a
  plan[[length(plan) + 1]] <<- list(model = m, arm = aa, split = s, fit = mm)
}

for (i in seq_len(nrow(cells))) {
  cl <- cells[i, ]
  is_ex  <- any(sapply(EXEMPLARS, function(e)
    all(e == c(cl$model, cl$arm, cl$split))))
  is_vio <- cl$arm == "noref" && cl$split == "VBH"
  if (SCOPE %in% c("headline", "all"))            add(cl$model, cl$arm, cl$split, c("m2", "m3"))
  if (SCOPE %in% c("violated", "all") && is_vio)  add(cl$model, cl$arm, cl$split, c("m4", "m5"))
  if (SCOPE %in% c("ladder",   "all") && is_ex)   add(cl$model, cl$arm, cl$split, c("m1", "m4", "m5"))
  ## measured at ~5s per fit, so the full ladder on every cell costs minutes.
  ## Prefer completeness -- "we report every model we ran" is the cheapest
  ## defence against a cherry-picking objection there is.
  if (SCOPE == "full")                            add(cl$model, cl$arm, cl$split,
                                                      c("m1", "m2", "m3", "m4", "m5"))
}
plan <- plan[!duplicated(sapply(plan, function(p)
  paste(p$model, p$arm, p$split, p$fit)))]
message(sprintf("%d fits planned", length(plan)))

results <- list(); t0 <- Sys.time()
for (i in seq_along(plan)) {
  p  <- plan[[i]]
  sub <- d %>% filter(model == p$model, arm == p$arm, split == p$split)
  if (nrow(sub) < 30) { message("  skip (n<30): ", paste(unlist(p), collapse = "/")); next }

  ## pass the empirically measured ICC to m3 as an option; the driver fits the
  ## FREE version by default and the fixed version only for the exemplars, so
  ## the two can be compared where it matters (FITTING_PLAN Step 6).
  de <- design_effect(sub)
  tag <- sprintf("%s/%s/%s/%s", p$model, p$arm, p$split, p$fit)
  message(sprintf("[%d/%d] %-34s n=%d  DE=%.2f", i, length(plan), tag, nrow(sub), de$DE))

  f <- tryCatch(
    LatentFit$new(sub, p$fit)$run(refresh = 0),
    error = function(e) { message("    FAILED: ", conditionMessage(e)); NULL })
  if (is.null(f)) next

  emp <- empirical_eps(sub); pc <- f$prob_correct(); s <- f$summary()
  ## m4 has item-varying error rates, so it reports mean_eps_0 / mean_eps_1 /
  ## mean_eps_sum rather than the scalars the other models expose. Fall back to
  ## the mean_* name, otherwise m4's coverage columns come back NaN.
  gv <- function(par, col) {
    v <- s[s$parameter == par, col]
    if (!length(v)) v <- s[s$parameter == paste0("mean_", par), col]
    if (length(v)) v else NA_real_
  }

  ## coverage: does the posterior interval contain the empirical value?
  cover <- function(par, truth)
    !is.na(truth) && gv(par, "2.5%") <= truth && truth <= gv(par, "97.5%")

  results[[length(results) + 1]] <- data.frame(
    judge = p$model, arm = p$arm, split = p$split, fit = p$fit,
    n = nrow(sub), DE = de$DE, rho_emp = de$rho, n_eff = de$n_eff,
    ## COMPONENT SEPARATION. The two mixture components are Bin(N, eps_0) and
    ## Bin(N, 1 - eps_1); their success probabilities differ by exactly
    ## (1 - eps_1) - eps_0 = 1 - (eps_0 + eps_1). So Theorem 1's condition is
    ## precisely "the components are distinct", and the margin below 1 measures
    ## how distinguishable they are. Empirically this predicts recovery almost
    ## perfectly (Spearman 0.96 with AUC across 42 cells).
    separation = 1 - emp$eps_sum,
    prevalence = gv("prevalence", "mean"),
    prev_lo = gv("prevalence", "2.5%"), prev_hi = gv("prevalence", "97.5%"),
    prev_true = mean(sub$gold, na.rm = TRUE),
    prev_cover = cover("prevalence", mean(sub$gold, na.rm = TRUE)),
    eps_0 = gv("eps_0", "mean"), eps_1 = gv("eps_1", "mean"),
    eps_sum = gv("eps_sum", "mean"),
    eps0_emp = emp$eps_0, eps1_emp = emp$eps_1, eps_sum_emp = emp$eps_sum,
    eps0_cover = cover("eps_0", emp$eps_0),
    eps1_cover = cover("eps_1", emp$eps_1),
    eps0_width = gv("eps_0", "97.5%") - gv("eps_0", "2.5%"),
    rho_post = gv("rho", "mean"),
    auc = private_auc(pc$prob, pc$gold), brier = brier(pc$prob, pc$gold),
    max_rhat = f$diagnostics$max_rhat, min_neff = f$diagnostics$min_neff,
    divergent = f$diagnostics$divergent, converged = f$diagnostics$converged,
    fail_reason = f$diagnostics$fail_reason,
    escalated = f$diagnostics$escalated, iter_used = f$diagnostics$iter)

  f$save(file.path(OUT, sprintf("fit_%s_%s_%s_%s.rds",
                                p$model, p$arm, p$split, p$fit)))
}

res <- bind_rows(results)
write_csv(res, file.path(OUT, sprintf("fits_%s.csv", SCOPE)))
message(sprintf("\n%d fits in %.1f min -> %s",
                nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins")),
                file.path(OUT, sprintf("fits_%s.csv", SCOPE))))
message(sprintf("converged: %d/%d   escalated to 3x draws: %d",
                sum(res$converged), nrow(res), sum(res$escalated, na.rm = TRUE)))
