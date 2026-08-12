## 00_step0_identifiability.R --------------------------------------------
## THE GATE. Run this before touching real data.
##
## Question: at the intra-class correlation the pilot measured (rho ~ 0.85), can
## a two-class latent mixture be distinguished from a SINGLE overdispersed
## population? Both produce U-shaped distributions of k_c with mass at the
## extremes. If the model cannot separate them, collecting more items does not
## help -- the confound lives in the likelihood, not the sample size.
##
## Two studies:
##   0.1 NULL   -- simulate ONE population, no latent classes. Does the model
##                 falsely "find" two?
##   0.2 POWER  -- simulate a genuine two-class mixture WITH overdispersion.
##                 Does the beta-binomial recover it? Does the binomial
##                 undercover?
##
## Usage:  Rscript 00_step0_identifiability.R [n_rep]

suppressPackageStartupMessages({ library(dplyr) })
source("latentFit.R")
source("utils_metrics.R")

args  <- commandArgs(trailingOnly = TRUE)
N_REP <- if (length(args)) as.integer(args[1]) else 40L
N_ROUND <- 10L; N_ITEM <- 250L
RHOS  <- c(0.30, 0.60, 0.85)
OUT   <- "../out_fit"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

## beta-binomial draw with mean eps and ICC rho
rbb <- function(n, N, eps, rho) {
  s <- (1 - rho) / rho
  rbinom(n, N, rbeta(n, eps * s, (1 - eps) * s))
}

## ---- 0.1 NULL: one population, no classes ------------------------------
message("=== 0.1 NULL RECOVERY: no latent classes exist ===")
null_res <- list()
for (rho in RHOS) for (r in seq_len(N_REP)) {
  set.seed(1000 * rho + r)
  k <- rbb(N_ITEM, N_ROUND, eps = 0.30, rho = rho)   # single population
  d <- data.frame(k = k, N = N_ROUND, gold = NA)
  f <- tryCatch(LatentFit$new(d, "m3")$run(refresh = 0), error = function(e) NULL)
  if (is.null(f)) next
  s <- f$summary()
  gv <- function(p, col) s[s$parameter == p, col]
  null_res[[length(null_res) + 1]] <- data.frame(
    rho_true = rho, rep = r,
    prevalence = gv("prevalence", "mean"),
    prev_lo = gv("prevalence", "2.5%"), prev_hi = gv("prevalence", "97.5%"),
    eps_sum = gv("eps_sum", "mean"), rho_est = gv("rho", "mean"),
    converged = f$diagnostics$converged, max_rhat = f$diagnostics$max_rhat,
    divergent = f$diagnostics$divergent, fail_reason = f$diagnostics$fail_reason)
}
null_df <- bind_rows(null_res)
saveRDS(null_df, file.path(OUT, "step0_null.rds"))

cat("\nA model fitted to data with NO classes should NOT be confident about a\n",
    "prevalence. Watch for prevalence pinned away from 0.5 with tight intervals.\n\n", sep = "")
print(null_df %>% group_by(rho_true) %>% summarise(
  n = n(), conv_rate = mean(converged),
  prev_mean = mean(prevalence), prev_sd = sd(prevalence),
  mean_ci_width = mean(prev_hi - prev_lo),
  pct_confident = mean((prev_hi - prev_lo) < 0.20),
  .groups = "drop"))
cat("\n  pct_confident = share of replicates giving a <0.20-wide 95% interval\n",
    "  for a prevalence that does not exist. High => FALSE POSITIVE => gate fails.\n", sep = "")

## ---- 0.2 POWER: real two-class mixture, with overdispersion -------------
message("\n=== 0.2 POWER RECOVERY: two classes DO exist ===")
TRUTH <- list(prevalence = 0.50, eps_0 = 0.15, eps_1 = 0.05)  # pilot-like, identified
pow_res <- list()
for (rho in RHOS) for (r in seq_len(N_REP)) {
  set.seed(2000 * rho + r)
  D <- rbinom(N_ITEM, 1, TRUTH$prevalence)
  k <- integer(N_ITEM)
  k[D == 1] <- rbb(sum(D == 1), N_ROUND, 1 - TRUTH$eps_1, rho)
  k[D == 0] <- rbb(sum(D == 0), N_ROUND,     TRUTH$eps_0, rho)
  d <- data.frame(k = k, N = N_ROUND, gold = D)
  for (m in c("m2", "m3")) {
    f <- tryCatch(LatentFit$new(d, m)$run(refresh = 0), error = function(e) NULL)
    if (is.null(f)) next
    s <- f$summary(); gv <- function(p, col) s[s$parameter == p, col]
    pc <- f$prob_correct()
    pow_res[[length(pow_res) + 1]] <- data.frame(
      rho_true = rho, rep = r, model = m,
      prev = gv("prevalence", "mean"),
      prev_cover = gv("prevalence", "2.5%") <= TRUTH$prevalence &&
                   TRUTH$prevalence <= gv("prevalence", "97.5%"),
      prev_width = gv("prevalence", "97.5%") - gv("prevalence", "2.5%"),
      e0 = gv("eps_0", "mean"),
      e0_cover = gv("eps_0", "2.5%") <= TRUTH$eps_0 &&
                 TRUTH$eps_0 <= gv("eps_0", "97.5%"),
      auc = private_auc(pc$prob, pc$gold),
      converged = f$diagnostics$converged,
      fail_reason = f$diagnostics$fail_reason)
  }
}
pow_df <- bind_rows(pow_res)
saveRDS(pow_df, file.path(OUT, "step0_power.rds"))

cat("\nCOVERAGE of nominal 95% intervals (should be ~0.95):\n\n")
print(pow_df %>% group_by(rho_true, model) %>% summarise(
  n = n(), conv = mean(converged),
  cover_prev = mean(prev_cover), cover_eps0 = mean(e0_cover),
  width_prev = mean(prev_width), auc = mean(auc), .groups = "drop"))
cat("\nWHY FITS FAILED (blank = converged):\n\n")
print(pow_df %>% count(model, fail_reason) %>% arrange(model, desc(n)))

cat("\n  Expected: m2 (binomial) undercovers badly as rho rises -- its intervals\n",
    "  ignore the overdispersion. m3 (beta-binomial) should stay near 0.95.\n",
    "  If m3 also undercovers at rho = 0.85, the gate FAILS.\n", sep = "")

cat("\n=== GATE DECISION ===\n")
cat("  PASS  if null prevalence intervals stay wide AND m3 coverage ~0.95 at rho=0.85\n")
cat("  PASS-WITH-CAVEAT if only the unsupervised case fails (=> anchors necessary)\n")
cat("  FAIL  if m3 undercovers or the null study false-positives at rho=0.85\n")
cat("        -> rescope; see FITTING_PLAN.md Step 0.3\n")
