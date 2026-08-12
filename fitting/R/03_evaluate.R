## 03_evaluate.R ---------------------------------------------------------
## Turns the fit tables into the artefacts the paper needs.
##
## The headline is TABLE 3: coverage of nominal 95% credible intervals,
## binomial vs beta-binomial. The pilot measured design effects of 5.5-9.4, so
## the binomial should undercover by roughly sqrt(DE) ~ 2.5-3x. Converting an
## assumption violation into a coverage number is what makes it actionable.
##
## Usage: Rscript 03_evaluate.R

suppressPackageStartupMessages({ library(dplyr); library(readr); library(tidyr) })
OUT <- "../out_fit"

res <- list.files(OUT, "^fits_.*\\.csv$", full.names = TRUE) %>%
  lapply(read_csv, show_col_types = FALSE) %>% bind_rows() %>% distinct()
if (!nrow(res)) stop("no fit tables in ", OUT, " -- run 02_fit_cells.R first")

hr <- function(t) cat("\n", strrep("=", 74), "\n", t, "\n", strrep("=", 74), "\n", sep = "")

## ---- convergence first: a non-converged fit is evidence, not a nuisance ----
hr("CONVERGENCE")
print(res %>% group_by(fit) %>% summarise(
  n = n(), converged = sum(converged), rate = mean(converged),
  escalated = sum(escalated, na.rm = TRUE),
  worst_rhat = max(max_rhat), any_divergent = sum(divergent > 0), .groups = "drop"))
cat("\n  'escalated' = fits that needed 3x draws (6000 iterations) to converge.\n",
    "  Step 0 showed this is an rhat/mixing issue, not step size: adapt_delta\n",
    "  0.99 did not help but 2000 -> 6000 draws took convergence from 2-3/5 to\n",
    "  5/5. Report the escalation count -- it is a real cost of the model.\n", sep = "")
cat("\n  A model that fails to converge at high rho is REPORTED as failing.\n",
    "  Given the identifiability concern, that is a result (FITTING_PLAN 8.2).\n", sep = "")

ok <- res %>% filter(converged)

## ---- TABLE 3a: SEPARATION, the strongest result ------------------------
hr("TABLE 3a -- COMPONENT SEPARATION PREDICTS RECOVERY")
cat("\n  The two mixture components are Bin(N, eps_0) and Bin(N, 1 - eps_1).\n",
    "  Their success probabilities differ by exactly\n",
    "      (1 - eps_1) - eps_0 = 1 - (eps_0 + eps_1)\n",
    "  so COMPONENT SEPARATION IS ONE MINUS THE ERROR SUM. Theorem 1's condition\n",
    "  eps_0 + eps_1 < 1 is therefore precisely 'the components are distinct',\n",
    "  and the margin below 1 measures HOW distinguishable they are.\n",
    "  Identification is not a binary gate -- it is a continuous quantity a\n",
    "  practitioner can estimate from a small labelled sample.\n\n", sep = "")

sep_tab <- ok %>% filter(fit == "m3", !is.na(separation), !is.na(auc)) %>%
  mutate(band = cut(separation, c(-Inf, 0.2, 0.5, 0.8, Inf),
                    labels = c("<0.2 (merged)", "0.2-0.5", "0.5-0.8", ">0.8 (distinct)"))) %>%
  group_by(band) %>%
  summarise(cells = n(), AUC = mean(auc, na.rm = TRUE),
            prev_err = mean(abs(prevalence - prev_true), na.rm = TRUE),
            sum_err = mean(abs(eps_sum - eps_sum_emp), na.rm = TRUE),
            .groups = "drop")
print(sep_tab)

sp <- ok %>% filter(fit == "m3", !is.na(separation), !is.na(auc))
cat(sprintf("\n  Spearman(separation, AUC) = %+.3f over %d cells\n",
            cor(sp$separation, sp$auc, method = "spearman"), nrow(sp)))
cat("  Below ~0.2 separation, AUC drops BELOW CHANCE -- the model does not merely\n",
    "  fail, it inverts the classes. That is the violated regime, and it appears\n",
    "  in cells where eps_0 + eps_1 never actually exceeds 1.\n", sep = "")
write_csv(sep_tab, file.path(OUT, "table3a_separation.csv"))

## ---- estimator bias ----------------------------------------------------
hr("ERROR-SUM BIAS: IS THE BINOMIAL SYSTEMATICALLY LOW?")
print(ok %>% filter(fit %in% c("m2", "m3")) %>%
  group_by(fit) %>%
  summarise(cells = n(),
            under = sum(eps_sum < eps_sum_emp, na.rm = TRUE),
            median_gap = median(eps_sum - eps_sum_emp, na.rm = TRUE),
            mean_abs_gap = mean(abs(eps_sum - eps_sum_emp), na.rm = TRUE),
            .groups = "drop"))
cat("\n  A universal one-sided gap is BIAS, not just miscalibration, and it lands\n",
    "  harder with readers than a coverage table: the binomial does not merely\n",
    "  understate its uncertainty, it understates the error rate itself.\n", sep = "")

## ---- TABLE 3: coverage, the headline ----------------------------------
hr("TABLE 3 -- COVERAGE OF NOMINAL 95% INTERVALS")
cov_tab <- ok %>%
  group_by(fit, split) %>%
  summarise(cells = n(),
            cover_prevalence = mean(prev_cover, na.rm = TRUE),
            cover_eps0 = mean(eps0_cover, na.rm = TRUE),
            cover_eps1 = mean(eps1_cover, na.rm = TRUE),
            mean_eps0_width = mean(eps0_width, na.rm = TRUE),
            .groups = "drop")
print(cov_tab)
cat("\n  Nominal is 0.95. Report the VB rows as the headline: on VBH only ~24\n",
    "  positives support the empirical eps_1 (SE ~0.09), so the TARGET is nearly\n",
    "  as uncertain as the interval being tested (FITTING_PLAN 0.5).\n", sep = "")

if (all(c("m2", "m3") %in% ok$fit)) {
  w <- ok %>% filter(fit %in% c("m2", "m3")) %>%
    select(judge, arm, split, fit, eps0_width) %>%
    pivot_wider(names_from = fit, values_from = eps0_width) %>%
    mutate(inflation = m3 / m2)
  cat(sprintf("\n  Interval width m3/m2: median %.2fx (expected ~sqrt(DE) ~ 2.5-3x)\n",
              median(w$inflation, na.rm = TRUE)))
}

## ---- TABLE 1 companion: estimated vs empirical error rates -------------
hr("ESTIMATED vs EMPIRICAL ERROR RATES")
print(ok %>% filter(fit %in% c("m2", "m3"), split == "VB") %>%
  transmute(judge, arm, fit,
            eps_sum_est = round(eps_sum, 3), eps_sum_emp = round(eps_sum_emp, 3),
            diff = round(eps_sum - eps_sum_emp, 3),
            prev_est = round(prevalence, 3), prev_true = round(prev_true, 3),
            auc = round(auc, 3)) %>% arrange(judge, arm, fit), n = 60)

## ---- TABLE 4: the 2x2 --------------------------------------------------
hr("TABLE 4 -- THE 2x2: SUPERVISION x REGIME")
tab4 <- ok %>%
  mutate(regime = ifelse(eps_sum_emp >= 1, "violated", "identified"),
         supervision = ifelse(fit == "m5", "semi-supervised", "unsupervised")) %>%
  filter(fit %in% c("m3", "m4", "m5")) %>%
  group_by(regime, supervision) %>%
  summarise(cells = n(), auc = mean(auc, na.rm = TRUE),
            prev_err = mean(abs(prevalence - prev_true), na.rm = TRUE),
            cover_prev = mean(prev_cover, na.rm = TRUE), .groups = "drop")
print(tab4)
cat("\n  Expect anchors to add little where the condition holds and to be\n",
    "  decisive where it does not. Both halves are the result.\n", sep = "")

## ---- does H_c still earn its place? ------------------------------------
if (all(c("m3", "m4") %in% ok$fit)) {
  hr("M3 vs M4 -- DOES THE DIFFICULTY COVARIATE STILL HELP?")
  print(ok %>% filter(fit %in% c("m3", "m4")) %>%
    select(judge, arm, split, fit, auc, brier, prev_cover) %>%
    pivot_wider(names_from = fit, values_from = c(auc, brier, prev_cover)) %>%
    mutate(auc_gain = auc_m4 - auc_m3))
  cat("\n  D7 found H_c absorbs 0-35% of the overdispersion, so expect a small\n",
      "  gain. Reporting that honestly is the finding.\n", sep = "")
}

## ---- box vs sum constraint ---------------------------------------------
if ("m1" %in% ok$fit) {
  hr("M1 vs M2 -- COST OF THE BOX CONSTRAINT")
  print(ok %>% filter(fit %in% c("m1", "m2")) %>%
    select(judge, arm, split, fit, eps_0, eps_1, eps_sum, eps_sum_emp, prev_cover) %>%
    arrange(judge, arm, fit))
  cat("\n  On gemma-3-1b the empirical eps_0 is 0.789, so M1's ceiling of 0.5 is\n",
      "  unreachable. The bias this induces is the point of including M1.\n", sep = "")
}

write_csv(cov_tab, file.path(OUT, "table3_coverage.csv"))
write_csv(tab4,    file.path(OUT, "table4_two_by_two.csv"))
message("\nwrote table3_coverage.csv and table4_two_by_two.csv to ", OUT)
