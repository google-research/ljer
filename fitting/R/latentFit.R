## latentFit.R -----------------------------------------------------------
## R6 wrapper for the model ladder, in the style of latentLogit.R.
##
## Differences from latentLogit.R, all deliberate:
##   * model selection is explicit ("m1".."m5") rather than inferred from the
##     data, so a misspecified fit is always a choice on the record;
##   * difficulty is STANDARDISED (mean 0, sd 1), not min-max rescaled.
##     latentLogit.R line ~240 min-max rescales, which makes the intercept and
##     slope dataset-dependent and outlier-sensitive; z-scoring keeps them
##     comparable across cells;
##   * priors are widened. The published LogNormal(0, 0.5) slope prior has a 95%
##     interval of (0.375, 2.66) while the simulation truths were 3.5-6.0, i.e.
##     at the 99.4th-99.98th percentile. Slopes here are Normal and may be
##     negative (Reviewer 79bE, Q3);
##   * every fit returns its convergence diagnostics, and a failing fit is
##     reported as failing rather than silently reseeded.

## BACKEND: cmdstanr if available, else rstan. The Stan code is identical.
## cmdstanr is preferred -- it ships its own toolchain and avoids the
## Rcpp/RcppEigen compile path, which is where rstan breaks on modern macOS
## (the classic symptom is Eigen/Core: fatal error: 'new' file not found,
## usually caused by a stale ~/.R/Makevars pinning -mmacosx-version-min).
suppressPackageStartupMessages({
  library(R6); library(dplyr)
})
BACKEND <- if (requireNamespace("cmdstanr", quietly = TRUE)) "cmdstanr" else "rstan"
if (BACKEND == "rstan") suppressPackageStartupMessages(library(rstan))
message("latentFit backend: ", BACKEND)

STAN_DIR <- file.path(dirname(sys.frame(1)$ofile %||% "."), "..", "stan")
`%||%` <- function(a, b) if (is.null(a)) b else a

LatentFit <- R6Class(
  "LatentFit",

  public = list(

    model_name = NULL, stan_file = NULL, data = NULL, stan_data = NULL,
    fit = NULL, diagnostics = NULL,

    ## dat: data.frame with columns k, N, gold, and optionally H
    ## model: one of "m1","m2","m3","m4","m5"
    initialize = function(dat, model = "m3", stan_dir = NULL,
                          anchor_frac = 0.10, W = NULL, seed = 20260807,
                          rho_fixed = NULL, priors = list()) {

      stopifnot(all(c("k", "N") %in% names(dat)))
      stopifnot(model %in% c("m1", "m2", "m3", "m4", "m5"))
      self$model_name <- model
      self$data <- dat

      files <- c(m1 = "m1_binom_box.stan",      m2 = "m2_binom_sum.stan",
                 m3 = "m3_betabinom_sum.stan",  m4 = "m4_betabinom_hetero.stan",
                 m5 = "m5_betabinom_semisup.stan")
      self$stan_file <- file.path(stan_dir %||% STAN_DIR, files[[model]])
      if (!file.exists(self$stan_file))
        stop("Stan file not found: ", self$stan_file)

      pr <- modifyList(list(
        mu_theta = 0, sigma_theta = 1.5,   # prevalence: weakly informative on logit
        a_v = 1, b_v = 1,                  # sum: uniform on (0,1)
        a_rho = 2, b_rho = 2,              # ICC: centred, allows the full range
        sigma_slope = 2                    # was LogNormal(0,0.5); far too tight
      ), priors)

      if (model == "m5") {
        ## Hold out an anchor subset, STRATIFIED ON THE GOLD LABEL.
        ##
        ## A simple random draw can miss a class entirely on a skewed split --
        ## VerifyBench-Hard is 291/709, so at a 5% fraction on 100 items a
        ## simple draw picks 5 items and lands all-negative about 16% of the
        ## time. Theorem 3 condition (i) requires at least one observation from
        ## each latent class; without it the anchors cannot break the
        ## label-switching symmetry and the model silently reverts to the
        ## unsupervised failure mode. Stratifying guarantees the condition holds.
        set.seed(seed)
        n <- nrow(dat)
        if (!"gold" %in% names(dat)) stop("m5 needs a 'gold' column for anchors")
        n_anchor <- max(2L, round(anchor_frac * n))
        pos <- which(dat$gold == 1); neg <- which(dat$gold == 0)
        if (!length(pos) || !length(neg))
          stop("m5 needs both classes present in the data")
        ## allocate proportionally, but never fewer than one per class
        n_pos <- max(1L, min(length(pos), round(n_anchor * length(pos) / n)))
        n_neg <- max(1L, min(length(neg), n_anchor - n_pos))
        idx <- c(sample(pos, n_pos), sample(neg, n_neg))
        L <- dat[idx, , drop = FALSE]; U <- dat[-idx, , drop = FALSE]
        stopifnot(length(unique(L$gold)) == 2)   # guaranteed by stratification
        w <- W %||% max(1, floor(nrow(U) / nrow(L)))   # matches latentLogit.R
        self$stan_data <- list(
          C_U = nrow(U), k_U = U$k, N_U = U$N,
          C_L = nrow(L), k_L = L$k, N_L = L$N, D_L = as.integer(L$gold),
          W = w, mu_theta = pr$mu_theta, sigma_theta = pr$sigma_theta,
          a_rho = pr$a_rho, b_rho = pr$b_rho)
        private$anchor_idx <- idx
      } else {
        sd_ <- list(C = nrow(dat), k = dat$k, N = dat$N,
                    mu_theta = pr$mu_theta, sigma_theta = pr$sigma_theta)
        if (model %in% c("m2", "m3"))  sd_ <- c(sd_, list(a_v = pr$a_v, b_v = pr$b_v))
        if (model %in% c("m3", "m4"))  sd_ <- c(sd_, list(a_rho = pr$a_rho, b_rho = pr$b_rho))
        if (model == "m3")
          sd_ <- c(sd_, list(fix_rho  = as.integer(!is.null(rho_fixed)),
                             rho_fixed = rho_fixed %||% 0.5))
        if (model == "m4") {
          if (!"H" %in% names(dat)) stop("m4 needs a difficulty column 'H'")
          h <- dat$H
          if (all(is.na(h))) stop("H is entirely missing")
          h[is.na(h)] <- mean(h, na.rm = TRUE)
          sd_ <- c(sd_, list(H = as.numeric(scale(h)),  # z-score, not min-max
                             sigma_slope = pr$sigma_slope))
        }
        self$stan_data <- sd_
      }
      invisible(self)
    },

    ## escalate: if the first pass does not meet the convergence criteria, refit
    ## ONCE at 3x the draws. Step 0 established that m3's failures are rhat, not
    ## step size -- adapt_delta 0.99 did not help, but 2000 -> 6000 iterations
    ## took convergence from 2-3/5 to 5/5 at both rho = 0.30 and rho = 0.85.
    ## Escalating only the failures costs ~1.7x rather than 3x, since roughly
    ## two thirds of fits already pass at 2000.
    ##
    ## The escalated fit is flagged, so the paper can report how many cells
    ## needed it. That number is itself a practical cost of the beta-binomial
    ## and belongs in the methods.
    run = function(chains = 4, iter = 2000, warmup = 1000, seed = 20260807,
                   adapt_delta = 0.95, max_treedepth = 12, refresh = 0,
                   escalate = TRUE, escalate_factor = 3) {
      if (BACKEND == "cmdstanr") {
        mod <- cmdstanr::cmdstan_model(self$stan_file, compile = TRUE)
        self$fit <- mod$sample(
          data = self$stan_data, chains = chains,
          parallel_chains = min(chains, parallel::detectCores()),
          iter_warmup = warmup, iter_sampling = iter - warmup,
          seed = seed, adapt_delta = adapt_delta,
          max_treedepth = max_treedepth, refresh = refresh,
          show_messages = FALSE)
      } else {
        self$fit <- rstan::stan(
          file = self$stan_file, data = self$stan_data,
          chains = chains, iter = iter, warmup = warmup, seed = seed,
          refresh = refresh,
          control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth))
      }
      self$diagnostics <- private$check(self$fit)
      self$diagnostics$escalated <- FALSE
      self$diagnostics$iter <- iter

      if (escalate && !self$diagnostics$converged) {
        it2 <- iter * escalate_factor
        message(sprintf("    not converged (%s); refitting at iter=%d",
                        self$diagnostics$fail_reason, it2))
        first <- self$diagnostics
        ok <- tryCatch({
          self$run(chains = chains, iter = it2, warmup = it2 / 2, seed = seed,
                   adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                   refresh = refresh, escalate = FALSE)
          TRUE
        }, error = function(e) { message("    escalation failed: ",
                                         conditionMessage(e)); FALSE })
        if (ok) {
          self$diagnostics$escalated <- TRUE
          self$diagnostics$iter <- it2
          self$diagnostics$first_pass_reason <- first$fail_reason
        } else {
          self$diagnostics <- first          # keep the honest first-pass result
        }
      }
      invisible(self)
    },

    ## tidy posterior summary for the parameters we report
    summary = function(pars = NULL) {
      if (is.null(self$fit)) stop("call $run() first")
      wanted <- pars %||% c("prevalence", "eps_0", "eps_1", "eps_sum", "rho", "s",
                            "mean_eps_0", "mean_eps_1", "mean_eps_sum",
                            "a_v", "b_v", "a_u", "b_u")
      if (BACKEND == "cmdstanr") {
        have <- intersect(wanted, self$fit$metadata()$stan_variables)
        ## quantile() returns NAMED values, so posterior uses those names
        ## ("2.5%") rather than the argument name -- unname() forces the column
        ## to be called exactly what we ask for, which is what the callers index.
        s <- self$fit$summary(
          variables = have, mean = mean, sd = stats::sd,
          lo  = ~ unname(stats::quantile(.x, 0.025)),
          med = ~ unname(stats::quantile(.x, 0.500)),
          hi  = ~ unname(stats::quantile(.x, 0.975)),
          rhat = posterior::rhat, ess = posterior::ess_bulk)
        data.frame(parameter = s$variable, mean = s$mean, sd = s$sd,
                   `2.5%` = s$lo, `50%` = s$med, `97.5%` = s$hi,
                   n_eff = s$ess, Rhat = s$rhat, check.names = FALSE)
      } else {
        have <- intersect(wanted, self$fit@model_pars)
        s <- rstan::summary(self$fit, pars = have,
                            probs = c(0.025, 0.5, 0.975))$summary
        out <- as.data.frame(s)[, c("mean", "sd", "2.5%", "50%", "97.5%",
                                    "n_eff", "Rhat")]
        out$parameter <- rownames(out); rownames(out) <- NULL
        out[, c("parameter", setdiff(names(out), "parameter"))]
      }
    },

    ## posterior P(D_c = 1 | k_c). For m5 this covers the HELD-OUT items only --
    ## anchors are excluded, so recovery is never scored on revealed labels.
    prob_correct = function() {
      pc <- if (BACKEND == "cmdstanr")
        as.numeric(self$fit$summary("prob_correct", mean = mean)$mean)
      else
        colMeans(rstan::extract(self$fit, "prob_correct")$prob_correct)
      if (self$model_name == "m5")
        list(prob = pc, gold = self$data$gold[-private$anchor_idx],
             anchor_idx = private$anchor_idx)
      else
        list(prob = pc, gold = self$data$gold, anchor_idx = integer(0))
    },

    loo = function() {
      if (BACKEND == "cmdstanr") return(self$fit$loo())
      ll <- loo::extract_log_lik(self$fit, "log_lik", merge_chains = FALSE)
      loo::loo(ll, r_eff = loo::relative_eff(exp(ll)))
    },

    ## persist a fit in a backend-appropriate way
    save = function(path) {
      if (BACKEND == "cmdstanr") self$fit$save_object(file = path)
      else saveRDS(self$fit, path)
      invisible(path)
    }
  ),

  private = list(

    anchor_idx = integer(0),

    ## A fit that fails these is REPORTED as failing. Given the identifiability
    ## concern (FITTING_PLAN Step 0), non-convergence is itself evidence.
    ## Convergence is judged on the MODEL PARAMETERS ONLY.
    ##
    ## prob_correct[1..C] and log_lik[1..C] are generated quantities, and for any
    ## item whose posterior probability is effectively constant across draws the
    ## ESS collapses to ~0 and Rhat comes back NaN. Taking min(ESS) over all
    ## variables therefore fails the whole fit for a reason that has nothing to
    ## do with whether the sampler explored the posterior. That is what produced
    ## the spurious 42-65% "non-convergence" in the first Step 0 run -- worst at
    ## rho = 0.3, which should have been the easiest case.
    check = function(fit) {
      drop <- "^(prob_correct|log_lik|lp__|eps_0\\[|eps_1\\[|u\\[|v\\[)"
      if (BACKEND == "cmdstanr") {
        s  <- fit$summary()
        s  <- s[!grepl(drop, s$variable), , drop = FALSE]
        d  <- fit$diagnostic_summary(quiet = TRUE)
        mr <- max(s$rhat, na.rm = TRUE)
        mn <- min(s$ess_bulk, na.rm = TRUE)
        div <- sum(d$num_divergent); td <- sum(d$num_max_treedepth)
        bfmi <- if (length(d$ebfmi)) min(d$ebfmi, na.rm = TRUE) else NA_real_
      } else {
        sm  <- rstan::summary(fit)$summary
        sm  <- sm[!grepl(drop, rownames(sm)), , drop = FALSE]
        sp  <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
        mr  <- max(sm[, "Rhat"],  na.rm = TRUE)
        mn  <- min(sm[, "n_eff"], na.rm = TRUE)
        div <- sum(sapply(sp, function(x) sum(x[, "divergent__"])))
        td  <- sum(sapply(sp, function(x) sum(x[, "treedepth__"] >= 12)))
        bfmi <- tryCatch(min(rstan::get_bfmi(fit)), error = function(e) NA_real_)
      }
      reasons <- c(
        if (!is.finite(mr) || mr >= 1.01) "rhat",
        if (!is.finite(mn) || mn <= 400)  "ess",
        if (div > 0)                       "divergent",
        if (!is.na(bfmi) && bfmi <= 0.2)   "bfmi")
      list(max_rhat = mr, min_neff = mn, divergent = div,
           treedepth_hits = td, min_bfmi = bfmi,
           fail_reason = if (length(reasons)) paste(reasons, collapse = "+") else "",
           converged = length(reasons) == 0)
    }
  )
)
