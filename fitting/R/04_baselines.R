## 04_baselines.R --------------------------------------------------------
## Non-Bayesian comparators: majority vote, Dawid-Skene (EM), and
## prediction-powered inference. All operate on the same (k, N, gold) cells.

source("utils_metrics.R")

## B0 -- majority vote. Ties broken toward "incorrect" (conservative).
majority_vote <- function(dat) {
  pred <- as.integer(dat$k / dat$N > 0.5)
  list(prevalence = mean(pred),
       prob = dat$k / dat$N,           # for AUC, use the vote share
       pred = pred,
       eps_0 = mean(pred[dat$gold == 0] == 1),
       eps_1 = mean(pred[dat$gold == 1] == 0))
}

## B1 -- Dawid-Skene by EM. With one annotator replicated N times, the
## sufficient statistic collapses to k and the confusion matrix reduces to
## (eps_0, eps_1). This IS the base model fitted by maximum likelihood, which is
## exactly why it is the right comparator: it isolates what the Bayesian
## treatment adds (uncertainty, priors, causal parameters) from what the latent
## class structure adds (nothing new).
dawid_skene <- function(dat, tol = 1e-8, max_iter = 500) {
  k <- dat$k; N <- dat$N; C <- length(k)
  p <- 0.5; e0 <- 0.2; e1 <- 0.2      # init away from the boundary
  for (it in seq_len(max_iter)) {
    l1 <- log(p)      + dbinom(k, N, 1 - e1, log = TRUE)
    l0 <- log(1 - p)  + dbinom(k, N,     e0, log = TRUE)
    m  <- pmax(l1, l0)
    z  <- exp(l1 - m) / (exp(l1 - m) + exp(l0 - m))     # E-step
    p_new  <- mean(z)                                    # M-step
    e1_new <- sum(z * (N - k)) / sum(z * N)
    e0_new <- sum((1 - z) * k) / sum((1 - z) * N)
    if (max(abs(c(p_new - p, e0_new - e0, e1_new - e1))) < tol) {
      p <- p_new; e0 <- e0_new; e1 <- e1_new; break
    }
    p <- p_new; e0 <- e0_new; e1 <- e1_new
  }
  ## EM has no protection against label switching; the published remedy is to
  ## initialise at the majority vote, which fails precisely when the majority
  ## vote is wrong -- the violated regime. Flag rather than silently relabel.
  list(prevalence = p, eps_0 = e0, eps_1 = e1, eps_sum = e0 + e1,
       prob = z, iterations = it,
       label_switch_risk = (e0 + e1) > 1)
}

## B2 -- prediction-powered inference for the population rate only.
## Rectifier estimator: mean of the model predictions over the whole sample,
## corrected by the observed bias on the labelled subset.
ppi_prevalence <- function(dat, anchor_idx, alpha = 0.05) {
  yhat <- as.integer(dat$k / dat$N > 0.5)
  L <- anchor_idx; U <- setdiff(seq_len(nrow(dat)), L)
  bias <- mean(yhat[L] - dat$gold[L])
  est  <- mean(yhat) - bias
  se   <- sqrt(var(yhat[U]) / length(U) +
               var(yhat[L] - dat$gold[L]) / length(L))
  z <- qnorm(1 - alpha / 2)
  list(prevalence = est, lo = est - z * se, hi = est + z * se, se = se)
}
