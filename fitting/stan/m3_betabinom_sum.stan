// M3: as M2 but with a BETA-BINOMIAL likelihood. The paper's contribution.
//
// The pilot measured design effects of 5.5-9.4 in every model x arm cell, i.e.
// an intra-class correlation rho of 0.50-0.93 and N_eff of 1.06-1.81 out of ten
// draws. The binomial likelihood assumes rho = 0 and therefore reports credible
// intervals too narrow by roughly sqrt(DE) ~ 2.5-3 on population parameters.
//
// Parameterisation: k ~ BetaBin(N, eps*s, (1-eps)*s) with s = (1-rho)/rho.
// Then E[k]/N = eps exactly, and the ICC is rho = 1/(s+1).
//
// The prior goes on RHO, not on s. At the measured rho = 0.85, s ~ 0.18, so any
// prior placed on s directly would be badly scaled.
//
// One concentration parameter is shared across both mixture components. Two free
// concentrations plus a two-component mixture is where the latent-class /
// overdispersion confound (see FITTING_PLAN Step 0) bites hardest; relax only if
// the identifiability gate shows it is safe.

data {
  int<lower=1> C;
  array[C] int<lower=0> k;
  array[C] int<lower=1> N;
  real mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> a_v;
  real<lower=0> b_v;
  real<lower=0> a_rho;                 // Beta prior on the ICC
  real<lower=0> b_rho;
  int<lower=0, upper=1> fix_rho;       // 1 = hold rho at rho_fixed
  real<lower=0, upper=1> rho_fixed;    // e.g. the D6-implied value
}

parameters {
  real theta;
  // Bounds are away from 0 and 1 on purpose. s = (1-rho)/rho collapses as
  // rho -> 1, and eps -> 0 as u or v -> 0; their product then underflows to
  // exactly 0, which beta_binomial_lpmf rejects. These bounds keep
  // eps*s >= ~1e-8, comfortably positive, while excluding nothing the data
  // could plausibly want (the pilot measured rho in 0.50-0.93).
  real<lower=1e-3, upper=1-1e-3> v;
  real<lower=1e-3, upper=1-1e-3> u;
  array[fix_rho ? 0 : 1] real<lower=0.01, upper=1-0.01> rho_raw;
}

transformed parameters {
  real<lower=0, upper=1> eps_0 = u * v;
  real<lower=0, upper=1> eps_1 = (1 - u) * v;
  real<lower=0, upper=1> rho = fix_rho ? rho_fixed : rho_raw[1];
  real<lower=0> s = (1 - rho) / rho;   // beta-binomial concentration
}

model {
  theta ~ normal(mu_theta, sigma_theta);
  v ~ beta(a_v, b_v);
  u ~ beta(1, 1);
  if (!fix_rho) rho_raw[1] ~ beta(a_rho, b_rho);

  real p = inv_logit(theta);
  for (c in 1:C) {
    target += log_mix(p,
                beta_binomial_lpmf(k[c] | N[c], (1 - eps_1) * s, eps_1 * s),
                beta_binomial_lpmf(k[c] | N[c], eps_0 * s, (1 - eps_0) * s));
  }
}

generated quantities {
  real prevalence = inv_logit(theta);
  real eps_sum = v;
  real n_eff_ratio = 1.0 / (1 + (max(N) - 1) * rho);   // N_eff / N
  vector[C] prob_correct;
  vector[C] log_lik;
  {
    real p = inv_logit(theta);
    for (c in 1:C) {
      real l1 = log(p)   + beta_binomial_lpmf(k[c] | N[c], (1 - eps_1) * s, eps_1 * s);
      real l0 = log1m(p) + beta_binomial_lpmf(k[c] | N[c], eps_0 * s, (1 - eps_0) * s);
      log_lik[c] = log_sum_exp(l1, l0);
      prob_correct[c] = exp(l1 - log_lik[c]);
    }
  }
}
