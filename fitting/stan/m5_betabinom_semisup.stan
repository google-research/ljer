// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// M5: semi-supervised beta-binomial with ground-truth anchors.
//
// This is the variant for the VIOLATED regime, where eps_0c + eps_1c >= 1 and
// the unsupervised mixture is not identified from the data alone. The pilot
// found all seven judges exceed 1.0 on VerifyBench-Hard under the no-reference
// arm (1.057 to 1.317), so this is the model that has to work there.
//
// Accordingly the sum restriction is DROPPED: eps_0, eps_1 are free on (0,1).
// Label-switching symmetry is broken by the labelled subset instead, exactly as
// Theorem 3 states -- D_c is observed and fixed for those items, so it does not
// invert under the transformation T.
//
// The labelled contribution is the JOINT likelihood
//     P(k_c, D_c | Theta) = P(D_c | theta) * P(k_c | D_c, eps)
// matching latent_model_semisupervised.stan. Note the published Equation (2)
// omits the P(D_c | theta) factor; the code has it and the code is right.
//
// W is passed as DATA so the tempering weight can be varied for the sensitivity
// analysis both NeurIPS reviewers asked for. W = 1 recovers an ordinary Bayesian
// posterior; W > 1 defines a power posterior whose intervals are not calibrated
// in the strict sense. Default W = floor(C_U / C_L), matching latentLogit.R.

data {
  int<lower=0> C_U;                        // unlabelled items
  array[C_U] int<lower=0> k_U;
  array[C_U] int<lower=1> N_U;

  int<lower=1> C_L;                        // labelled anchors
  array[C_L] int<lower=0> k_L;
  array[C_L] int<lower=1> N_L;
  array[C_L] int<lower=0, upper=1> D_L;    // observed latent state

  real<lower=0> W;                         // tempering weight on the labelled block
  real mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> a_rho;
  real<lower=0> b_rho;
}

parameters {
  real theta;
  real<lower=1e-3, upper=1-1e-3> eps_0;  // sum UNRESTRICTED -- may exceed 1
  real<lower=1e-3, upper=1-1e-3> eps_1;
  real<lower=0.01, upper=1-0.01> rho;
}

transformed parameters {
  real<lower=0> s = (1 - rho) / rho;
}

model {
  theta ~ normal(mu_theta, sigma_theta);
  eps_0 ~ beta(1, 1);
  eps_1 ~ beta(1, 1);
  rho ~ beta(a_rho, b_rho);

  real p = inv_logit(theta);

  // Part A -- unlabelled: marginalised mixture
  for (c in 1:C_U) {
    target += log_mix(p,
                beta_binomial_lpmf(k_U[c] | N_U[c], (1 - eps_1) * s, eps_1 * s),
                beta_binomial_lpmf(k_U[c] | N_U[c], eps_0 * s, (1 - eps_0) * s));
  }

  // Part B -- labelled: exact joint, upweighted by W
  for (c in 1:C_L) {
    real lp = bernoulli_lpmf(D_L[c] | p);
    if (D_L[c] == 1)
      lp += beta_binomial_lpmf(k_L[c] | N_L[c], (1 - eps_1) * s, eps_1 * s);
    else
      lp += beta_binomial_lpmf(k_L[c] | N_L[c], eps_0 * s, (1 - eps_0) * s);
    target += W * lp;
  }
}

generated quantities {
  real prevalence = inv_logit(theta);
  real eps_sum = eps_0 + eps_1;            // NOT restricted to < 1
  vector[C_U] prob_correct;                // held-out items only
  vector[C_U] log_lik;
  {
    real p = inv_logit(theta);
    for (c in 1:C_U) {
      real l1 = log(p)   + beta_binomial_lpmf(k_U[c] | N_U[c], (1 - eps_1) * s, eps_1 * s);
      real l0 = log1m(p) + beta_binomial_lpmf(k_U[c] | N_U[c], eps_0 * s, (1 - eps_0) * s);
      log_lik[c] = log_sum_exp(l1, l0);
      prob_correct[c] = exp(l1 - log_lik[c]);
    }
  }
}
