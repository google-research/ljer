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

// M2: base homogeneous model, BINOMIAL likelihood, SUM identifying restriction.
//
// Differs from the published latent_model_restricted.stan in one respect: that
// model declares
//     real<lower=0, upper=0.5> epsilon_0;
// which is the BOX constraint. The pilot measured eps_0 = 0.789 for gemma-3-1b
// with eps_0 + eps_1 = 0.909, i.e. the box constraint is violated while the sum
// condition still holds -- so the box version is misspecified on real data.
//
// The identifying restriction from Theorem 1 is eps_0 + eps_1 < 1. We impose it
// exactly, without rejection sampling, by reparameterising:
//     v = eps_0 + eps_1  in (0,1)   the SUM
//     u = eps_0 / v      in (0,1)   the SPLIT
//     eps_0 = u*v,  eps_1 = (1-u)*v
// Both are unconstrained in the transformed space, so HMC never sees a reject,
// and eps_0 + eps_1 < 1 holds by construction.

data {
  int<lower=1> C;                      // number of items
  array[C] int<lower=0> k;             // "Yes" verdicts per item
  array[C] int<lower=1> N;             // parsed rounds per item
  // priors
  real mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> a_v;                   // Beta prior on the sum
  real<lower=0> b_v;
}

parameters {
  real theta;                          // logit of the prevalence P(D_c = 1)
  // bounded away from 0/1 for the same numerical reason as m3, so the two
  // models are compared on the same parameter space
  real<lower=1e-3, upper=1-1e-3> v;   // eps_0 + eps_1
  real<lower=1e-3, upper=1-1e-3> u;   // share of the sum carried by eps_0
}

transformed parameters {
  real<lower=0, upper=1> eps_0 = u * v;        // P(Yes | D_c = 0), false positive
  real<lower=0, upper=1> eps_1 = (1 - u) * v;  // P(No  | D_c = 1), false negative
}

model {
  theta ~ normal(mu_theta, sigma_theta);
  v ~ beta(a_v, b_v);
  u ~ beta(1, 1);

  real p = inv_logit(theta);
  for (c in 1:C) {
    target += log_mix(p,
                      binomial_lpmf(k[c] | N[c], 1 - eps_1),   // D_c = 1
                      binomial_lpmf(k[c] | N[c], eps_0));      // D_c = 0
  }
}

generated quantities {
  real prevalence = inv_logit(theta);
  real eps_sum = v;
  vector[C] prob_correct;              // posterior P(D_c = 1 | k_c)
  vector[C] log_lik;                   // for PSIS-LOO
  {
    real p = inv_logit(theta);
    for (c in 1:C) {
      real l1 = log(p)     + binomial_lpmf(k[c] | N[c], 1 - eps_1);
      real l0 = log1m(p)   + binomial_lpmf(k[c] | N[c], eps_0);
      log_lik[c] = log_sum_exp(l1, l0);
      prob_correct[c] = exp(l1 - log_lik[c]);
    }
  }
}
