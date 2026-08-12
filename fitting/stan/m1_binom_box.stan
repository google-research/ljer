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

// M1: the published base model, reproduced for comparison ONLY.
//
// This mirrors latent_model_restricted.stan: error rates are constrained to
// (0, 0.5) -- the BOX constraint. It is included so the paper can quantify the
// bias that constraint induces, using gemma-3-1b where the pilot measured
// eps_0 = 0.789 with a sum of 0.909: the box constraint is violated, the sum
// condition is not. Fitting M1 there is fitting a misspecified model on purpose.
//
// Do not use M1 for headline results. Use M2.

data {
  int<lower=1> C;
  array[C] int<lower=0> k;
  array[C] int<lower=1> N;
  real mu_theta;
  real<lower=0> sigma_theta;
}

parameters {
  real theta;
  real<lower=0, upper=0.5> eps_0;      // BOX constraint -- the point of M1
  real<lower=0, upper=0.5> eps_1;
}

model {
  theta ~ normal(mu_theta, sigma_theta);
  eps_0 ~ beta(1, 1);                  // truncated to (0, 0.5) by the declaration
  eps_1 ~ beta(1, 1);

  real p = inv_logit(theta);
  for (c in 1:C) {
    target += log_mix(p,
                      binomial_lpmf(k[c] | N[c], 1 - eps_1),
                      binomial_lpmf(k[c] | N[c], eps_0));
  }
}

generated quantities {
  real prevalence = inv_logit(theta);
  real eps_sum = eps_0 + eps_1;
  vector[C] prob_correct;
  vector[C] log_lik;
  {
    real p = inv_logit(theta);
    for (c in 1:C) {
      real l1 = log(p)   + binomial_lpmf(k[c] | N[c], 1 - eps_1);
      real l0 = log1m(p) + binomial_lpmf(k[c] | N[c], eps_0);
      log_lik[c] = log_sum_exp(l1, l0);
      prob_correct[c] = exp(l1 - log_lik[c]);
    }
  }
}
