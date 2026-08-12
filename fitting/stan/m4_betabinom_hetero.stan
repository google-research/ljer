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

// M4: beta-binomial with error rates varying by the self-reported difficulty H_c.
//
// Two departures from the published latent_model_hetero.stan, both deliberate.
//
// (1) That model writes logit(2 * eps_jc) = alpha_j + gamma_j * H_c, which forces
//     eps_jc < 0.5 by construction -- the box constraint again, violated in the
//     pilot data. Here we instead let the SUM and the SPLIT depend on H_c:
//         logit(v_c) = a_v + b_v * H_c        the sum, in (0,1)
//         logit(u_c) = a_u + b_u * H_c        the split, in (0,1)
//         eps_0c = u_c * v_c,  eps_1c = (1 - u_c) * v_c
//     This keeps eps_0c + eps_1c < 1 for EVERY item -- the actual identifying
//     restriction -- while allowing either rate to exceed 0.5.
//
// (2) That model declares real<lower=0> beta_0, a hard non-negativity constraint
//     on the slope, so it cannot represent a null or negative relationship at all.
//     Reviewer 79bE asked precisely this. Slopes here are unconstrained with a
//     Normal prior, so "no relationship" is reachable and testable.
//
// Note the pilot (D7) found H_c absorbs only 0-35% of the overdispersion, so M4
// is expected to add little over M3. Reporting that honestly is the result.

data {
  int<lower=1> C;
  array[C] int<lower=0> k;
  array[C] int<lower=1> N;
  vector[C] H;                         // difficulty, standardised by the caller
  real mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> sigma_slope;           // prior sd on b_v, b_u
  real<lower=0> a_rho;
  real<lower=0> b_rho;
}

parameters {
  real theta;
  real a_v;                            // sum intercept   (logit scale)
  real b_v;                            // sum slope on H
  real a_u;                            // split intercept (logit scale)
  real b_u;                            // split slope on H
  real<lower=0.01, upper=1-0.01> rho;
}

transformed parameters {
  // inv_logit of an unbounded linear predictor can reach 0 or 1 numerically,
  // and eps*s would then underflow to exactly 0 (beta_binomial_lpmf rejects
  // that). Clamp to the same interval m2/m3 use so the ladder is comparable.
  vector<lower=1e-3, upper=1-1e-3>[C] v =
    fmin(fmax(inv_logit(a_v + b_v * H), 1e-3), 1 - 1e-3);
  vector<lower=1e-3, upper=1-1e-3>[C] u =
    fmin(fmax(inv_logit(a_u + b_u * H), 1e-3), 1 - 1e-3);
  vector<lower=0, upper=1>[C] eps_0 = u .* v;
  vector<lower=0, upper=1>[C] eps_1 = (1 - u) .* v;
  real<lower=0> s = (1 - rho) / rho;
}

model {
  theta ~ normal(mu_theta, sigma_theta);
  a_v ~ normal(0, 2);
  a_u ~ normal(0, 2);
  b_v ~ normal(0, sigma_slope);        // unconstrained: may be zero or negative
  b_u ~ normal(0, sigma_slope);
  rho ~ beta(a_rho, b_rho);

  real p = inv_logit(theta);
  for (c in 1:C) {
    target += log_mix(p,
                beta_binomial_lpmf(k[c] | N[c], (1 - eps_1[c]) * s, eps_1[c] * s),
                beta_binomial_lpmf(k[c] | N[c], eps_0[c] * s, (1 - eps_0[c]) * s));
  }
}

generated quantities {
  real prevalence = inv_logit(theta);
  real mean_eps_0 = mean(eps_0);
  real mean_eps_1 = mean(eps_1);
  real mean_eps_sum = mean(v);
  vector[C] prob_correct;
  vector[C] log_lik;
  {
    real p = inv_logit(theta);
    for (c in 1:C) {
      real l1 = log(p)   + beta_binomial_lpmf(k[c] | N[c], (1 - eps_1[c]) * s, eps_1[c] * s);
      real l0 = log1m(p) + beta_binomial_lpmf(k[c] | N[c], eps_0[c] * s, (1 - eps_0[c]) * s);
      log_lik[c] = log_sum_exp(l1, l0);
      prob_correct[c] = exp(l1 - log_lik[c]);
    }
  }
}
