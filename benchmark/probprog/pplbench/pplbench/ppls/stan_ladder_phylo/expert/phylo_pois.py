from .._phylo_ladder_base import PhyloLadderBase


class PhyloPois(PhyloLadderBase):
    """S2: brms's generated code for this model.

    brms (``phen_pois ~ cofactor + (1 | gr(phylo, cov = A)) + (1 | obs)``)
    computes ``Lcov <- t(chol(A))`` once in R (R/data-predictor.R,
    data_gr_global) and emits ``r_1_1 = (sd_1[1] * (Lcov_1 * z_1[1]))``
    (R/stan-predictor.R). Here the factor is computed in ``transformed data``
    from the same A every variant receives, and ``sd = sqrt(sigma_sq)`` because
    this model's parameter is the variance. chol(sigma_sq A) = sqrt(sigma_sq)
    chol(A) is exactly the rewrite SICM derives from the naive form. The effect
    is a model local, as in the GP ladder's expert program (brms places it in
    transformed parameters, which would add N values per draw to the CSV).
    """

    def get_code(self) -> str:
        return """
// Expert placement (brms): chol(A) computed once from data,
// f = sqrt(sigma_sq) * (Lcov * eta).
data {
  int N;
  array[N] real x;
  array[N] int y;
  matrix[N, N] A;
}
transformed data {
  matrix[N, N] Lcov = cholesky_decompose(A);
}
parameters {
  real<lower=0> sigma_sq;
  vector[N] eta;
  real b0;
  real b1;
  real<lower=0> sigma_obs;
  vector[N] eps;
}
model {
  vector[N] f = sqrt(sigma_sq) * (Lcov * eta);

  sigma_sq ~ exponential(1);
  for (i in 1:N) {
    eta[i] ~ normal(0, 1);
  }
  b0 ~ normal(0, 1);
  b1 ~ normal(0, 1);
  sigma_obs ~ normal(0, 1);
  for (i in 1:N) {
    eps[i] ~ normal(0, 1);
  }
  for (i in 1:N) {
    y[i] ~ poisson_log(b0 + b1 * x[i] + f[i] + sigma_obs * eps[i]);
  }
}
"""
