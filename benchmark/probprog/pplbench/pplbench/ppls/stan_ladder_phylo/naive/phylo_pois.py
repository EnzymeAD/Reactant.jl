from .._phylo_ladder_base import PhyloLadderBase


class PhyloPois(PhyloLadderBase):
    """S0: direct translation, everything in ``model``.

    Statement for statement the SlicStan input
    (evidence/real_data_rows_2026-09-29/slicstan_phylo/phylo_pois_naive.slic),
    written as one ``model`` block with local variables. Parameters are declared
    in the source order, which the SlicStan variant keeps, so with a shared seed
    the two programs start from the same point and their wall-time ratio is a
    per-gradient-cost ratio.
    """

    def get_code(self) -> str:
        return """
// Phylogenetic Poisson regression -- naive placement: cov = sigma_sq * A,
// its Cholesky factor, and the latent map all inside the model block.
data {
  int N;
  array[N] real x;
  array[N] int y;
  matrix[N, N] A;
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
  matrix[N, N] cov = sigma_sq * A;
  matrix[N, N] L = cholesky_decompose(cov);
  vector[N] f = L * eta;

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
