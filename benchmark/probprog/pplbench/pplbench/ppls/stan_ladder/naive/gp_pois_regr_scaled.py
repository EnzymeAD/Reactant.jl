from .._gp_ladder_base import GPLadderBase


class GPPoisRegrScaled(GPLadderBase):
    """S0: direct translation, everything in ``model``.

    Expression-for-expression the same program as the SlicStan input
    (evidence/related_work/gp_pois_regr_naive.slic), written the way Stan
    requires it: a single ``model`` block with local variables. Same
    arithmetic as the SlicStan-translated variant, so with a shared seed the
    two produce identical chains and their wall-time ratio is a pure
    per-gradient-cost ratio.
    """

    def get_code(self) -> str:
        return """
// GP Poisson Regression (scaled, NCP) -- naive placement: kernel build,
// Cholesky, and latent map all inside the model block.
data {
  int N;
  real rho;
  array[N] real x;
  array[N] int y;
}
parameters {
  real<lower=0> alpha;
  vector[N] f_tilde;
}
model {
  matrix[N, N] K;
  for (i in 1:N) {
    for (j in 1:N) {
      K[i, j] = exp((0 - 0.5) * pow(((x[i] - x[j]) / rho), 2));
    }
  }
  matrix[N, N] cov = alpha * alpha * K;
  matrix[N, N] L = cholesky_decompose(cov);
  vector[N] f = L * f_tilde;

  alpha ~ normal(0, 2);
  for (i in 1:N) {
    f_tilde[i] ~ normal(0, 1);
  }
  for (i in 1:N) {
    y[i] ~ poisson_log(f[i]);
  }
}
"""
