from .._gp_ladder_base import GPLadderBase


class GPPoisRegrScaled(GPLadderBase):
    """S1: SlicStan's translation of the naive program, placement verbatim.

    Source: evidence/related_work/gp_pois_regr_slicstan_raw.stan (SlicStan
    commit bbe1208, ``--from-file gp_pois_regr_naive.slic``). Three
    syntax-only edits for stanc3 v2.38, none of which moves a statement:
      * ``real[N] x`` / ``int[N] y``  ->  ``array[N] real x`` / ``array[N] int y``
        (SlicStan emits the pre-2.33 array syntax)
      * ``real alpha``  ->  ``real<lower=0> alpha``  (SlicStan's grammar has no
        bound constraints; the half-normal prior needs the bound)
    SlicStan placed the kernel loop in ``transformed data`` and the
    parameter-level statements (cov, L, f) in ``transformed parameters``.
    """

    def get_code(self) -> str:
        return """
// SlicStan output (placement verbatim; see class docstring for the
// syntax-only edits).
data {
  int N;
  real rho;
  array[N] real x;
  array[N] int y;

}
transformed data {
  matrix[N,N] K;
  for(i in 1 : N){
    for(j in 1 : N){
      K[i,j] = exp((0 - 0.5) * pow(((x[i] - x[j]) / rho),2));

    }

  }

}
parameters {
  real<lower=0> alpha;
  vector[N] f_tilde;

}
transformed parameters {
  matrix[N,N] L;
  matrix[N,N] cov;
  vector[N] f;
  cov = alpha * alpha * K;
  L = cholesky_decompose(cov);
  f = L * f_tilde;

}
model {
  alpha ~ normal(0, 2);
  for(i in 1 : N){
    f_tilde[i] ~ normal(0, 1);

  }
  for(i in 1 : N){
    y[i] ~ poisson_log(f[i]);

  }

}
"""
