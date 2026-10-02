from .._phylo_ladder_base import PhyloLadderBase


class PhyloPois(PhyloLadderBase):
    """S1: SlicStan's translation of the naive program, placement verbatim.

    Source: evidence/real_data_rows_2026-09-29/slicstan_phylo/
    phylo_pois_slicstan_raw.stan (SlicStan commit bbe1208,
    ``--from-file phylo_pois_naive.slic``). Edits, none of which moves a
    statement between blocks:
      * ``real[N] x`` / ``int[N] y``  ->  ``array[N] real x`` / ``array[N] int y``
        (SlicStan emits the pre-2.33 array syntax)
      * data and parameter declarations put back in source order (SlicStan
        sorts them by name, which declares ``A`` before ``N`` and would not
        compile, and would reorder the unconstrained parameter vector)
      * ``real sigma_sq`` / ``real sigma_obs``  ->  ``real<lower=0> ...``
        (SlicStan's grammar has no bound constraints)
      * ``sigma_sq ~ gamma(1, 1)``  ->  ``sigma_sq ~ exponential(1)``, the same
        density (SlicStan's distribution table has no exponential, so the
        input states it as gamma(1, 1))
    SlicStan places nothing in ``transformed data`` (no statement is
    data-only) and cov, L and f in ``transformed parameters``.
    """

    def get_code(self) -> str:
        return """
// SlicStan output (placement verbatim; see class docstring for the edits).
data {
  int N;
  array[N] real x;
  array[N] int y;
  matrix[N,N] A;

}
parameters {
  real<lower=0> sigma_sq;
  vector[N] eta;
  real b0;
  real b1;
  real<lower=0> sigma_obs;
  vector[N] eps;

}
transformed parameters {
  matrix[N,N] L;
  matrix[N,N] cov;
  vector[N] f;
  cov = sigma_sq * A;
  L = cholesky_decompose(cov);
  f = L * eta;

}
model {
  sigma_sq ~ exponential(1);
  for(i in 1 : N){
    eta[i] ~ normal(0, 1);

  }
  b0 ~ normal(0, 1);
  b1 ~ normal(0, 1);
  sigma_obs ~ normal(0, 1);
  for(i in 1 : N){
    eps[i] ~ normal(0, 1);

  }
  for(i in 1 : N){
    y[i] ~ poisson_log((((b0 + b1 * x[i]) + f[i]) + sigma_obs * eps[i]));

  }

}
"""
