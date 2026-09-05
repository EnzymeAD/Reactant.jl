from .._gp_ladder_base import GPLadderBase


class GPPoisRegrScaled(GPLadderBase):
    """S1': SlicStan's placement with cov/L/f as ``model`` locals.

    Identical to the ``slicstan`` variant except that the three
    parameter-level statements SlicStan put in ``transformed parameters`` are
    declared as local variables of the ``model`` block. This changes nothing
    about *where* the work is done per gradient evaluation; it only stops
    CmdStan from writing the N x N matrices L and cov to the output CSV on
    every draw, an I/O cost that SlicStan's placement incurs and the naive
    program does not. Reported alongside ``slicstan`` so the I/O share of the
    difference is visible.
    """

    def get_code(self) -> str:
        return """
// SlicStan placement (kernel in transformed data), parameter-level
// statements as model locals.
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
model {
  matrix[N,N] cov = alpha * alpha * K;
  matrix[N,N] L = cholesky_decompose(cov);
  vector[N] f = L * f_tilde;
  alpha ~ normal(0, 2);
  for(i in 1 : N){
    f_tilde[i] ~ normal(0, 1);
  }
  for(i in 1 : N){
    y[i] ~ poisson_log(f[i]);
  }
}
"""
