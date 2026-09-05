from .._gp_ladder_base import GPLadderBase


class GPPoisRegrScaled(GPLadderBase):
    """S2: the CholScale rewrite applied by hand.

    chol(alpha^2 K) = alpha * chol(K) for alpha >= 0, so the factorization
    depends only on data and moves to ``transformed data``; the per-gradient
    path keeps one O(N^2) matrix-vector product. This is what SICM derives
    automatically from the naive form; it is the upper reference for the
    Stan side of the ladder.
    """

    def get_code(self) -> str:
        return """
// Expert placement: chol(K) factored out of alpha and hoisted to
// transformed data (the CholScale identity applied by hand).
data {
  int N;
  real rho;
  array[N] real x;
  array[N] int y;
}
transformed data {
  matrix[N,N] K;
  matrix[N,N] L_K;
  for(i in 1 : N){
    for(j in 1 : N){
      K[i,j] = exp((0 - 0.5) * pow(((x[i] - x[j]) / rho),2));
    }
  }
  L_K = cholesky_decompose(K);
}
parameters {
  real<lower=0> alpha;
  vector[N] f_tilde;
}
model {
  vector[N] f = alpha * (L_K * f_tilde);
  alpha ~ normal(0, 2);
  for(i in 1 : N){
    f_tilde[i] ~ normal(0, 1);
  }
  for(i in 1 : N){
    y[i] ~ poisson_log(f[i]);
  }
}
"""
