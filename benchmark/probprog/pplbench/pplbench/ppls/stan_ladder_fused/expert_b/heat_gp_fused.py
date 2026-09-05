from .._fused_ladder_base import FusedLadderBase


class HeatGPFused(FusedLadderBase):
    """S3: expert A plus the CholScale identity applied by hand.

    chol(tau^2 K) = tau * chol(K) for tau >= 0, so the GP factorization depends
    only on data and moves to transformed data; the per-gradient GP cost
    becomes one O(D^2) matrix-vector product. One dynamics Cholesky per gradient
    remains. This is what SICM's CholeskyScaleFactorization derives.
    """

    def get_code(self) -> str:
        return """
// FUSED LADDER S3 (expert B): expert A plus the CholScale identity by hand.
// chol(tau^2 K) = tau * chol(K) for tau >= 0, so chol(K) is data-only and moves
// to transformed data; the per-gradient GP cost is one O(D^2) matvec.
data {
  int D;
  int M_obs;
  int obs_step;
  int S;
  real rho;
  real dt;
  real sigma;
  array[D] real s;
  matrix[D,D] Lap;
  matrix[D,D] Idm;
  matrix[S,D] H;
  matrix[M_obs,S] y;
}
transformed data {
  matrix[D,D] K;
  matrix[D,D] L_K;
  for(i in 1 : D){
    for(j in 1 : D){
      K[i,j] = exp((0 - 1) * (sqrt((s[i] - s[j]) * (s[i] - s[j])) / rho));
    }
  }
  L_K = cholesky_decompose(K);
}
parameters {
  vector[D] u_tilde;
  real z_kappa;
  real z_tau;
}
model {
  matrix[D,D] M;
  matrix[D,D] LM;
  vector[D] u;
  vector[D] zf;
  vector[S] mobs;
  real tau;
  real kappa;
  tau = exp(z_tau);
  kappa = exp(z_kappa);
  u = tau * (L_K * u_tilde);
  z_tau ~ normal(0, 1);
  z_kappa ~ normal(0, 1);
  for(i in 1 : D){
    u_tilde[i] ~ normal(0, 1);
  }
  M = (Idm + dt * kappa * Lap);
  LM = cholesky_decompose(M);
  for(jw in 1 : M_obs){
    for(t in 1 : obs_step){
      zf = mdivide_left_tri_low(LM, u);
      u = mdivide_right_tri_low(zf', LM)';
    }
    mobs = H * u;
    for(jj in 1 : S){
      y[jw,jj] ~ normal(mobs[jj], sigma);
    }
  }
}
"""
