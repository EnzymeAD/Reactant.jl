from .._fused_ladder_base import FusedLadderBase


class HeatGPFused(FusedLadderBase):
    """S0: direct translation, everything in model.

    Expression-for-expression the same program as the SlicStan input
    (related_work/slic/heat_gp_fused_naive.slic), written the way Stan requires
    it: one model block with local variables. Per gradient: the O(D^2)
    kernel build, one O(D^3) GP Cholesky, and T O(D^3) dynamics Choleskys.
    """

    def get_code(self) -> str:
        return """
// FUSED LADDER S0 (naive): kernel build, GP Cholesky and the per-step dynamics
// Cholesky all inside the model block. The statistician's direct translation.
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
parameters {
  vector[D] u_tilde;
  real z_kappa;
  real z_tau;
}
model {
  matrix[D,D] K;
  matrix[D,D] Kt;
  matrix[D,D] LK;
  matrix[D,D] M;
  matrix[D,D] LM;
  vector[D] u;
  vector[D] zf;
  vector[S] mobs;
  real tau;
  real kappa;
  tau = exp(z_tau);
  kappa = exp(z_kappa);
  for(i in 1 : D){
    for(j in 1 : D){
      K[i,j] = exp((0 - 1) * (sqrt((s[i] - s[j]) * (s[i] - s[j])) / rho));
    }
  }
  Kt = tau * tau * K;
  LK = cholesky_decompose(Kt);
  u = LK * u_tilde;
  z_tau ~ normal(0, 1);
  z_kappa ~ normal(0, 1);
  for(i in 1 : D){
    u_tilde[i] ~ normal(0, 1);
  }
  for(jw in 1 : M_obs){
    for(t in 1 : obs_step){
      M = (Idm + dt * kappa * Lap);
      LM = cholesky_decompose(M);
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
