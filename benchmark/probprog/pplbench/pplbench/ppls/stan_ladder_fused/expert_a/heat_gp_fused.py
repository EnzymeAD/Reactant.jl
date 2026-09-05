from .._fused_ladder_base import FusedLadderBase


class HeatGPFused(FusedLadderBase):
    """S2: the tier-1 hoist applied by hand.

    M(kappa) = I + dt*kappa*Lap and its Cholesky are loop-invariant in t, so an
    expert lifts both above the time loop: T Choleskys per gradient become 1.
    This is what standard LICM recovers on the Impulse side once the model is
    lowered, and what SlicStan's three-level placement structurally cannot
    express. The GP Cholesky is still recomputed every gradient.
    """

    def get_code(self) -> str:
        return """
// FUSED LADDER S2 (expert A): the tier-1 hoist by hand. M(kappa) and its
// Cholesky are invariant across BOTH the window loop and the step loop, so both
// move above them. The GP Cholesky is still recomputed every gradient.
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
  for(i in 1 : D){
    for(j in 1 : D){
      K[i,j] = exp((0 - 1) * (sqrt((s[i] - s[j]) * (s[i] - s[j])) / rho));
    }
  }
}
parameters {
  vector[D] u_tilde;
  real z_kappa;
  real z_tau;
}
transformed parameters {
  real tau;
  real kappa;
  matrix[D,D] Kt;
  matrix[D,D] LK;
  vector[D] u0;
  tau = exp(z_tau);
  kappa = exp(z_kappa);
  Kt = tau * tau * K;
  LK = cholesky_decompose(Kt);
  u0 = LK * u_tilde;
}
model {
  matrix[D,D] M;
  matrix[D,D] LM;
  vector[D] u;
  vector[D] zf;
  vector[S] mobs;
  u = u0;
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
