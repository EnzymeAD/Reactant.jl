from .._fused_ladder_base import FusedLadderBase


class HeatGPFused(FusedLadderBase):
    """S1s: SlicStan's placement AND SlicStan's spelling of the solve.

    Identical to slicstan except that the implicit-Euler solve is the
    hand-written forward/back substitution SlicStan's grammar forces (it has
    cholesky_decompose but no triangular solve, no inverse, no transpose).
    Reported so the cost of the uniform library-solve substitution used by every
    other rung is visible rather than assumed.
    """

    def get_code(self) -> str:
        return """
// FUSED LADDER S1s (SlicStan, substitution): same placement, with SlicStan's
// own hand-written forward/back substitution instead of Stan's library
// triangular solve (SlicStan's grammar has no mdivide_*).
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
  vector[D] w;
  vector[S] mobs;
  u = u0;
  z_tau ~ normal(0, 1);
  z_kappa ~ normal(0, 1);
  for(i in 1 : D){
    u_tilde[i] ~ normal(0, 1);
  }
  for(jw in 1 : M_obs){
    for(t in 1 : obs_step){
      M = (Idm + dt * kappa * Lap);
      LM = cholesky_decompose(M);
      for(i in 1 : D){
        zf[i] = u[i];
        for(j in 1 : D){
          if((j < i)){
            zf[i] = (zf[i] - LM[i,j] * zf[j]);
          }
        }
        zf[i] = (zf[i] / LM[i,i]);
      }
      for(kk in 1 : D){
        w[(D + (1 - kk))] = zf[(D + (1 - kk))];
        for(j in 1 : D){
          if((j > (D + (1 - kk)))){
            w[(D + (1 - kk))] = (w[(D + (1 - kk))] - LM[j,(D + (1 - kk))] * w[j]);
          }
        }
        w[(D + (1 - kk))] = (w[(D + (1 - kk))] / LM[(D + (1 - kk)),(D + (1 - kk))]);
      }
      u = w;
    }
    mobs = H * u;
    for(jj in 1 : S){
      y[jw,jj] ~ normal(mobs[jj], sigma);
    }
  }
}
"""
