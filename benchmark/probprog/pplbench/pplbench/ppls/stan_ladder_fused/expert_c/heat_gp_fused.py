from .._fused_ladder_base import FusedLadderBase


class HeatGPFused(FusedLadderBase):
    """S4: expert B plus the EigenLift identity applied by hand.

    M(kappa) = 1*I + (dt*kappa)*Lap is a scalar shift of a FIXED operator, so one
    symmetric eigendecomposition of Lap in transformed data replaces the
    dynamics Cholesky outright: M^-1 b = V ((V' b) ./ (1 + dt*kappa*lambda)).
    Per gradient: zero factorizations, two O(D^2) products per time step, and
    both products are against a DATA matrix rather than a parameter-dependent
    triangular factor. This is what SICM's CholeskyEigenLift derives; it is the
    upper reference for the Stan side of the ladder.
    """

    def get_code(self) -> str:
        return """
// FUSED LADDER S4 (expert C): expert B plus the EigenLift identity by hand.
// M(kappa) = 1*I + (dt*kappa)*Lap is a scalar shift of a FIXED operator, so one
// symmetric eigendecomposition of Lap in transformed data replaces the dynamics
// Cholesky outright: M^-1 b = V ((V' b) ./ (1 + dt*kappa*lambda)).
// Per gradient: zero factorizations, and every per-step product is against a
// DATA matrix rather than a parameter-dependent triangular factor.
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
  matrix[D,D] Vv;
  vector[D] lam;
  for(i in 1 : D){
    for(j in 1 : D){
      K[i,j] = exp((0 - 1) * (sqrt((s[i] - s[j]) * (s[i] - s[j])) / rho));
    }
  }
  L_K = cholesky_decompose(K);
  Vv = eigenvectors_sym(Lap);
  lam = eigenvalues_sym(Lap);
}
parameters {
  vector[D] u_tilde;
  real z_kappa;
  real z_tau;
}
model {
  vector[D] dinv;
  vector[D] u;
  vector[S] mobs;
  real tau;
  real kappa;
  tau = exp(z_tau);
  kappa = exp(z_kappa);
  u = tau * (L_K * u_tilde);
  dinv = 1.0 ./ (1.0 + dt * kappa * lam);
  z_tau ~ normal(0, 1);
  z_kappa ~ normal(0, 1);
  for(i in 1 : D){
    u_tilde[i] ~ normal(0, 1);
  }
  for(jw in 1 : M_obs){
    for(t in 1 : obs_step){
      u = Vv * (dinv .* (Vv' * u));
    }
    mobs = H * u;
    for(jj in 1 : S){
      y[jw,jj] ~ normal(mobs[jj], sigma);
    }
  }
}
"""
