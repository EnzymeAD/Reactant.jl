from .._fused_ladder_base import FusedLadderBase


class HeatGPFused(FusedLadderBase):
    """S1: SlicStan's translation of the naive program, placement verbatim.

    Source: related_work/slic/heat_gp_fused_slicstan_raw.stan (SlicStan commit
    bbe1208). Fix-ups for stanc3 v2.38, none of which moves a statement:
      * real[D] s -> array[D] real s (SlicStan emits pre-2.33 syntax)
      * data declarations reordered (SlicStan emits them alphabetically, so
        matrix[S,D] H precedes int S)
      * variables SlicStan declares in transformed parameters but assigns
        in model are declared as model locals instead; stanc3 rejects
        the raw form with "Cannot assign to global variable ... declared in
        previous blocks". The transformed-parameters u is renamed u0.
      * the substitution loops are replaced by Stan's library triangular solve,
        UNIFORMLY across every rung of this ladder (SlicStan's grammar has no
        mdivide_*; the slicstan_subst rung measures that spelling's cost).
    SlicStan hoisted the kernel loop to transformed data (TIER 0) but left
    chol(tau^2 K) at parameter level (TIER 2 not split) and the whole time loop,
    Cholesky included, in model (TIER 1 not hoisted).
    """

    def get_code(self) -> str:
        return """
// FUSED LADDER S1 (SlicStan): placement exactly as emitted by SlicStan.
// Fix-ups: array syntax, data declarations reordered, and the model-assigned
// variables declared as model locals (SlicStan's raw output assigns
// transformed-parameters variables inside `model`, which stanc3 rejects).
// None of these moves a statement.
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
