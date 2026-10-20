from .._base import PrimatesBrownianStanBase


class PrimatesBrownian(PrimatesBrownianStanBase):
    """The textbook program with the user guide's rewrite: the Cholesky factor
    of the fixed R in transformed data, and the likelihood on the scaled factor.
    Every other statement is unchanged."""

    def get_code(self) -> str:
        return """data{
     matrix[151,151] V;
     matrix[151,151] Imat;
     int N_spp;
     vector[151] B;
     vector[151] G;
     vector[151] M;
     matrix[151,151] R;
}
transformed data{
     matrix[N_spp,N_spp] L_R = cholesky_decompose( R );
}
parameters{
     real a;
     real bG;
     real bM;
     real<lower=0> sigma_sq;
}
model{
     vector[151] mu;
    sigma_sq ~ exponential( 1 );
    bM ~ normal( 0 , 0.5 );
    bG ~ normal( 0 , 0.5 );
    a ~ normal( 0 , 1 );
    for ( i in 1:151 ) {
        mu[i] = a + bM * M[i] + bG * G[i];
    }
    B ~ multi_normal_cholesky( mu , sqrt( sigma_sq ) * L_R );
}
"""
