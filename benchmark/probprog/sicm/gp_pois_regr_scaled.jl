# GP Poisson Regression — Scaled (reuses motivating model at larger N)
#
# Same model as motivating/gp_pois_regr.jl but paired with synthetic
# data generators that can produce datasets at N > 11.
# SICM: CholeskyScaleFactorization hoists cholesky(alpha^2 * K_base).

include(joinpath(@__DIR__, "../motivating/gp_pois_regr.jl"))
