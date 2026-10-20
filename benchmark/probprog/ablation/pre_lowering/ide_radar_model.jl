#=
IDE radar row: model and data (shared by ide_radar.jl and fig9a_paired_ide_radar.jl)
====================================================================================

Integro-difference equation (IDE) model of

  C. K. Wikle, A. Zammit-Mangion and N. Cressie. Spatio-Temporal Statistics with R.
  Chapman & Hall/CRC, 2019. Lab 5.2 (spatio-temporal IDE), R package IDE
  (A. Zammit-Mangion, CRAN; github.com/andrewzm/IDE), on the Sydney radar data
  (STRbook::radar_STIDF, 12 reflectivity images of 1120 pixels at 10-minute intervals).

The model is the Lab's first fit, `IDE(f = z ~ 1, data = radar_STIDF, dt = 10 min,
grid_size = 41, forecast = 2, hindcast = 2)` with the default spatially invariant kernel and the
default process basis (two-resolution bisquare, r = 129). The fixed maps (basis on the
quadrature grid, Gram matrix, basis at the pixels, the data) are exported from the saved fit
STRbook::IDE_Radar_results by ide_radar_data/export_ide_radar.R.

    k(s, x; theta) = k1 exp(-|x - s - (k3, k4)|^2 / k2)          on the 41 x 41 grid
    M(theta) = GRAM^-1 PHI_g' K(theta) PHI_g area^2               (129 x 129 propagator)
    alpha_0 ~ N(0, s2eta I),   alpha_t = M(theta) alpha_{t-1} + eta_t,   eta_t ~ N(0, s2eta I)
    z_t = beta + PHI alpha_t + eps_t,   eps_t ~ N(0, s2eps I)      (data at t = 2..13)

Spelling. The IDE package evaluates this marginal likelihood through one sparse precision
matrix over all time points and fits it by maximum likelihood (DEoptim). Here the same marginal
likelihood is computed by a Kalman filter over the 14 modelled time points that carry
likelihood (2 hindcast steps without data, then the 12 images; the 2 forecast steps add
nothing), the standard state-space form of a dynamic spatio-temporal model (Chapter 5 of the
book). The update is in information form, so the 1120 pixels of an image enter through PHI'PHI
and PHI'z_t. The propagator M(theta) is written where a state-space program builds its
transition, inside the filter's time step, as the operator of the IDE process model is
applied at every step. It depends on the sampled kernel parameters only, so it is
loop-invariant, and its adjoint has to be taken at every step unless it is hoisted before AD.
Agreement with the package: the filter reproduces the package's negative log likelihood at the
DEoptim optimum (30145.7228957, ide_radar_data/ide_radar.json, `deoptim_bestval`).

Bayesian additions (the source fits by maximum likelihood with beta profiled by GLS):
position (1 x 7) = (log k1, log k2, k3, k4, log s2eps, log s2eta, beta). Each of the first six
gets a normal prior whose central 95% interval is the package's DEoptim search box
(`plausible_ranges`); beta ~ N(0, 10^2).

IDE_GRID selects the quadrature grid: 41 (the package default and the Lab's choice) or 21
(`IDE(grid_size = 21)`: every other point of the 41-grid, with the package's
area = prod(range / 21)).
=#
using JSON3
using LinearAlgebra

const IDE_DATA_PATH = get(ENV, "IDE_DATA",
    joinpath(@__DIR__, "ide_radar_data", "ide_radar.json"))
const IDE_GRID = parse(Int, get(ENV, "IDE_GRID", "41"))
const IDE_RAW = JSON3.read(read(IDE_DATA_PATH, String))
_ide_mat(rows) = permutedims(reduce(hcat, [collect(Float64, r) for r in rows]))

const R = Int(IDE_RAW["r"])                                    # 129 basis functions
const G41 = Int(IDE_RAW["grid_size"])
@assert IDE_GRID in (G41, (G41 - 1) ÷ 2 + 1) "IDE_GRID must be $G41 or $((G41 - 1) ÷ 2 + 1)"
const GRID_IDX = let st = (G41 - 1) ÷ (IDE_GRID - 1)
    # expand.grid order: the first coordinate varies fastest
    [i2 + G41 * i1 + 1 for i1 in 0:st:(G41 - 1) for i2 in 0:st:(G41 - 1)]
end
const NG = length(GRID_IDX)
const SGRID = _ide_mat(IDE_RAW["s_grid"])[GRID_IDX, :]        # NG x 2 (columns s2, s1)
const PHIG = _ide_mat(IDE_RAW["PHI_grid"])[GRID_IDX, :]       # NG x R
const AREA = Float64(IDE_RAW["area"]) * (G41 / IDE_GRID)^2
const GINV = inv(PHIG' * PHIG .* AREA)                         # GRAM^-1
const PHI1 = _ide_mat(IDE_RAW["PHI_obs_block"])                # 1120 x R (same every image)
const NOBS = size(PHI1, 1)
const ZD = _ide_mat(IDE_RAW["Z"])                              # 1120 x 12
const DTI = collect(Int, IDE_RAW["data_time_index"])            # 0-based 2..13
const NT = maximum(DTI) + 1                                    # 14 filter steps
const LOG2PI = log(2 * pi)

# Data-only filter inputs. Step t (0-based) has PZ[t+1, :] = PHI'z_t and
# ZSTAT[t+1, :] = (z_t'z_t, sum z_t, w_t), w_t = 1 for an image and 0 for a hindcast step.
const AINF = PHI1' * PHI1                                      # R x R information matrix
const S1 = reshape(vec(sum(PHI1; dims=1)), R, 1)               # PHI'1
const PZ, ZSTAT = let pz = zeros(NT, R), st = zeros(NT, 3)
    for (j, t) in enumerate(DTI)
        z = ZD[:, j]
        pz[t + 1, :] = PHI1' * z
        st[t + 1, :] = [sum(z .* z), sum(z), 1.0]
    end
    pz, st
end

# Priors: central 95% interval = the package's DEoptim box.
const PRIOR_MU, PRIOR_SD = let P = IDE_RAW["plausible_ranges"]
    lo(k) = Float64(P[1][k]); hi(k) = Float64(P[2][k])
    box(a, b) = (0.5 * (a + b), 0.25 * (b - a))
    bx = [box(log(lo("k1")), log(hi("k1"))), box(log(lo("k2")), log(hi("k2"))),
          box(lo("k3"), hi("k3")), box(lo("k4"), hi("k4")),
          box(log(lo("sigma2_eps")), log(hi("sigma2_eps"))),
          box(log(lo("sigma2_eta")), log(hi("sigma2_eta"))), (0.0, 10.0)]
    ([b[1] for b in bx], [b[2] for b in bx])
end
const NP = 7
const LOGPRIOR_CONST = -sum(log.(PRIOR_SD)) - 0.5 * NP * LOG2PI

# Grid coordinates as a column and a row, so that (C .+ shift) .- CT is the NG x NG
# displacement matrix of construct_kernel (rows s, columns x).
const C1 = reshape(SGRID[:, 1], NG, 1)
const C1T = reshape(SGRID[:, 1], 1, NG)
const C2 = reshape(SGRID[:, 2], NG, 1)
const C2T = reshape(SGRID[:, 2], 1, NG)
const PHIGT = permutedims(PHIG)
const I_R = Matrix{Float64}(I, R, R)
const TRIL = [i >= j ? 1.0 : 0.0 for i in 1:R, j in 1:R]   # lower-triangle mask
const M0 = zeros(R, 1)
const P0 = zeros(R, R)
const LL0 = zeros(1, 1)

# ── Model: log joint density ─────────────────────────────────────────────────
function ide_logpdf(position, C1, C1T, C2, C2T, PHIG, PHIGT, GINV, AINF, S1, PZ, ZSTAT,
                    I_R, TRIL, M0, P0, LL0)
    lk1 = @allowscalar position[1]
    lk2 = @allowscalar position[2]
    k3 = @allowscalar position[3]
    k4 = @allowscalar position[4]
    ls2eps = @allowscalar position[5]
    ls2eta = @allowscalar position[6]
    beta = @allowscalar position[7]
    k1 = exp(lk1)
    k2 = exp(lk2)
    s2eps = exp(ls2eps)
    s2eta = exp(ls2eta)

    m = M0 .+ 0.0                                   # R x 1 filter mean
    P = P0 .+ 0.0                                   # R x R filter covariance
    ll = LL0 .+ 0.0                                 # 1 x 1 log-likelihood accumulator
    @trace track_numbers = false for t in 0:(NT - 1)
        # Propagator of the IDE process model: kernel on the quadrature grid, projected on
        # the process basis. Depends on (k1..k4) only: loop-invariant.
        D0 = (C1 .+ k3) .- C1T                      # NG x NG
        D1 = (C2 .+ k4) .- C2T
        K = k1 .* exp.(-(D0 .* D0 .+ D1 .* D1) ./ k2)
        Mp = (GINV * (PHIGT * (K * PHIG))) .* (AREA * AREA)
        # Predict.
        m = Mp * m
        P = Mp * P * transpose(Mp) .+ s2eta .* I_R
        # Update with image t (information form; w = 0 at the hindcast steps).
        pz = reshape(PZ[t + 1, :], R, 1)
        st = reshape(ZSTAT[t + 1, :], 1, 3)
        zz = st[:, 1:1]
        sz = st[:, 2:2]
        w = st[:, 3:3]
        LP = Reactant.Ops.cholesky(P; lower=true) .* TRIL       # P = LP LP'
        Aw = w .* AINF
        Gm = I_R .+ (transpose(LP) * Aw * LP) ./ s2eps
        LG = Reactant.Ops.cholesky(Gm; lower=true)
        LGo = LinearAlgebra.LowerTriangular(LG)
        bz = w .* (pz .- beta .* S1)                # PHI'(z_t - beta)
        g = bz .- Aw * m                            # PHI'(z_t - beta - PHI m)
        ee = w .* (zz .- (2 .* beta) .* sz .+ (beta * beta * NOBS)) .-
             2 .* (transpose(m) * bz) .+ transpose(m) * (Aw * m)
        u = transpose(LP) * g
        v = transpose(LGo) \ (LGo \ u)
        quad = (ee .- (transpose(u) * v) ./ s2eps) ./ s2eps
        logdetG = 2 .* sum(log.(LinearAlgebra.diag(LG)))
        ll = ll .- 0.5 .* (w .* (NOBS * (LOG2PI + ls2eps)) .+ logdetG .+ quad)
        m = m .+ (LP * v) ./ s2eps
        Wm = LGo \ transpose(LP)
        P = transpose(Wm) * Wm
    end

    zp1 = (lk1 - PRIOR_MU[1]) / PRIOR_SD[1]
    zp2 = (lk2 - PRIOR_MU[2]) / PRIOR_SD[2]
    zp3 = (k3 - PRIOR_MU[3]) / PRIOR_SD[3]
    zp4 = (k4 - PRIOR_MU[4]) / PRIOR_SD[4]
    zp5 = (ls2eps - PRIOR_MU[5]) / PRIOR_SD[5]
    zp6 = (ls2eta - PRIOR_MU[6]) / PRIOR_SD[6]
    zp7 = (beta - PRIOR_MU[7]) / PRIOR_SD[7]
    logprior = -0.5 * (zp1 * zp1 + zp2 * zp2 + zp3 * zp3 + zp4 * zp4 + zp5 * zp5 +
                       zp6 * zp6 + zp7 * zp7) + LOGPRIOR_CONST
    return sum(ll) + logprior
end

const MARGS_HOST = (C1, C1T, C2, C2T, PHIG, PHIGT, GINV, AINF, S1, PZ, ZSTAT, I_R, TRIL, M0, P0, LL0)

"""Position of the IDE package's maximum-likelihood fit (DEoptim optimum, GLS beta)."""
function ide_fit_position()
    rp = IDE_RAW["reference_points"][1]
    k = collect(Float64, rp["k"])
    return reshape([log(k[1]), log(k[2]), k[3], k[4], log(Float64(rp["s2eps"])),
                    log(Float64(rp["s2eta"])), Float64(rp["beta"])], 1, NP)
end
