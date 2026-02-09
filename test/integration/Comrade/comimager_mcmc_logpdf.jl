ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"
using Pkg;
Pkg.activate(@__DIR__);

@static if VERSION ≥ v"1.10-" && VERSION < v"1.11"
    Pkg.add([
        PackageSpec(; name="ComradeBase", rev="main"),
        PackageSpec(; name="Comrade", rev="ptiede-reactant"),
        PackageSpec(; name="VLBISkyModels", rev="ptiede-reactnfft"),
        PackageSpec(; name="VLBILikelihoods", rev="ptiede-reactant"),
        PackageSpec(; name="VLBIImagePriors", rev="ptiede-reactantperf"),
        PackageSpec(;
            url="https://github.com/ptiede/TransformVariables.jl", rev="ptiede-reactant"
        ),
        PackageSpec(; url="https://github.com/ptiede/NFFT.jl", rev="ptiede-reactant"),
    ])
    Pkg.develop(; path=joinpath(@__DIR__, "../../../"))
end

Pkg.instantiate()
Pkg.precompile()

using Reactant
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
using LinearAlgebra
using AbstractFFTs

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[] = false
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true

using VLBISkyModels
using VLBILikelihoods
using Comrade
using Distributions
using VLBIImagePriors
using LogExpFunctions
import TransformVariables as TV

using Downloads
using Distributions
using Enzyme

using Pyehtim
using Test

# --- Reactant extension overrides (same as comimager.jl) ---

function LogExpFunctions.logistic(@nospecialize x::Reactant.TracedRNumber)
    Reactant.@opcall logistic(x)
end
LogExpFunctions.log1pexp(x::Reactant.TracedRNumber) = log(1 + exp(x))

Distributions.logpdf(d::Uniform, x::Reactant.TracedRNumber) = oftype(x, -log(d.b - d.a))
function Distributions.logpdf(d::Exponential, x::Reactant.TracedRNumber)
    λ = rate(d)
    z = log(λ) - λ * x
    @trace if x < 0
        out = oftype(z, -Inf)
    else
        out = z
    end
    return z
end

function Distributions.logpdf(d::Distributions.DiagNormal, x::Reactant.AnyTracedRVector)
    l = VLBILikelihoods._unnormed_logpdf_μΣ(d.μ, d.Σ.diag, x)
    n = VLBILikelihoods._gaussnorm(d.μ, d.Σ.diag)
    return l + n
end

# --- Sky model (same as comimager.jl) ---

function sky(θ, metadata)
    (; z, ρs, σ) = θ
    (; srf, grid, mimg) = metadata
    x = genfield(StationaryRandomField(MarkovPS(ρs), srf), z)
    x .*= σ
    mx = maximum(x)
    bmimg = baseimage(mimg)
    rast = @. exp(x - mx) * bmimg
    rast ./= sum(rast)
    return ContinuousImage(rast, grid, DeltaPulse{3}())
end

# --- Data loading and posterior setup (same as comimager.jl) ---

const dataf = joinpath(
    @__DIR__, "../../../../deps/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"
)

obs = ehtim.obsdata.load_uvfits(dataf)
obsavg = scan_average(obs).add_fractional_noise(0.02)
dvis = extract_table(obsavg, Visibilities())

npix = 64
fovx = μas2rad(200.0)
fovy = μas2rad(200.0)

grd = imagepixels(fovx, fovy, npix, npix)
pl = StationaryRandomFieldPlan(grd)
mimg = intensitymap(modify(Gaussian(), Stretch(μas2rad(25.0))), grd)
skymeta = (; srf=pl, grid=grd, mimg=mimg)

ρs = ntuple(Returns(Uniform(0.01, max(size(grd)...))), 3)
zprior = std_dist(pl)
prior = (z=zprior, ρs=ρs, σ=Exponential(1.0))

skymr = SkyModel(sky, prior, grd; metadata=skymeta, algorithm=VLBISkyModels.ReactantAlg())

g(x) = exp(complex(x.lg, x.gp))
G = SingleStokesGain(g)

intpr = (
    lg=ArrayPrior(
        IIDSitePrior(ScanSeg(), Normal(0.0, 0.2));
        LM=IIDSitePrior(ScanSeg(), Normal(0.0, 1.0)),
    ),
    gp=ArrayPrior(
        IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^2)));
        refant=SEFDReference(0.0),
        phase=true,
    ),
)
intmodel = InstrumentModel(G, intpr)

# Non-Reactant version for prior_sample and reference logdensity
skym = SkyModel(sky, prior, grd; metadata=skymeta)
post = VLBIPosterior(skym, intmodel, dvis)
tpost = asflat(post)

# Reactant version for compiled logdensity
postr = VLBIPosterior(skymr, intmodel, dvis)
tpostr = asflat(postr)

# --- mcmc_logpdf integration ---

x = prior_sample(tpost)
pos_size = length(x)
println("Position size: $pos_size")

# Verify logdensity works before MCMC
xr = Reactant.to_rarray(x)
ld_ref = logdensityof(tpost, x)
ld_jit = @jit(logdensityof(tpostr, xr))
println("Reference logdensity: $ld_ref")
println("JIT logdensity:       $ld_jit")
@test ld_jit ≈ ld_ref

# The logpdf function for mcmc_logpdf.
# Calling convention: (position_vec, args...) where position is tensor<1×N×f64>.
function comrade_logpdf(x, tpostr)
    return logdensityof(tpostr, x[1, :])
end

function comrade_nuts_program(
    rng,
    tpostr,
    initial_position,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
)
    samples, diagnostics, rng = ProbProg.mcmc_logpdf(
        rng,
        comrade_logpdf,
        initial_position,
        tpostr;
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        max_tree_depth=10,
        num_warmup,
        num_samples,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
    return samples, diagnostics
end

seed = Reactant.to_rarray(UInt64[42, 0])
rng = ReactantRNG(seed)
initial_position = Reactant.to_rarray(reshape(x, 1, pos_size))
step_size = ConcreteRNumber(1e-3)
inverse_mass_matrix = ConcreteRArray(ones(Float64, pos_size))

num_warmup = 200
num_samples = 100

println("\nCompiling NUTS...")
compile_time_s = @elapsed begin
    compiled = @compile optimize = :probprog comrade_nuts_program(
        rng,
        tpostr,
        initial_position,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
    )
end
println("Compile time: $(round(compile_time_s, digits=1)) s")

println("\nRunning NUTS ($num_warmup warmup + $num_samples samples)...")
run_time_s = @elapsed begin
    samples, diagnostics = compiled(
        rng,
        tpostr,
        initial_position,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
    )
    samples_arr = Array(samples)
    diagnostics_arr = Array(diagnostics)
end
println("Run time: $(round(run_time_s, digits=1)) s")

println("\nSamples shape: $(size(samples_arr))")
println("Diagnostics: $diagnostics_arr")

@test size(samples_arr) == (num_samples, pos_size)

# Verify sampled positions have finite logdensity
for i in 1:num_samples
    ld = logdensityof(tpost, samples_arr[i, :])
    println("  Sample $i logdensity: $ld")
    @test isfinite(ld)
end

# Summary statistics
using MCMCDiagnosticTools  # triggers ReactantMCMCDiagnosticToolsExt for ESS/R-hat
summary = ProbProg.mcmc_summary(samples_arr)
display(summary)
