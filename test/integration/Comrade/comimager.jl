
using Pkg;
Pkg.activate(@__DIR__);

# ENV["JULIA_DEBUG"] = "Reactant"

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
        PackageSpec(;
            url="https://github.com/ptiede/NFFT.jl", rev="ptiede-reactant"
        )
    ])
    Pkg.develop(; path=joinpath(@__DIR__, "../../../"))
end

Pkg.instantiate()
Pkg.precompile()

using Reactant
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
using LinearAlgebra
using AbstractFFTs

#TODO register custom packages
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

# const dataurl = "https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/EHTC_M87pol2017_Nov2023/hops_data/April06/SR2_M87_2017_096_lo_hops_ALMArot.uvfits"
# const dataf = Base.download(dataurl)
const dataf = joinpath(@__DIR__, "../../../../deps/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits")

# TODO upstream to VLBISkyModels

# TODO Make ReactantLogExpFunctionsExt.
function LogExpFunctions.logistic(@nospecialize x::Reactant.TracedRNumber)
    Reactant.@opcall logistic(x)
end
LogExpFunctions.log1pexp(x::Reactant.TracedRNumber) = log(1 + exp(x))

# TODO Make Distributions package that is compatible with Reactant
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

# THis is very much needed or else `@compile` hangs
function Distributions.logpdf(d::Distributions.DiagNormal, x::Reactant.AnyTracedRVector)
    l = VLBILikelihoods._unnormed_logpdf_μΣ(d.μ, d.Σ.diag, x)
    n = VLBILikelihoods._gaussnorm(d.μ, d.Σ.diag)
    return l + n
end

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

# @testset "Comrade Integration Imaging" begin
obs = ehtim.obsdata.load_uvfits(dataf)
obsavg = scan_average(obs).add_fractional_noise(0.02)
dvis = extract_table(obsavg, Visibilities())

npix = 64
fovx = μas2rad(200.0)
fovy = μas2rad(200.0)

# Now let's form our cache's. First, we have our usual image cache which is needed to numerically
# compute the visibilities.
grd = imagepixels(fovx, fovy, npix, npix)
pl = StationaryRandomFieldPlan(grd)
mimg = intensitymap(modify(Gaussian(), Stretch(μas2rad(25.0))), grd)
skymeta = (; srf=pl, grid=grd, mimg=mimg)

ρs = ntuple(Returns(Uniform(0.01, max(size(grd)...))), 3)
zprior = std_dist(pl)
prior = (z=zprior, ρs=ρs, σ=Exponential(1.0))

skymr = SkyModel(sky, prior, grd; metadata=skymeta, algorithm=VLBISkyModels.ReactantAlg()) # Need to do this so that we allocate proper Reactant arrays for internal stuff
skym = SkyModel(sky, prior, grd; metadata=skymeta)

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

postr = VLBIPosterior(skymr, intmodel, dvis)
post = VLBIPosterior(skym, intmodel, dvis)

tpost = asflat(post)
tpostr = asflat(postr)

x = prior_sample(tpost)
xr = Reactant.to_rarray(x)
@test @jit(logdensityof(tpostr, xr)) ≈ logdensityof(tpost, x)
l(tpostr, xr) = logdensityof(tpostr, xr)
lr = @compile sync = true l(tpostr, xr)
# @benchmark lr($tpostr, $xr)

logdensityofref(tpostr, xr) = logdensityof(tpostr[], xr)
gl(tpostr, xr) = last(Enzyme.gradient(Reverse, logdensityofref, Ref(tpostr), xr))
Reactant.@profile gl(tpostr, xr)
glr = @compile sync = true gl(tpostr, xr)


## If you want to profile the Claude NUFFT you can do this
plr = postr.skymodel.grid.plan_forward.plan
xr = Reactant.to_rarray(rand(ComplexF64, size(grd)))
nf(pl, x) = sum(abs2, pl*x)
Reactant.@profile nf(plr, xr)
@code_hlo nf(plr, xr)

nfr(plr, x) = nf(plr[], x)
dnf(pl, xr) = last(Enzyme.gradient(Reverse, nfr, Ref(pl), xr))
dnfr = @compile sync=true dnf(plr, xr)
Reactant.@profile dnf(plr, xr)
@code_hlo dnf(plr, xr)
# using FiniteDifferences
# fdm = central_fdm(5, 1)
# gfd,  = grad(fdm, tpost, x)
# @test glr(tpostr, xr) ≈ gfd
