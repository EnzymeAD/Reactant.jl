
using Pkg;
Pkg.activate(@__DIR__);

@static if VERSION ≥ v"1.10-" && VERSION < v"1.11"
    Pkg.add([
        PackageSpec(; name="Reactant", path=joinpath(@__DIR__, "../../..")),
        PackageSpec(; name="ComradeBase", rev="ptiede-reactantex"),
        PackageSpec(; name="Comrade", rev="ptiede-reactant"),
        PackageSpec(; name="VLBISkyModels", rev="ptiede-copyto"),
        PackageSpec(; name="VLBILikelihoods", rev="ptiede-reactant"),
        PackageSpec(; name="VLBIImagePriors", rev="ptiede-reactantperf"),
        PackageSpec(;
            url="https://github.com/ptiede/TransformVariables.jl", rev="ptiede-reactant"
        ),
    ])
end

Pkg.instantiate()
Pkg.precompile()

using Reactant
using NFFT
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
include("reactant_nfft.jl")

using Downloads
using Distributions

using Pyehtim
using Test

const dataurl = "https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/EHTC_M87pol2017_Nov2023/hops_data/April06/SR2_M87_2017_096_lo_hops_ALMArot.uvfits"
const dataf = Base.download(dataurl)

# TODO Make ReactantLogExpFunctionsExt.
function LogExpFunctions.logistic(@nospecialize x::Reactant.TracedRNumber)
    Reactant.@opcall logistic(x)
end
LogExpFunctions.log1pexp(x::Reactant.TracedRNumber) = log(1 + exp(x))

#!!!! TODO Everything in this block needs to be upstreamed to TransformVariables.jl
# The major problem is that most of them require @allowscalar which is not very nice enforce in their
# codebase since it is expensive for CPU code. The other temporary solution is to make a ReactantTransformVariablesExt.jl package.
function TV.transform_with(
    flag::TV.NoLogJac, t::TV.ScalarTransform, x::Reactant.AnyTracedRVector, index
)
    return transform(t, @allowscalar x[index]), flag, index + 1
end

# TODO Upstream to TransformVariables.jl
function TV.transform_with(
    ::TV.LogJac, t::TV.ScalarTransform, x::Reactant.AnyTracedRVector, index
)
    return TV.transform_and_logjac(t, @allowscalar x[index])..., index + 1
end

# TODO This is needed for TransformVariables but @allowscalar is rather annoying here.
function TV._transform_tuple(flag::TV.LogJacFlag, x::Reactant.AnyTracedRVector, index, ts)
    tfirst = first(ts)
    out = TV.transform_with(flag, tfirst, x, index)
    @allowscalar yfirst = out[1]
    @allowscalar ℓfirst = out[2]
    @allowscalar index′ = out[3]
    # yrest, ℓrest, index′′
    trest = Base.tail(ts)
    outrest = TV._transform_tuple(flag, x, index′, trest)
    @allowscalar yrest = outrest[1]
    @allowscalar ℓrest = outrest[2]
    @allowscalar index′′ = outrest[3]
    return (yfirst, yrest...), ℓfirst + ℓrest, index′′
end

# TODO Upstream to TransformVariables.jl
function TV._transform_tuple(
    flag::TV.LogJacFlag, x::Reactant.AnyTracedRVector, index, ::Tuple{}
)
    return (), TV.logjac_zero(flag, eltype(x)), index
end

# TODO Upstream to TransformVariables.jl
TV.logjac_zero(::TV.LogJac, ::Type{T}) where {T<:Reactant.RNumber} = log(one(T))

# TODO Upstream to TransformVariables.jl (essentially identical just need to loosen types)
function TV.transform_with(
    flag::TV.LogJacFlag,
    t::TV.ArrayTransformation{TV.Identity},
    x::Reactant.AnyTracedRVector,
    index,
)
    (; dims) = t
    index′ = index + dimension(t)
    y = reshape(x[index:(index′ - 1)], t.dims)
    return y, TV.logjac_zero(flag, eltype(x)), index′
end

#!!!! End of TransformVariables.jl upstream block

# TODO Upstream to VLBIImagePriors (allowscalar needed for Reactant)
function TV.transform_with(
    flag::TV.LogJacFlag, ::AngleTransform, y::Reactant.AnyTracedRVector, index
)
    T = eltype(y)
    ℓi = TV.logjac_zero(flag, T)
    x1 = @allowscalar y[index]
    x2 = @allowscalar y[index + 1]
    r = sqrt(x1^2 + x2^2)
    # Use log-normal with μ = 0, σ = 1/4
    σ = oftype(r, 1 / 4)
    if !(flag isa TV.NoLogJac)
        lr = log(r)
        ℓi = -lr^2 * inv(2 * σ^2) - lr
    end

    return atan(x1, x2), ℓi, index + 2
end

# TODO to upstream to VLBIImagePriors
function TV.transform_with(
    flag::TV.LogJacFlag,
    t::TV.ArrayTransformation{<:AngleTransform},
    y::Reactant.AnyTracedRVector,
    index,
)
    (; inner_transformation, dims) = t
    T = eltype(y)
    ℓ = TV.logjac_zero(flag, T)
    out = similar(y, dims)
    @trace for i in eachindex(out)
        θ, ℓi, index2 = TV.transform_with(flag, inner_transformation, y, index)
        index = index2
        ℓ += ℓi
        @allowscalar out[i] = θ
    end
    return out, ℓ, index
end

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
    (; srf, grid) = metadata
    x = genfield(StationaryRandomField(MarkovPS(ρs), srf), z)
    x .*= σ
    rast = exp.(x .- maximum(x))
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
skymeta = (; srf=pl, grid=grd)

ρs = ntuple(Returns(Uniform(0.01, max(size(grd)...))), 3)
zprior = std_dist(pl)
prior = (z=zprior, ρs=ρs, σ=Exponential(1.0))

skymr = SkyModel(sky, prior, grd; metadata=skymeta, algorithm=ReactantAlg()) # Need to do this so that we allocate proper Reactant arrays for internal stuff
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
# end
