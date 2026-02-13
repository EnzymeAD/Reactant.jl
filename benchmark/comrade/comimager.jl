
# TODO Make Distributions package that is compatible with Reactant
Distributions.logpdf(d::Uniform, x::Reactant.TracedRNumber) = oftype(x, -log(d.b - d.a))

Distributions.minimum(::Exponential{T}) where {T<:AbstractFloat} = zero(T)
Distributions.maximum(::Exponential{T}) where {T<:AbstractFloat} = convert(T, Inf)

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

# SO DUMB but Distributions doesn't support RNG from Flat32 vonmises distributions
function Distributions._rand!(
    rng::Random.AbstractRNG, d::DiagonalVonMises, x::AbstractVector{<:Float32}
)
    dv = Distributions.product_distribution(
        Distributions.VonMises.(Float64.(d.μ), Float64.(d.κ))
    )
    x64 = rand(rng, dv)
    x .= Float32.(x64)
    return x
end

const MyDiagNormal{T} = MvNormal{T,Distributions.PDMats.PDiagMat{T,Vector{T}},Vector{T}}

# THis is very much needed or else `@compile` hangs
function Distributions.logpdf(d::MyDiagNormal, x::Reactant.AnyTracedRVector)
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
    return ContinuousImage(rast, grid, DeltaPulse{eltype(mimg)}())
end

function convert_table(T, dvis)
    dt = datatable(dvis)

    @reset dt.baseline.U = T.(dt.baseline.U)
    @reset dt.baseline.V = T.(dt.baseline.V)
    @reset dt.baseline.Ti = T.(dt.baseline.Ti)
    @reset dt.baseline.Fr = T.(dt.baseline.Fr)
    @reset dt.measurement = Complex{T}.(dt.measurement)
    @reset dt.noise = T.(dt.noise)
    dvisT = Comrade.rebuild(dvis, dt)
    Td = Comrade.datumtype(dvisT)
    config = arrayconfig(dvisT)
    confT = Comrade.EHTArrayConfiguration(;
        bandwidth=T(config.bandwidth),
        tarr=config.tarr,
        scans=config.scans,
        mjd=config.mjd,
        ra=T(config.ra),
        dec=T(config.dec),
        source=config.source,
        timetype=config.timetype,
        datatable=config.datatable,
    )

    return EHTObservationTable{Td}(dvisT.measurement, dvisT.noise, confT)
end

function build_post(fov, npix, dataf, backend)
    if backend == "TPU"
        T = Float32
    else
        T = Float64
    end

    # @testset "Comrade Integration Imaging" begin
    obs = ehtim.obsdata.load_uvfits(dataf)
    obsavg = obs.add_fractional_noise(0.02)
    dvis0 = extract_table(obsavg, Visibilities())

    dvis = convert_table(T, dvis0)

    npix = npix
    fovx = T(fov)
    fovy = T(fov)

    # Now let's form our cache's. First, we have our usual image cache which is needed to numerically
    # compute the visibilities.
    grd = imagepixels(fovx, fovy, npix, npix)
    pl = StationaryRandomFieldPlan(grd)
    mimg = intensitymap(modify(Gaussian(), Stretch(μas2rad(T(25.0)))), grd)
    skymeta = (; srf=pl, grid=grd, mimg=mimg)

    ρs = ntuple(Returns(Uniform(T(0.01), T(max(size(grd)...)))), 3)
    zprior = std_dist(pl)
    prior = (z=zprior, ρs=ρs, σ=Exponential(T(1.0)))

    skymr = SkyModel(
        sky, prior, grd; metadata=skymeta, algorithm=VLBISkyModels.ReactantAlg()
    ) # Need to do this so that we allocate proper Reactant arrays for internal stuff

    g(x) = exp(complex(x.lg, x.gp))
    G = SingleStokesGain(g)

    intpr = (
        lg=ArrayPrior(
            IIDSitePrior(IntegSeg(), Normal(T(0.0), T(0.2)));
            LM=IIDSitePrior(IntegSeg(), Normal(T(0.0), T(1.0))),
        ),
        gp=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(T(0.0), T(inv(π^2))));
            refant=SEFDReference(T(0.0)),
            phase=true,
        ),
    )
    intmodel = InstrumentModel(G, intpr)

    postr = VLBIPosterior(skymr, intmodel, dvis)
    tpostr = asflat(postr)

    return tpostr
end
