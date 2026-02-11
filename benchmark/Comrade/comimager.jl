
# ENV["JULIA_DEBUG"] = "Reactant"


const dataurl = "https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/EHTC_M87pol2017_Nov2023/hops_data/April06/SR2_M87_2017_096_lo_hops_ALMArot.uvfits"
const dataf = Base.download(dataurl)


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


function build_post(fov, npix, dataf)

    # @testset "Comrade Integration Imaging" begin
    obs = ehtim.obsdata.load_uvfits(dataf)
    obsavg = obs.add_fractional_noise(0.02)
    dvis = extract_table(obsavg, Visibilities())

    npix = npix
    fovx = fov
    fovy = fov

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
    tpostr = asflat(postr)

    return tpostr
end

if abspath(PROGRAM_FILE) == @__FILE__
    tpostr = build_post(μas2rad(200.0), 64, dataf)
    results = Dict()
    backend = get_backend()
    run_comrade_benchmark!(results, "Comrade EHT Imaging 64 x 64", backend, tpostr, "forward", "test")
end