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
include("reactant_nfft.jl")

using Downloads
using Distributions


using Pyehtim
using Test

const dataurl = "https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/EHTC_M87pol2017_Nov2023/hops_data/April11/SR2_M87_2017_101_lo_hops_ALMArot.uvfits"
const dataf = Base.download(dataurl)


function sky(θ, metadata)
    (;z, ρs, σ) = θ
    (;srf, grid) = metadata
    x = genfield(StationaryRandomField(MarkovPS(ρs), srf), z)
    x .*= σ
    rast = to_simplex(CenteredLR(), x)
    return ContinuousImage(rast, grid, DeltaPulse{3}())
end

@testset "Comrade Integration Imaging" begin 

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
    skymeta = (; srf=pl, grid=grd);


    ρs = ntuple(Returns(Uniform(0.01, max(size(grd)...))), 3)
    zprior = std_dist(pl)
    prior = (
        z = zprior, 
        ρs = ρs,
        σ = Exponential(1.0)
    )

    skymr = SkyModel(sky, prior, grd; metadata=skymeta, algorithm=ReactantAlg()) # Need to do this so that we allocate proper Reactant arrays for internal stuff
    skym  = SkyModel(sky, prior, grd; metadata=skymeta)

    g(x) = exp(complex(x.lg, x.gp))
    G = SingleStokesGain(g)

    intpr = (
        lg = ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.2)); LM = IIDSitePrior(ScanSeg(), Normal(0.0, 1.0))),
        gp = ArrayPrior(IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^2))); refant = SEFDReference(0.0), phase = true),
    )
    intmodel = InstrumentModel(G, intpr)

    postr = VLBIPosterior(skymr, intmodel, dvis)
    post = VLBIPosterior(skym, intmodel, dvis)

    tpost = asflat(post)
    tpostr = asflat(postr)

    x = prior_sample(tpost)
    xr = Reactant.to_rarray(x)
    @jit(logdensityof(tpostr, xr)) ≈ logdensityof(tpost, x)
end