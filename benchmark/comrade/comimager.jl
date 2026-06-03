@sky function skym(grid; T, srf, mimg)
    (; z, ρs, σ) = θ
    z ~ std_dist(srf)
    ρs ~ ntuple(Returns(VLBIUniform(T(0.01), T(max(size(grid)...)))), 3)
    fg ~ VLBIUniform(T(0.0), T(1.0))
    σ ~ VLBIExponential(T(1.0))
    x = genfield(StationaryRandomField(MarkovPS(ρs), srf), z)
    x .*= σ
    mx = maximum(x)
    bmimg = baseimage(mimg)
    rast = @. exp(x - mx) * bmimg
    rast ./= sum(rast)
    return ContinuousImage(rast, grid, DeltaPulse{T}())
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

function build_post(::Type{T}, fov, npix, dataf) where {T}
    uvd = VLBIFiles.load(VLBIFiles.UVData, dataf)
    dvis0 = extract_table(uvd, Visibilities(; time_average=VLBI.GapBasedScans()))
    add_fractional_noise!(dvis0, T(0.01))

    dvis = convert_table(T, dvis0)

    npix = npix
    fovx = T(fov)
    fovy = T(fov)

    # Now let's form our cache's. First, we have our usual image cache which is needed
    # to numerically compute the visibilities.
    grd = imagepixels(fovx, fovy, npix, npix)
    pl = StationaryRandomFieldPlan(grd)
    mimg = intensitymap(modify(Gaussian(), Stretch(μas2rad(T(25.0)))), grd)
    skymr = skym(grd; T=T, srf=pl, mimg=mimg)
    

    g(x) = exp(complex(x.lg, x.gp))
    G = SingleStokesGain(g)

    intpr = (
        lg=ArrayPrior(
            IIDSitePrior(IntegSeg(), VLBIGaussian(T(0.0), T(0.2)));
            LM=IIDSitePrior(IntegSeg(), VLBIGaussian(T(0.0), T(1.0))),
        ),
        gp=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(T(0.0), T(inv(π^2))));
            refant=SEFDReference(T(0.0)),
            phase=true,
        ),
    )
    intmodel = InstrumentModel(G, intpr)

    postr = Comrade.prepare_device(VLBIPosterior(skymr, intmodel, dvis), ComradeBase.ReactantEx())
    tpostr = asflat(postr)

    return tpostr
end
