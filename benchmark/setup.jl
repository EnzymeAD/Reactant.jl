using Boltz: Vision
using Lux: Lux
using MLDataDevices: AbstractDevice, CPUDevice, CUDADevice
using Random: Random
using Reactant: Reactant, @compile

using Enzyme: Enzyme
using Zygote: Zygote

# Helper Functions
@inline synchronize(::CPUDevice) = nothing
@inline synchronize(::CUDADevice) = CUDA.synchronize()

@inline reclaim(::CPUDevice) = GC.gc()
@inline reclaim(::CUDADevice) = CUDA.reclaim()

@inline sumabs2(model, x, p, st) = sum(abs2, first(Lux.apply(model, x, p, st)))
@inline sumabs2(model, x) = sum(abs2, model(x))

function benchmark_group_to_backend(benchmark_group::String)
    benchmark_group == "CPU" && return CPUDevice()
    benchmark_group == "CUDA" && return CUDADevice()
    return error("Unknown backend: $(benchmark_group)")
end

function general_lux_setup(model, x_dims)
    rng = Random.default_rng()  # don't use any other rng
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

function setup_benchmarks!(suite::BenchmarkGroup, backend::String)
    dev = benchmark_group_to_backend(backend)

    setup_vit_benchmark!(suite, backend, dev)

    return nothing
end

# Lux Benchmarks
function setup_vit_benchmark!(suite::BenchmarkGroup, backend, dev::AbstractDevice)
    for mode in (:tiny, :small, :base), bsize in (4, 16, 32)
        benchmark_name = "ViT $(mode) (256 x 256 x 3 x $(bsize))"

        setup_lux_forward_pass_benchmark!(
            suite, benchmark_name, backend, Vision.ViT(mode), (256, 256, 3, bsize), dev
        )
    end
end

function setup_lux_forward_pass_benchmark!(
    suite::BenchmarkGroup,
    benchmark_name::String,
    backend::String,
    model,
    x_dims,
    dev::AbstractDevice,
)
    suite[benchmark_name]["forward"][backend]["Lux"] = @benchmarkable begin
        Lux.apply($model, x, ps, st_test)
        synchronize($dev)
    end setup = begin
        reclaim($dev)
        x, ps, st = $dev(general_lux_setup($model, $x_dims))
        st_test = Lux.testmode(st)
    end

    suite[benchmark_name]["forward"][backend]["Reactant"] = @benchmarkable begin
        y, _ = apply_compiled($model, x_ra, ps_ra, st_test_ra)
        Reactant.synchronize(y)
    end setup = begin
        reclaim($dev)
        x, ps, st = general_lux_setup($model, $x_dims)
        st_test = Lux.testmode(st)
        x_ra = Reactant.to_rarray(x)
        ps_ra = Reactant.to_rarray(ps)
        st_test_ra = Reactant.to_rarray(st_test)
        apply_compiled = @compile Lux.apply($model, x_ra, ps_ra, st_test_ra)
    end

    return nothing
end
