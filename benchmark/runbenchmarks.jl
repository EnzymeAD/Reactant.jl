using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using CpuId: CpuId
using InteractiveUtils: versioninfo
using LinearAlgebra: BLAS
using Statistics: median

# To run benchmarks on a specific GPU backend, add AMDGPU / CUDA / Metal / oneAPI
# to benchmarks/Project.toml and change BENCHMARK_GROUP to the backend name
const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")
@info "Running benchmarks for $BENCHMARK_GROUP"

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20

if BENCHMARK_GROUP == "CPU"
    if Sys.isapple() && (Sys.ARCH == :aarch64 || Sys.ARCH == :arm64)
        @info "Running benchmarks on Apple with ARM CPUs. Using `AppleAccelerate.jl`."
        using AppleAccelerate: AppleAccelerate
    end

    if Sys.ARCH == :x86_64 && occursin("intel", lowercase(CpuId.cpubrand()))
        @info "Running benchmarks on Intel CPUs. Loading `MKL.jl`."
        using MKL: MKL
    end
end

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

const SUITE = BenchmarkGroup()

const BENCHMARK_CPU_THREADS = Threads.nthreads()
BLAS.set_num_threads(BENCHMARK_CPU_THREADS)

if BENCHMARK_GROUP == "CUDA"
    using LuxCUDA # ] add LuxCUDA to benchmarks/Project.toml
    Reactant.set_default_backend("gpu")
    @info "Running CUDA benchmarks" maxlog=1
    CUDA.versioninfo()
else
    @info "Running CPU benchmarks with $(BENCHMARK_CPU_THREADS) thread(s)" maxlog=1
end

# Main benchmark files
include("setup.jl")
setup_benchmarks!(SUITE, BENCHMARK_GROUP)

results = BenchmarkTools.run(SUITE; verbose=true)

filepath = joinpath(dirname(@__FILE__), "results")
mkpath(filepath)
filename = string(BENCHMARK_GROUP, "benchmarks.json")
BenchmarkTools.save(joinpath(filepath, filename), median(results))

@info "Saved results to $(joinpath(filepath, filename))"
