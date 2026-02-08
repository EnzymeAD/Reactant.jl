using Pkg;
Pkg.activate(@__DIR__);

@static if VERSION ≥ v"1.10-" && VERSION < v"1.11"
    Pkg.add([PackageSpec(; name="Reactant", path=joinpath(@__DIR__, "../../../../"))])
end

using NFFT
using CUDA
using Reactant
using LinearAlgebra
using Accessors
using AbstractFFTs
using BenchmarkTools
using NonuniformFFTs

include(joinpath("..", "reactant_nfft.jl"))

T = Float32

sz = (128, 128)
ksz = 5 * 10^2

k = rand(T, 2, ksz) .- T(0.5)

pnf = plan_nfft(NFFTBackend(), k, sz; precompute=NFFT.TENSOR)

kcu = CuArray(k)

# CUDA NFFT plan (sparse)
pnfcu = plan_nfft(NFFTBackend(), CuArray, kcu, sz)

# Make a dense version of the plan for comparison
pnfcu_dense = @set pnfcu.B = CuArray(Array(pnfcu.B))

# Reactant NFFT plan
pre = ReactantNFFTPlan(k, sz)

# NonuniformFFTs plan
pnu = NonuniformFFTs.NFFTPlan(kcu, sz)

img = rand(Complex{T}, sz...)
out = zeros(Complex{T}, ksz)

outcu = CuArray(out)
imgcu = CuArray(img)

imgr = Reactant.to_rarray(img)
outr = Reactant.to_rarray(out)

# Reactant NFFT
Reactant.@profile mul!(outr, pre, imgr)
nfftr! = @compile sync = true mul!(outr, pre, imgr)
@benchmark nfftr!($outr, $pre, $imgr)

function f(outr, pre, imgr)
    mul!(outr, pre[], imgr)
    return sum(abs2, outr)
end

function gf(outr, pre, imgr)
    grad = Enzyme.gradient(Reverse, f, outr, pre, imgr)
    return last(grad)
end

using Enzyme
gr = @jit gf(outr, Ref(pre), imgr)

using FiniteDifferences
fdm = FiniteDifferences.central_fdm(5, 1)
gfd, = grad(fdm, x -> sum(abs2, pnf * x), img)

# NFFT CUDA Dense
CUDA.@sync mul!(outcu, pnfcu_dense, imgcu)
CUDA.@profile mul!(outcu, pnfcu_dense, imgcu)
@benchmark CUDA.@sync mul!($outcu, $pnfcu_dense, $imgcu)

# NFFT CUDA Sparse
CUDA.@sync mul!(outcu, pnfcu, imgcu)  # warmup
CUDA.@profile mul!(outcu, pnfcu, imgcu)
@benchmark CUDA.@sync mul!($outcu, $pnfcu, $imgcu)

# NonuniformFFTs (CUDA)
CUDA.@sync mul!(outcu, pnu, imgcu)  # warmup
CUDA.@profile mul!(outcu, pnu, imgcu)
@benchmark CUDA.@sync mul!($outcu, $pnu, $imgcu)

# NFFT (CPU)
@benchmark mul!($out, $pnf, $img)
