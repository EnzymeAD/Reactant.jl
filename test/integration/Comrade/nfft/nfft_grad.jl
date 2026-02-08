using Pkg;
Pkg.activate(@__DIR__);

@static if VERSION ≥ v"1.10-" && VERSION < v"1.11"
    Pkg.add([PackageSpec(; name="Reactant", path=joinpath(@__DIR__, "../../../../"))])
end

using NFFT
using Reactant
using AbstractFFTs
using Test
using Enzyme
using LinearAlgebra

include(joinpath("..", "reactant_nfft.jl"))

T = Float32

sz = (8, 8)
ksz = 100

k = rand(T, 2, ksz) .- T(0.5)

pnf = plan_nfft(NFFTBackend(), k, sz; precompute=NFFT.TENSOR)

# Reactant NFFT plan
pre = ReactantNFFTPlan(k, sz)

img = rand(Complex{T}, sz...)
out = zeros(Complex{T}, ksz)

imgr = Reactant.to_rarray(img)
outr = Reactant.to_rarray(out)

function f(outr, pre, imgr)
    mul!(outr, pre[], imgr)
    return sum(abs2, outr)
end

function gf(outr, pre, imgr)
    grad = Enzyme.gradient(Reverse, f, outr, pre, imgr)
    return last(grad)
end

gr = @jit gf(outr, Ref(pre), imgr)

function fcpu(pnf, x)
    y = pnf * x
    return sum(abs2, y)
end

@test @jit(f(outr, Ref(pre), imgr)) ≈ fcpu(pnf, img)

using FiniteDifferences
fdm = FiniteDifferences.central_fdm(5, 1)
ffix = Base.Fix1(fcpu, pnf)
gfd, = grad(fdm, ffix, img)

@test gr ≈ gfd
