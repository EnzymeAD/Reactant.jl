using NFFT
using CUDA
using Reactant
using VLBISkyModels
using LinearAlgebra

include("reactant_nfft.jl")


sz =  (128, 128)
ksz = 256

k = rand(2, ksz) .- 0.5

pnf = plan_nfft(NFFTBackend(), k, sz; precompute=NFFT.TENSOR)

kcu = CuArray(k)
pnfcu = plan_nfft(NFFTBackend(), CuArray, kcu, sz)
pre = ReactantNFFTPlan(k, sz)


img = rand(ComplexF64, sz...)
out = zeros(ComplexF64, ksz)

outcu = CuArray(out)
imgcu = CuArray(img)

imgr = Reactant.to_rarray(img)
outr = Reactant.to_rarray(out)

rnfft! = @compile sync=true mul!(outr, pre, imgr)
@benchmark rnfft!(outr, pre, imgr)

@benchmark CUDA.@sync mul!($outcu, $pnfcu, $imgcu)

@benchmark mul!($out, $pnf, $img)