using PythonCall, Reactant, Test

pyimport("sys").path.append(@__DIR__)

asin_kernel = pyimport("libdevice").asin_kernel

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function asin_triton(x::AbstractVector{T}) where {T}
    out = similar(x)
    asin_kernel(x, out, length(x), 1024; grid=(cld(length(x), 1024),))
    return out
end

@testset "libdevice asin" begin
    if RunningOnCUDA
        x_ra = Reactant.to_rarray(rand(Float32, 2096))

        @test @jit(asin_triton(x_ra)) ≈ @jit(asin.(x_ra))
    end
end
