using PythonCall, Reactant, Test

pyimport("sys").path.append(@__DIR__)

add_kernel = pyimport("vector_add").add_kernel

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function vector_add_triton(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    out = similar(x)
    add_kernel(x, y, out, length(x), 1024; grid=(cld(length(x), 1024),))
    return out
end

@testset "vector_add" begin
    if RunningOnCUDA
        x_ra = Reactant.to_rarray(rand(Float32, 2096))
        y_ra = Reactant.to_rarray(rand(Float32, 2096))

        @test @jit(vector_add_triton(x_ra, y_ra)) ≈ @jit(x_ra .+ y_ra)
    end
end
