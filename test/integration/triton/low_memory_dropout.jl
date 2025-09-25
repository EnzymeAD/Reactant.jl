using PythonCall, Reactant, Test

pyimport("sys").path.append(@__DIR__)

low_memory_dropout_kernel = pyimport("low_memory_dropout").seeded_dropout_kernel

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function seeded_dropout(x::AbstractVector{T}, p::Number, seed) where {T}
    output = similar(x)
    mask = similar(x, Bool)
    low_memory_dropout_kernel(
        x, output, mask, length(x), p, seed, 1024; grid=(cld(length(x), 1024),)
    )
    return output, mask
end

function apply_dropout(x::AbstractVector{T}, mask::AbstractVector, p::Number) where {T}
    return x .* mask ./ (1 - p)
end

@testset "low_memory_dropout" begin
    if RunningOnCUDA
        x_ra = Reactant.to_rarray(rand(Float32, 2056))

        out, mask = @jit seeded_dropout(x_ra, 0.25f0, ConcreteRNumber(123))

        @test @jit(apply_dropout(x_ra, mask, 0.25f0)) ≈ out
    end
end
