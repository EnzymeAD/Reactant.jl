using Test, SafeTestsets

@testset "CUDA Integration" begin
    @safetestset "CUDA" begin
        include("cuda.jl")
    end

    @safetestset "KernelAbstractions" begin
        include("kernelabstractions.jl")
    end
end
