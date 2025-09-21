using Test

@testset "CUDA Integration" begin
    @testset "CUDA" begin
        include("cuda.jl")
    end

    @testset "KernelAbstractions" begin
        include("kernelabstractions.jl")
    end
end
