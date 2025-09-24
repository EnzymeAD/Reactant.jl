using Test

@testset "Flux.jl Integration" begin
    @testset "Zygote Integration" begin
        include("zygote.jl")
    end

    @testset "Flux Integration" begin
        include("flux.jl")
    end
end
