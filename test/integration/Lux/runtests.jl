using Test

@testset "Lux.jl Integration" begin
    @testset "LuxLib Primitives" begin
        include("luxlib.jl")
    end

    @testset "Lux Integration" begin
        include("lux.jl")
    end
end
