using Test

@testset "Fancy Array Types" begin
    @testset "OffsetArrays" begin
        include("offsetarrays.jl")
    end

    @testset "FillArrays" begin
        include("fillarrays.jl")
    end

    @testset "OneHotArrays" begin
        include("onehotarrays.jl")
    end
end
