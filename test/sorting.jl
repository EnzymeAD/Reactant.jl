using Reactant, Test

@testset "sort" begin end

@testset "sortperm" begin end

@testset "partialsort" begin end

@testset "partialsortperm" begin end

@testset "argmin / argmax" begin
    x = rand(2, 3)
    x_ra = Reactant.to_rarray(x)

    linargmin(x) = LinearIndices(x)[argmin(x)]
    linargmax(x) = LinearIndices(x)[argmax(x)]

    @test linargmin(x) == @jit(argmin(x_ra))
    @test linargmax(x) == @jit(argmax(x_ra))

    x = rand(2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    linargmin(x, d) = LinearIndices(x)[argmin(x; dims=d)]
    linargmax(x, d) = LinearIndices(x)[argmax(x; dims=d)]
    argmindims(x, d) = argmin(x; dims=d)
    argmaxdims(x, d) = argmax(x; dims=d)

    @test linargmin(x, 1) == @jit(argmindims(x_ra, 1))
    @test linargmax(x, 1) == @jit(argmaxdims(x_ra, 1))
    @test linargmin(x, 2) == @jit(argmindims(x_ra, 2))
    @test linargmax(x, 2) == @jit(argmaxdims(x_ra, 2))
    @test linargmin(x, 3) == @jit(argmindims(x_ra, 3))
    @test linargmax(x, 3) == @jit(argmaxdims(x_ra, 3))

    x = randn(2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test argmin(abs2, x) == @jit(argmin(abs2, x_ra))
    @test argmax(abs2, x) == @jit(argmax(abs2, x_ra))
end
