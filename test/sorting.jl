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

@testset "findmin / findmax" begin
    xvec = randn(10)
    xvec_ra = Reactant.to_rarray(xvec)

    x = randn(2, 3)
    x_ra = Reactant.to_rarray(x)

    function fwithlinindices(g, f, x; kwargs...)
        values, indices = g(f, x; kwargs...)
        return values, LinearIndices(x)[indices]
    end

    @test fwithlinindices(findmin, identity, x) == @jit(findmin(x_ra))
    @test fwithlinindices(findmax, identity, x) == @jit(findmax(x_ra))
    @test fwithlinindices(findmin, identity, xvec) == @jit(findmin(xvec_ra))
    @test fwithlinindices(findmax, identity, xvec) == @jit(findmax(xvec_ra))

    fmindims(x, d) = findmin(x; dims=d)
    fmindims(f, x, d) = findmin(f, x; dims=d)
    fmaxdims(x, d) = findmax(x; dims=d)
    fmaxdims(f, x, d) = findmax(f, x; dims=d)

    @test fwithlinindices(findmin, identity, x; dims=1) == @jit(fmindims(x_ra, 1))
    @test fwithlinindices(findmax, identity, x; dims=1) == @jit(fmaxdims(x_ra, 1))
    @test fwithlinindices(findmin, identity, x; dims=2) == @jit(fmindims(x_ra, 2))
    @test fwithlinindices(findmax, identity, x; dims=2) == @jit(fmaxdims(x_ra, 2))
    @test fwithlinindices(findmin, abs2, x; dims=1) == @jit(fmindims(abs2, x_ra, 1))
    @test fwithlinindices(findmax, abs2, x; dims=1) == @jit(fmaxdims(abs2, x_ra, 1))
    @test fwithlinindices(findmin, abs2, x; dims=2) == @jit(fmindims(abs2, x_ra, 2))
    @test fwithlinindices(findmax, abs2, x; dims=2) == @jit(fmaxdims(abs2, x_ra, 2))
end

@testset "findfirst / findlast" begin
    x = rand(Bool, 3, 4)
    x_ra = Reactant.to_rarray(x)

    ffirstlinindices(x) = LinearIndices(x)[findfirst(x)]
    ffirstlinindices(f, x) = LinearIndices(x)[findfirst(f, x)]
    flastlinindices(x) = LinearIndices(x)[findlast(x)]
    flastlinindices(f, x) = LinearIndices(x)[findlast(f, x)]

    @test ffirstlinindices(x) == @jit(findfirst(x_ra))
    @test flastlinindices(x) == @jit(findlast(x_ra))

    x = rand(1:256, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test ffirstlinindices(iseven, x) == @jit(findfirst(iseven, x_ra))
    @test flastlinindices(iseven, x) == @jit(findlast(iseven, x_ra))
end
