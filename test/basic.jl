using Reactant
using Test
using Enzyme

# Reactant.set_default_backend("gpu")

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))

using InteractiveUtils

@testset "2D sum" begin
    r_res = sum(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = sum(a)
    @test c_res ≈ r_res

    f = Reactant.compile(sum, (a,))

    @show @code_typed f(a)
    @show @code_llvm f(a)

    f_res = f(a)

    @test f_res ≈ r_res
end

@testset "Basic reduce max" begin
    r_res = fastmax(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = fastmax(a)
    @test c_res ≈ r_res

    f = Reactant.compile(fastmax, (a,))

    @show @code_typed f(a)
    @show @code_llvm f(a)

    f_res = f(a)

    @test f_res ≈ r_res
end

function mysoftmax!(x)
    max_ = fastmax(x)
    return x .- max_
end

@testset "Basic softmax" begin
    in = ones(2, 10)
    r_res = mysoftmax!(in)

    in = Reactant.ConcreteRArray(ones(2, 10))

    f = Reactant.compile(mysoftmax!, (in,))

    f_res = f(in)

    @test f_res ≈ r_res
end

@testset "Basic cos" begin
    c = Reactant.ConcreteRArray(ones(3, 2))

    f = Reactant.compile(cos, (c,))
    r = f(c)
    @test r ≈ cos.(ones(3, 2))
end

function sumcos(x)
    return sum(cos.(x))
end

function grad_ip(x)
    dx = Enzyme.make_zero(x)
    Reactant.autodiff(Reverse, sumcos, Active, Duplicated(x, dx))
    return dx
end

function resgrad_ip(x)
    dx = Enzyme.make_zero(x)
    res = Reactant.autodiff(ReverseWithPrimal, sumcos, Active, Duplicated(x, dx))
    return (res, dx)
end

@testset "Basic grad cos" begin
    c = Reactant.ConcreteRArray(ones(3, 2))

    f = Reactant.compile(grad_ip, (c,))
    r = f(c)

    @test r ≈ -sin.(ones(3, 2))

    f = Reactant.compile(resgrad_ip, (c,))
    orig, r = f(c)

    @test orig[2] ≈ sum(cos.(ones(3, 2)))
    @test r ≈ -sin.(ones(3, 2))
end

function mul(A, B)
    return A * B
end
@testset "Basic grad cos" begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    d = Reactant.ConcreteRArray(ones(70, 30))

    f = Reactant.compile(mul, (c, d))
    r = f(c, d)

    @test r ≈ mul(ones(50, 70), ones(70, 30))
end

@testset "ConcreteRArray" begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    similar(c)
end
