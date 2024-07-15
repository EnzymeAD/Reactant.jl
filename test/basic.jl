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

    f_res = f(a)

    @test f_res ≈ r_res
end

@testset "Basic reduce max" begin
    r_res = fastmax(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = fastmax(a)
    @test c_res ≈ r_res

    f = Reactant.compile(fastmax, (a,))

    f_res = f(a)

    @test f_res ≈ r_res
end

sinexp(x) = sin(exp(x))
sinexpbc(x) = sinexp.(x)

@testset "Broadcast combined" begin
    r_res = sinexpbc(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = sinexpbc(a)
    @test c_res ≈ r_res

    f = Reactant.compile(sinexpbc, (a,))

    f_res = f(a)

    @test f_res ≈ r_res
end

sumexp(x) = sum(exp, x)

@testset "Basic mapreduce" begin
    x = ones(Float32, 10)
    a = Reactant.ConcreteRArray(x)
    r_res = sumexp(x)

    f = Reactant.compile(sumexp, (a,))
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
    Enzyme.autodiff(Reverse, sumcos, Active, Duplicated(x, dx))
    return dx
end

function resgrad_ip(x)
    dx = Enzyme.make_zero(x)
    res = Enzyme.autodiff(ReverseWithPrimal, sumcos, Active, Duplicated(x, dx))
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

@testset "Reactant.@code_hlo" begin
    W = Reactant.ConcreteRArray(randn(Float32, 10, 20))
    # x = Reactant.ConcreteRArray(randn(Float32, 20, 5))
    # res = Reactant.@code_hlo W * x
    res = Reactant.@code_hlo sum(W)
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.reduce")
end
