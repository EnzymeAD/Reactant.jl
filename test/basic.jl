using Reactant
using Test
using Enzyme

fastmax(x::AbstractArray{T}) where T = reduce(max, x; dims=1, init = float(T)(-Inf))

@testset "Basic reduce max" begin
    r_res = fastmax(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = fastmax(a)
    @test c_res ≈ r_res

    f=Reactant.compile(fastmax, (a,))
    
    f_res = f(a)

    @test f_res ≈ r_res
end

function softmax!(x)
    max_ = fastmax(x)
    return x .- max_
end

@testset "Basic softmax" begin

    in = ones(2, 10)
    r_res = softmax!(in)

    in = Reactant.ConcreteRArray(ones(2, 10))

    f=Reactant.compile(softmax!, (in,))
    
    f_res = f(in)
    
    @test f_res ≈ r_res
end


@testset "Basic cos" begin
    c = Reactant.ConcreteRArray(ones(3,2))

    f=Reactant.compile(cos, (c,))
    r = f(c)
    @test r ≈ cos.(ones(3,2))
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
    c = Reactant.ConcreteRArray(ones(3,2))

    f=Reactant.compile(grad_ip, (c,))
    r = f(c)

    @test r ≈ -sin.(ones(3,2))

    f=Reactant.compile(resgrad_ip, (c,))
    orig, r = f(c)

    @test orig[2] ≈ sum(cos.(ones(3,2)))
    @test r ≈ -sin.(ones(3,2))
end
