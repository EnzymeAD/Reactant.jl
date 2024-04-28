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


