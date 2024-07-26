@testsetup module BasicTestSetup

using Enzyme

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))
sinexp(x) = sin(exp(x))
sinexpbc(x) = sinexp.(x)
sumexp(x) = sum(exp, x)
mysoftmax(x) = x .- fastmax(x)

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

mul(A, B) = A * B

export fastmax, sinexp, sinexpbc, sumexp, mysoftmax, sumcos, grad_ip, resgrad_ip, mul

end

@testitem "2D sum" setup=[BasicTestSetup] begin
    r_res = sum(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = sum(a)
    @test c_res ≈ r_res

    f = Reactant.compile(sum, (a,))

    f_res = f(a)

    @test f_res ≈ r_res
end

@testitem "Basic reduce max" setup=[BasicTestSetup] begin
    r_res = fastmax(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = fastmax(a)
    @test c_res ≈ r_res

    f = Reactant.compile(fastmax, (a,))

    f_res = f(a)

    @test f_res ≈ r_res
end


@testitem "Broadcast combined" setup=[BasicTestSetup] begin
    r_res = sinexpbc(ones(2, 10))

    a = Reactant.ConcreteRArray(ones(2, 10))

    c_res = sinexpbc(a)
    @test c_res ≈ r_res

    f = Reactant.compile(sinexpbc, (a,))

    f_res = f(a)

    @test f_res ≈ r_res
end

@testitem "Basic mapreduce" setup=[BasicTestSetup] begin
    x = ones(Float32, 10)
    a = Reactant.ConcreteRArray(x)
    r_res = sumexp(x)

    f = Reactant.compile(sumexp, (a,))
    f_res = f(a)

    @test f_res ≈ r_res
end

@testitem "Basic softmax" setup=[BasicTestSetup] begin
    in = ones(2, 10)
    r_res = mysoftmax(in)

    in = Reactant.ConcreteRArray(ones(2, 10))

    f = Reactant.compile(mysoftmax, (in,))

    f_res = f(in)

    @test f_res ≈ r_res
end

@testitem "Basic cos" setup=[BasicTestSetup] begin
    c = Reactant.ConcreteRArray(ones(3, 2))

    f = Reactant.compile(cos, (c,))
    r = f(c)
    @test r ≈ cos.(ones(3, 2))
end

@testitem "Basic grad cos" setup=[BasicTestSetup] begin
    c = Reactant.ConcreteRArray(ones(3, 2))

    f = Reactant.compile(grad_ip, (c,))
    r = f(c)

    @test r ≈ -sin.(ones(3, 2))

    f = Reactant.compile(resgrad_ip, (c,))
    orig, r = f(c)

    @test orig[2] ≈ sum(cos.(ones(3, 2)))
    @test r ≈ -sin.(ones(3, 2))
end

@testitem "Basic grad cos mul" setup=[BasicTestSetup] begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    d = Reactant.ConcreteRArray(ones(70, 30))

    f = Reactant.compile(mul, (c, d))
    r = f(c, d)

    @test r ≈ mul(ones(50, 70), ones(70, 30))
end

@testitem "ConcreteRArray" setup=[BasicTestSetup] begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    similar(c)
end

@testitem "Reactant.@code_hlo" setup=[BasicTestSetup] begin
    W = Reactant.ConcreteRArray(randn(Float32, 10, 20))
    x = Reactant.ConcreteRArray(randn(Float32, 20, 5))
    res = Reactant.@code_hlo W * x
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.dot_general")
end
