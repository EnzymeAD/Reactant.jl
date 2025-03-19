using Reactant
using Test
using Enzyme
using Statistics
using Random
Random.seed!(123)

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))

using InteractiveUtils

@testset "2D sum" begin
    x = rand(2, 10)

    r_res = sum(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar sum(a)
    @test c_res ≈ r_res

    @test @jit(sum(a)) ≈ r_res
end

@testset "Basic reduce max" begin
    x = rand(2, 10)

    r_res = fastmax(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar fastmax(a)
    @test c_res ≈ r_res

    @test @jit(fastmax(a)) ≈ r_res
end

sinexp(x) = sin(exp(x))
sinexpbc(x) = sinexp.(x)

@testset "Broadcast combined" begin
    x = rand(2, 10)

    r_res = sinexpbc(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar sinexpbc(a)
    @test c_res isa ConcreteRArray
    @test c_res ≈ r_res
    @test @jit(sinexpbc(a)) ≈ r_res
end

sumexp(x) = sum(exp, x)

sum_compare(x) = sum(x) > 0

@testset "Basic mapreduce" begin
    x = rand(Float32, 10)
    a = Reactant.to_rarray(x)
    r_res = sumexp(x)

    f_res = @jit sumexp(a)

    @test f_res ≈ r_res

    # Ensure we are tracing as scalars. Else this will fail due to > not being defined on
    # arrays
    @test @jit(sum_compare(a)) == sum_compare(x)
end

function mysoftmax!(x)
    max_ = fastmax(x)
    return x .- max_
end

@testset "Basic softmax" begin
    x = rand(2, 10)
    r_res = mysoftmax!(x)

    a = Reactant.to_rarray(x)

    f_res = @jit mysoftmax!(a)
    @test f_res ≈ r_res
end

bcast_cos(x) = cos.(x)

@testset "Basic cos" begin
    x = rand(3, 2)
    c = Reactant.to_rarray(x)

    @test @jit(bcast_cos(c)) ≈ cos.(x)
end

f_var(args...) = sum(args)

@testset "Vararg" begin
    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(3 * ones(3))
    z = Reactant.to_rarray(2.6 * ones(3))

    @test @jit(f_var(x, y, z)) ≈ [6.6, 6.6, 6.6]
end

sumcos(x) = sum(cos.(x))

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
    c = Reactant.to_rarray(ones(3, 2))

    @test @jit(grad_ip(c)) ≈ -sin.(ones(3, 2))

    orig, r = @jit(resgrad_ip(c))

    @test orig[2] ≈ sum(cos.(ones(3, 2)))
    @test r ≈ -sin.(ones(3, 2))
end

@testset "matmul" begin
    c = Reactant.to_rarray(ones(50, 70))
    d = Reactant.to_rarray(ones(70, 30))

    @test @jit(*(c, d)) ≈ *(ones(50, 70), ones(70, 30))
end

@testset "similar Reactant.to_rarray" begin
    c = Reactant.to_rarray(ones(50, 70))
    sim_c = similar(c)
    @test typeof(sim_c) == typeof(c) && size(sim_c) == size(sim_c)
end

@testset "@code_hlo" begin
    W = Reactant.to_rarray(randn(Float32, 10, 20))
    x = Reactant.to_rarray(randn(Float32, 20, 5))
    res = @code_hlo W * x
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.dot_general")
end

@testset "@code_hlo broadcasting" begin
    x = Reactant.to_rarray(randn(Float32, 2, 2))
    y = Reactant.to_rarray(randn(Float32, 2, 2))
    res = @code_hlo (.+)(x, y)
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.add")
end

@testset "Statistics: `mean` & `var`" begin
    x = randn(2, 3, 4)
    x_ca = Reactant.to_rarray(x)

    @test @jit(mean(x_ca)) ≈ mean(x)
    @test @jit(mean(x_ca; dims=1)) ≈ mean(x; dims=1)
    @test @jit(mean(x_ca; dims=(1, 2))) ≈ mean(x; dims=(1, 2))
    @test @jit(mean(x_ca; dims=(1, 3))) ≈ mean(x; dims=(1, 3))

    @test @jit(var(x_ca)) ≈ var(x)
    @test @jit(var(x_ca, dims=1)) ≈ var(x; dims=1)
    @test @jit(var(x_ca, dims=(1, 2); corrected=false)) ≈
        var(x; dims=(1, 2), corrected=false)
    @test @jit(var(x_ca; dims=(1, 3), corrected=false)) ≈
        var(x; dims=(1, 3), corrected=false)
end

@testset "concatenation" begin
    @testset "Number" begin
        x = fill(true)
        x_concrete = Reactant.to_rarray(x)

        # NOTE [,,,] is a call to `vect`, not `*cat`
        # f = Reactant.compile((x_concrete,)) do x
        #     return [x, x, x]
        # end
        # @test f(x_concrete) ≈ ones(3)

        # vcat
        test_vcat(x) = begin
            x = x[] # unwrap scalar
            [x; x; x]
        end
        y = @jit test_vcat(x_concrete)
        @test y == test_vcat(x)
        @test eltype(y) === Bool

        # hcat
        test_hcat(x) = begin
            x = x[] # unwrap scalar
            [x x x]
        end
        y = @jit test_hcat(x_concrete)
        @test y == test_hcat(x)
        @test eltype(y) === Bool

        # hvcat
        test_hvcat(x) = begin
            x = x[] # unwrap scalar
            [x x x; x x x]
        end
        y = @jit test_hvcat(x_concrete)
        @test y == test_hvcat(x)
        @test eltype(y) === Bool

        # hvncat
        test_hvncat(x) = begin
            x = x[] # unwrap scalar
            [x x x; x x x;;; x x x; x x x]
        end
        y = @jit test_hvncat(x_concrete)
        @test y == test_hvncat(x)
        @test eltype(y) === Bool

        # typed_vcat
        test_typed_vcat(x) = begin
            x = x[] # unwrap scalar
            Int[x; x; x]
        end
        y = @jit test_typed_vcat(x_concrete)
        @test y == test_typed_vcat(x)
        @test eltype(y) === Int

        # typed_hcat
        test_typed_hcat(x) = begin
            x = x[] # unwrap scalar
            Int[x x x]
        end
        y = @jit test_typed_hcat(x_concrete)
        @test y == test_typed_hcat(x)
        @test eltype(y) === Int

        # typed_hvcat
        test_typed_hvcat(x) = begin
            x = x[] # unwrap scalar
            Int[x x x; x x x]
        end
        y = @jit test_typed_hvcat(x_concrete)
        @test y == test_typed_hvcat(x)
        @test eltype(y) === Int

        # typed_hvncat
        test_typed_hvncat(x) = begin
            x = x[] # unwrap scalar
            Int[x x x; x x x;;; x x x; x x x]
        end
        y = @jit test_typed_hvncat(x_concrete)
        @test y == test_typed_hvncat(x)
        @test eltype(y) === Int
    end

    @testset "$(ndims(x))-dim Array" for x in [
        fill(true),
        [true, false],
        [true false],
        [true true; true false],
        [
            true true true true; true true true false;;;
            true true false true; true true false false;;;
            true false true true; true false true false
        ],
    ]
        x_concrete = Reactant.to_rarray(x)

        # NOTE [,,,] is a call to `vect`, not `*cat`
        # f = Reactant.compile((x_concrete,)) do x
        #     return [x, x, x]
        # end
        # @test f(x_concrete) ≈ ones(3)

        # vcat
        test_vcat(x) = [x; x; x]
        y = @jit test_vcat(x_concrete)
        @test y == test_vcat(x)
        @test eltype(y) === Bool

        # hcat
        test_hcat(x) = [x x x]
        y = @jit test_hcat(x_concrete)
        @test y == test_hcat(x)
        @test eltype(y) === Bool

        # hvcat
        test_hvcat(x) = [x x x; x x x]
        y = @jit test_hvcat(x_concrete)
        @test y == test_hvcat(x)
        @test eltype(y) === Bool

        # hvncat
        test_hvncat(x) = [x x x; x x x;;; x x x; x x x]
        y = @jit test_hvncat(x_concrete)
        @test y == test_hvncat(x)
        @test eltype(y) === Bool

        # typed_vcat
        test_typed_vcat(x) = Int[x; x; x]
        y = @jit test_typed_vcat(x_concrete)
        @test y == test_typed_vcat(x)
        @test eltype(y) === Int

        # typed_hcat
        test_typed_hcat(x) = Int[x x x]
        y = @jit test_typed_hcat(x_concrete)
        @test y == test_typed_hcat(x)
        @test eltype(y) === Int

        # typed_hvcat
        test_typed_hvcat(x) = Int[x x x; x x x]
        y = @jit test_typed_hvcat(x_concrete)
        @test y == test_typed_hvcat(x)
        @test eltype(y) === Int

        # typed_hvncat
        test_typed_hvncat(x) = Int[x x x; x x x;;; x x x; x x x]
        y = @jit test_typed_hvncat(x_concrete)
        @test y == test_typed_hvncat(x)
        @test eltype(y) === Int
    end

    @testset "Number and RArray" for a in [1.0f0, 1.0e0]
        typeof_a = typeof(a)
        _b = typeof_a.([2.0, 3.0, 4.0])
        _c = typeof_a.([2.0 3.0 4.0])
        b = Reactant.to_rarray(_b)
        c = Reactant.to_rarray(_c)

        # vcat test
        y = @jit vcat(a, b)
        @test y == vcat(a, _b)
        @test y isa ConcreteRArray{typeof_a,1}

        ## vcat test - adjoint
        y1 = @jit vcat(a, c')
        @test y1 == vcat(a, _c')
        @test y1 isa ConcreteRArray{typeof_a,2}

        # hcat test
        z = @jit hcat(a, c)
        @test z == hcat(a, _c)
        @test z isa ConcreteRArray{typeof_a,2}

        ## hcat test - adjoint
        z1 = @jit hcat(a, b')
        @test z1 == hcat(a, _b')
        @test z1 isa ConcreteRArray{typeof_a,2}
    end
end

@testset "repeat" begin
    @testset for (size, counts) in Iterators.product(
        [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
        [(), (1,), (2,), (2, 1), (1, 2), (2, 2), (2, 2, 2), (1, 1, 1, 1, 1)],
    )
        x = rand(size...)

        @testset "outer repeat" begin
            @test (@jit repeat(Reactant.to_rarray(x), counts...)) == repeat(x, counts...)
        end

        length(counts) < length(size) && continue

        @testset "inner repeat" begin
            @test (@jit repeat(Reactant.to_rarray(x); inner=counts)) ==
                repeat(x; inner=counts)
        end
    end
end

tuple_byref(x) = (; a=(; b=x))
tuple_byref2(x) = abs2.(x), tuple_byref(x)

@testset "Tuple byref" begin
    x = Reactant.to_rarray([1.0 -2.0; -3.0 4.0])
    @test @jit(tuple_byref(x)).a.b.data === x.data

    f2 = @compile tuple_byref2(x)
    r2 = f2(x)
    @test r2[2].a.b.data === x.data
    @test r2[1] == abs2.([1.0 -2.0; -3.0 4.0])
end

sum_xxᵀ(x) = sum(x .* x')

@testset "sum(x .* x')" begin
    @testset "size(x): $(size(x))" for x in (rand(4, 4), rand(4))
        x_ca = Reactant.to_rarray(x)

        @test @jit(sum_xxᵀ(x_ca)) ≈ sum_xxᵀ(x)
    end
end

@testset "similar" begin
    x = zeros(2, 3)
    y = Reactant.to_rarray(x)
    f = @compile similar(y)
    @test size(f(y)) == size(x)
    @test eltype(f(y)) == eltype(x)
end

@testset "Complex runtime: $CT" for CT in (ComplexF32, ComplexF64)
    a = Reactant.to_rarray(ones(CT, 2))
    b = Reactant.to_rarray(ones(CT, 2))
    c = Reactant.compile(+, (a, b))(a, b)
    @test c == ones(CT, 2) + ones(CT, 2)
end

@testset "Scalars" begin
    @testset "Only Scalars" begin
        x = (3, 3.14)

        f1(x) = x[1] * x[2]

        x_ra = Reactant.to_rarray(x; track_numbers=Number)
        f2 = @compile f1(x_ra)
        @test f2(Reactant.to_rarray((5, 5.2); track_numbers=Number)) ≈ 5 * 5.2
        @test f2(Reactant.to_rarray((5, 5.2); track_numbers=Number)) isa ConcreteRNumber

        x_ra = Reactant.to_rarray(x)
        f3 = @compile f1(x_ra)
        @test f3(Reactant.to_rarray((5, 5.2))) ≈ f1(x)
        @test !(f3(Reactant.to_rarray((5, 5.2))) isa ConcreteRNumber)
        @test f3(Reactant.to_rarray((5, 5.2))) isa Number

        x_ra = Reactant.to_rarray(x; track_numbers=Int)
        f4 = @compile f1(x_ra)
        @test f4(Reactant.to_rarray((5, 5.2); track_numbers=Int)) ≈ 5 * 3.14
        @test f4(Reactant.to_rarray((5, 5.2); track_numbers=Int)) isa ConcreteRNumber
    end

    @testset "Mixed" begin
        x = (3, [3.14])

        f1(x) = x[1] * x[2]

        x_ra = Reactant.to_rarray(x; track_numbers=Number)

        f2 = @compile f1(x_ra)
        res2 = f2(Reactant.to_rarray((5, [3.14]); track_numbers=Number))
        @test @allowscalar(only(res2)) ≈ 5 * 3.14
        @test res2 isa ConcreteRArray

        x_ra = Reactant.to_rarray(x)

        f3 = @compile f1(x_ra)
        res3 = f3(Reactant.to_rarray((5, [3.14])))
        @test @allowscalar(only(res3)) ≈ only(f1(x))
        @test res3 isa ConcreteRArray
    end
end

relu(x::T) where {T<:Number} = max(T(0), x)
relu(x) = relu.(x)

@testset "type casting" begin
    x = randn(2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(relu(x_ra)) ≈ relu(x)
end

@testset "concrete number to julia number" begin
    x = Reactant.to_rarray(3.14; track_numbers=Number)
    @test Float32(x) isa Float32
    @test Float64(x) isa Float64
    @test_throws InexactError Int(x)

    x = Reactant.to_rarray(3; track_numbers=Number)
    @test Float32(x) isa Float32
    @test Float64(x) isa Float64
    @test Int(x) isa Int
    @test float(x) isa ConcreteRNumber{Float64}
end

@testset "concrete number with fill" begin
    x = Reactant.to_rarray(10; track_numbers=Number)
    x_ra = @jit fill(x, (10, 10))
    @test fill(x, (10, 10)) == Array(x_ra)
end

@testset "clamp" begin
    x = randn(2, 3)
    x_ra = Reactant.to_rarray(x)

    y = @jit(clamp!(x_ra, 0.0, 0.25))
    @allowscalar begin
        @test maximum(y) ≤ 0.25
        @test minimum(y) ≥ 0.0
        @test maximum(x_ra) == maximum(y)
        @test minimum(x_ra) == minimum(y)
    end

    x = randn(2, 3)
    x_ra = Reactant.to_rarray(x)

    y = @jit(clamp.(x_ra, 0.0, 0.25))
    @allowscalar begin
        @test maximum(y) ≤ 0.25
        @test minimum(y) ≥ 0.0
        @test x_ra ≈ x
    end
end

@testset for op in [round, ceil, floor]
    for x in (rand(Float32, (3, 3)), rand(Float64))
        intop = Base.Fix1(op, Int)
        x_ra = Reactant.to_rarray.(x; track_numbers=Number)

        @test @jit(op.(x_ra)) ≈ op.(x)
        @test @jit(intop.(x_ra)) ≈ intop.(x)
    end
end

@testset "sign" begin
    x = collect(Float64, 0:0.01:1) .- 0.5
    x_ra = Reactant.to_rarray(x)
    @test Array(@jit(sign.(x_ra))) ≈ sign.(x)
end

@testset "aos_to_soa" begin
    using ArrayInterface

    x_res = collect(reshape(1.0:4.0, 2, 1, 2))
    x_ca = Reactant.to_rarray.(x_res; track_numbers=Number)

    y_ca1 = @allowscalar ArrayInterface.aos_to_soa(x_ca)
    @test y_ca1 ≈ x_res
    @test y_ca1 isa ConcreteRArray

    y_ca2 = @jit(ArrayInterface.aos_to_soa(x_ca))
    @test y_ca2 ≈ x_res
    @test y_ca2 isa ConcreteRArray
end

@testset "collect" begin
    x = randn(2, 3)
    x_ra = Reactant.to_rarray(x)

    @testset "Reactant.to_rarray" begin
        y = collect(x_ra)
        @test y == x
        @test y !== x_ra
    end

    @testset "TracedRArray" begin
        y = @jit(collect(x_ra))
        @test y == x
        @test y !== x_ra
    end

    x = 5
    x_ra = ConcreteRNumber(x)

    @testset "ConcreteRNumber" begin
        y = collect(x_ra)
        @test y isa Array{Int,0}
    end

    @testset "TracedRArray" begin
        y = @jit(collect(x_ra))
        @test y isa ConcreteRArray{Int,0}
        @test y == x
    end
end

function f_row_major(x::AbstractArray{T}) where {T}
    y = [1 2; 3 4; 5 6]
    if x isa Reactant.TracedRArray
        y = Reactant.TracedUtils.promote_to(
            Reactant.TracedRArray{Reactant.unwrapped_eltype(T),2}, y
        )
    end
    return x .+ y
end

@testset "array attributes: row major" begin
    x = zeros(Int, 3, 2)
    x_ra = Reactant.to_rarray(x)

    @test @jit(f_row_major(x_ra)) ≈ f_row_major(x)
end

@testset "ifelse" begin
    @test 1.0 == @test_warn r"`ifelse` with different element-types" @jit(
        ifelse(ConcreteRNumber(true), ConcreteRNumber(1.0), ConcreteRNumber(0.0f0))
    )
    @test @jit(
        ifelse(ConcreteRNumber(false), ConcreteRNumber(1.0), ConcreteRNumber(0.0f0))
    ) isa ConcreteRNumber{Float64}
    @test 0.0f0 ==
        @jit ifelse(ConcreteRNumber(false), ConcreteRNumber(1.0), ConcreteRNumber(0.0f0))
    @test @jit(
        ifelse(ConcreteRNumber(false), ConcreteRNumber(1.0f0), ConcreteRNumber(0.0f0))
    ) isa ConcreteRNumber{Float32}

    cond = ConcreteRNumber(true)
    x = ConcreteRNumber(1.0)
    @test @jit(ifelse(cond, x, 0.0)) == ConcreteRNumber(1.0)
    @test @jit(ifelse(cond, 0.0, x)) == ConcreteRNumber(0.0)
    @test @jit(ifelse(cond, 1.0, 0.0)) == ConcreteRNumber(1.0)
    @test @jit(ifelse(cond, 0.0, 1.0)) == ConcreteRNumber(0.0)
end

@testset "fill! and zero on Reactant.to_rarray" begin
    x_ra = Reactant.to_rarray(rand(3, 4))

    z = zero(x_ra)
    @test z isa ConcreteRArray
    @test size(z) == size(x_ra)
    @test all(iszero, Array(z))

    fill!(z, 1.0)
    @test all(==(1.0), Array(z))
end

@testset "Preserve Aliasing" begin
    x = Reactant.to_rarray([3])

    if x isa ConcretePJRTArray
        # For IFRT arrays we don't have unsafe_buffer_pointer implemented
        T = Any[nothing]

        function ip(m, T)
            @allowscalar m[1] = 2
            T[1] = m
            return m
        end

        res = @jit ip(x, T)
        @test @allowscalar res[1] == 2
        @test @allowscalar x[1] == 2
        @test @allowscalar T[1][1] == 2

        ptr_x = Base.unsafe_convert(
            Ptr{Float64}, Reactant.XLA.unsafe_buffer_pointer(x.data[1].buffer)
        )
        ptr_res = Base.unsafe_convert(
            Ptr{Float64}, Reactant.XLA.unsafe_buffer_pointer(res.data[1].buffer)
        )
        ptr_T1 = Base.unsafe_convert(
            Ptr{Float64}, Reactant.XLA.unsafe_buffer_pointer(T[1].data[1].buffer)
        )

        @test ptr_x == ptr_res == ptr_T1
    end
end

@testset "eltype conversion inside interpreter" begin
    function test_convert(x::AbstractArray{T}, eta) where {T}
        eta = T(eta)
        return x .* eta, eta
    end

    res = @jit test_convert(
        Reactant.to_rarray(rand(4, 2)), Reactant.to_rarray(3.0f0; track_numbers=Number)
    )

    @test res[1] isa ConcreteRArray{Float64,2}
    @test res[2] isa ConcreteRNumber{Float64}
end

@testset "stack" begin
    x = rand(4, 4)
    y = rand(4, 4)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(stack((x_ra, x_ra))) ≈ stack((x, x))
    @test @jit(stack((x_ra, x_ra); dims=2)) ≈ stack((x, x); dims=2)
    @test @jit(stack((x_ra, y_ra); dims=2)) ≈ stack((x, y); dims=2)
    @test @jit(stack((x_ra, y_ra, x_ra); dims=1)) ≈ stack((x, y, x); dims=1)

    # Test that we don't hit illegal instruction; `x` is intentionally not a traced array
    @test @jit(stack((x, x))) isa Any
    @test @jit(stack((x, x); dims=2)) isa Any
    @test @jit(stack((x, y); dims=2)) isa Any
    @test @jit(stack((x, y, x); dims=1)) isa Any
end

@testset "unstable stack" begin
    x = rand(4, 4)
    y = rand(4, 4)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    function s1(x)
        xs = []
        push!(xs, x)
        push!(xs, x)
        return stack(xs)
    end
    function s2(x)
        xs = []
        push!(xs, x)
        push!(xs, x)
        return stack(xs; dims=2)
    end
    function s3(x, y)
        xs = []
        push!(xs, x)
        push!(xs, y)
        return stack(xs; dims=2)
    end
    function s4(x, y)
        xs = []
        push!(xs, x)
        push!(xs, y)
        push!(xs, x)
        return stack(xs; dims=2)
    end

    @test @jit(s1(x_ra)) ≈ s1(x)
    @test @jit(s2(x_ra)) ≈ s2(x)
    @test @jit(s3(x_ra, y_ra)) ≈ s3(x, y)
    @test @jit(s4(x_ra, y_ra)) ≈ s4(x, y)

    # Test that we don't hit illegal instruction; `x` is intentionally not a traced array
    @test @jit(s1(x)) isa Any
    @test @jit(s2(x)) isa Any
    @test @jit(s3(x, y)) isa Any
    @test @jit(s4(x, y)) isa Any
end

@testset "duplicate args (#226)" begin
    first_arg(x, y) = x
    x_ra = Reactant.to_rarray(rand(2, 2))
    res = @jit first_arg(x_ra, x_ra)
    @test res ≈ x_ra
end

@testset "Common Trig Functions" begin
    x = rand(Float32, 4, 16)
    x_ra = Reactant.to_rarray(x)

    @testset for fn in (sinpi, cospi, tanpi, sin, cos, tan)
        @test @jit(fn.(x_ra)) ≈ fn.(x)
        @test @jit(fn.(x_ra)) isa ConcreteRArray{Float32,2}
    end

    x = 0.235f0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    @testset for fn in (sinpi, cospi, tanpi, sin, cos, tan)
        @test @jit(fn.(x_ra)) ≈ fn.(x)
        @test @jit(fn.(x_ra)) isa ConcreteRNumber{Float32}
    end
    @testset for fn in (sincospi, sincos)
        res = @jit fn(x_ra)
        @test res[1] ≈ fn(x)[1]
        @test res[2] ≈ fn(x)[2]
        @test res[1] isa ConcreteRNumber{Float32}
        @test res[2] isa ConcreteRNumber{Float32}
    end
end

@testset "isfinite" begin
    x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN])
    @test @jit(isfinite.(x)) == [true, false, false, false, false]

    x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN] .* im)
    @test @jit(isfinite.(x)) == [true, false, false, false, false]
end

@testset "isnan" begin
    x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN])
    @test @jit(isnan.(x)) == [false, true, false, false, true]

    x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN] .* im)
    @test @jit(isnan.(x)) == [false, true, false, false, true]
end

@testset "isnan/isfinite" begin
    @test isnan(Reactant.to_rarray(NaN; track_numbers=Number))
    @test !isnan(Reactant.to_rarray(0.0; track_numbers=Number))
    @test isfinite(Reactant.to_rarray(0.0; track_numbers=Number))
    @test !isfinite(Reactant.to_rarray(Inf; track_numbers=Number))
end

@testset "isinf" begin
    @test Bool(@jit(isinf(ConcreteRNumber(Inf))))
    @test Bool(@jit(isinf(ConcreteRNumber(-Inf))))
    @test !Bool(@jit(isinf(ConcreteRNumber(2))))
    @test !Bool(@jit(isinf(ConcreteRNumber(2.0))))
    @test !Bool(@jit(isinf(ConcreteRNumber(true))))
end

@testset "mod and rem" begin
    a = [-1.1, 7.7, -3.3, 9.9, -5.5]
    b = [6.6, -2.2, -8.8, 4.4, -10.1]

    expected_mod = mod.(a, b)
    @test @jit(mod.(Reactant.to_rarray(a), Reactant.to_rarray(b))) ≈ expected_mod
    @test @jit(mod.(a, Reactant.to_rarray(b))) ≈ expected_mod
    @test @jit(mod.(Reactant.to_rarray(a), b)) ≈ expected_mod

    expected_rem = rem.(a, b)
    @test @jit(rem.(Reactant.to_rarray(a), Reactant.to_rarray(b))) ≈ expected_rem
    @test @jit(rem.(a, Reactant.to_rarray(b))) ≈ expected_rem
    @test @jit(rem.(Reactant.to_rarray(a), b)) ≈ expected_rem
end

@testset "xor" begin
    for a in (true, false), b in (true, false)
        @test @jit(xor(ConcreteRNumber(a), ConcreteRNumber(b))) == xor(a, b)
    end
end

@testset "signbit" begin
    for x in (-4, -3.14, -0.0f0, 0.0, 0, 5, 6.28f0)
        @test @jit(signbit(ConcreteRNumber(x))) == signbit(x)
    end
end

@testset "copysign" begin
    for a in (-3.14, -2, 0.0, 2.71, 42), b in (-7, -0.57, -0.0, 1, 3.14)
        # Make sure also the return type is correct
        @test Reactant.to_number(@jit(copysign(ConcreteRNumber(a), ConcreteRNumber(b)))) ===
            copysign(a, b)
    end
end

@testset "reduce integers" begin
    x = rand(Bool, 100)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra)) == sum(x)

    x = rand(Int16, 100)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra)) == sum(x)
end

@testset "/ on integers" begin
    @test @jit(/(ConcreteRNumber(2), ConcreteRNumber(4))) ≈ 0.5
    @test @jit(/(ConcreteRNumber(2), 4)) ≈ 0.5
    @test @jit(/(2, ConcreteRNumber(4))) ≈ 0.5
    @test @jit(/(2, ConcreteRNumber(Int32(4)))) ≈ 0.5
end

@testset "Broadcasting with Range" begin
    x = Reactant.to_rarray(rand(10))
    fn(x) = x .+ (1:length(x))

    @test @jit(fn(x)) ≈ fn(Array(x))
end

function fntest1(x)
    y = similar(x, 1, 1, 8)
    sum!(y, x)
    return y
end

function fntest2(x)
    y = similar(x, 2, 1, 8)
    sum!(y, x)
    return y
end

function fntest3(x)
    y = similar(x, 2, 1, 1)
    sum!(abs2, y, x)
    return y
end

@testset "mapreducedim!" begin
    x = reshape(collect(Float32, 1:64), 2, 4, 8) ./ 64
    x_ra = Reactant.to_rarray(x)

    @test Array(@jit(fntest1(x_ra))) ≈ fntest1(x)
    @test Array(@jit(fntest2(x_ra))) ≈ fntest2(x)
    @test Array(@jit(fntest3(x_ra))) ≈ fntest3(x)
end

@testset "don't expand ranges by default" begin
    fn(x) = Reactant.TracedUtils.broadcast_to_size(x, (length(x),))

    hlo = repr(@code_hlo(fn(1:10000)))
    @test contains(hlo, "stablehlo.iota")
    @test contains(hlo, "stablehlo.add")
    @test Array(@jit(fn(1:10000))) ≈ collect(1:10000)

    hlo = repr(@code_hlo(fn(32:10000)))
    @test contains(hlo, "stablehlo.iota")
    @test contains(hlo, "stablehlo.add")
    @test Array(@jit(fn(32:10000))) ≈ collect(32:10000)

    hlo = repr(@code_hlo(fn(0:10000)))
    @test contains(hlo, "stablehlo.iota")
    @test !contains(hlo, "stablehlo.add")
    @test Array(@jit(fn(0:10000))) ≈ collect(0:10000)

    hlo = repr(@code_hlo(fn(Base.OneTo(10000))))
    @test contains(hlo, "stablehlo.iota")
    @test contains(hlo, "stablehlo.add")
    @test Array(@jit(fn(Base.OneTo(10000)))) ≈ collect(Base.OneTo(10000))
end

function dip!(x)
    x[:a] = x[:a] .* x[:b]
    return nothing
end

@testset "Dict" begin
    x = Dict{Symbol,Vector{Float32}}()
    x[:a] = 2.7 * ones(4)
    x[:b] = 3.1 * ones(4)

    ra = Reactant.to_rarray(x)
    @jit dip!(ra)
    ra[:a] ≈ (2.7 * 2) * ones(4)
end

@testset "@code_xla" begin
    x_ra = Reactant.to_rarray(ones(4))
    hlo = repr(@code_xla(sin.(x_ra)))
    @test contains(hlo, "HloModule")
    @test contains(hlo, "sine")
end

@testset "Raise keyword" begin
    v = randn(Float32, 16)
    rv = Reactant.to_rarray(v)
    @test sin.(v) ≈ @jit raise = true sin.(rv)
    @test cos.(v) ≈ @jit raise = false cos.(rv)
    @test exp.(v) ≈ @jit raise = "canonicalize" exp.(rv)
    @test_throws Reactant.MLIR.IR.AddPipelineException @jit raise = "this_pass-does_not_ExisT" exp.(
        rv
    )
end

@testset "map!" begin
    x = randn(Float32, 2, 3)
    y = zeros(Float32, 2, 3)

    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test Array(@jit(map!(abs2, y_ra, x_ra))) ≈ map!(abs2, y, x)
    @test Array(y_ra) ≈ y
end

@testset "ConcreteRArray inplace broadcast" begin
    x = Reactant.to_rarray(zeros(Float32, 2, 3))
    y = Reactant.to_rarray(reshape(collect(Float32, 1:6), 2, 3))

    x .= y ./ 2

    @test Array(x) ≈ Array(y) ./ 2

    x = zeros(Float32, 2, 3)
    x .= y ./ 2

    @test Array(x) ≈ Array(y) ./ 2

    x = view(zeros(Float32, 2, 5), :, 1:3)
    x .= y ./ 2

    @test Array(x) ≈ Array(y) ./ 2
end

@testset "Hlo Cost Analysis" begin
    x_ra = Reactant.to_rarray(rand(4, 4))
    mul_comp = @compile x_ra * x_ra
    cost = Reactant.XLA.cost_analysis(mul_comp)

    @test cost isa Reactant.XLA.HloCostAnalysisProperties
end

function fractional_idx(times, t)
    n₂ = searchsortedfirst(times, t)
    n₁ = max(1, n₂ - 1)
    Nt = length(times)
    n₂ = min(Nt, n₂)

    t₁ = times[n₁]
    t₂ = times[n₂]

    ñ = (t - t₁) / (t₂ - t₁)

    return ñ, n₁, n₂
end

@testset "Fractional index" begin
    times = 0:0.01:4.5
    res = @jit fractional_idx(times, ConcreteRNumber(2.143))
    @test res[1] == 0.29999999999997334
    @test res[2] == 215
    @test res[3] == 216
end

mulpi(x) = π * x

@testset "Irrational promotion" begin
    x = Reactant.to_rarray(ones(2))
    y = @jit mulpi(x)
    @test all(Array(y) .≈ π)
end

@testset "copyto! ConcreteArray" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    y_ra = Reactant.to_rarray(zeros(2, 2))
    copyto!(view(x_ra, 1:2, 1:2), y_ra)
    @test Array(x_ra) ==
        [0.0 0.0 1.0 1.0; 0.0 0.0 1.0 1.0; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
end

@testset "copy(::Broadcast.Broadcasted{ArrayStyle{ConcreteRArray}})" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    res = copy(Broadcast.broadcasted(-, Broadcast.broadcasted(+, x_ra, 1)))
    @test res ≈ -(Array(x_ra) .+ 1)
end

@testset "typemin/typemax" begin
    fn(x) = [typemin(eltype(x)), typemax(eltype(x))]

    x_ra = Reactant.to_rarray(ones(4))
    @test @jit(fn(x_ra)) == fn(ones(4))

    x_ra = Reactant.to_rarray(ones(Int, 4))
    @test @jit(fn(x_ra)) == fn(ones(Int, 4))
end
