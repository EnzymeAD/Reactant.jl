using Reactant
using Test
using Enzyme
using Statistics

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))

using InteractiveUtils

@testset "2D sum" begin
    x = rand(2, 10)

    r_res = sum(x)

    a = Reactant.ConcreteRArray(x)

    c_res = @allowscalar sum(a)
    @test c_res ≈ r_res

    @test @jit(sum(a)) ≈ r_res
end

@testset "Basic reduce max" begin
    x = rand(2, 10)

    r_res = fastmax(x)

    a = Reactant.ConcreteRArray(x)

    c_res = @allowscalar fastmax(a)
    @test c_res ≈ r_res

    @test @jit(fastmax(a)) ≈ r_res
end

sinexp(x) = sin(exp(x))
sinexpbc(x) = sinexp.(x)

@testset "Broadcast combined" begin
    x = rand(2, 10)

    r_res = sinexpbc(x)

    a = Reactant.ConcreteRArray(x)

    c_res = @allowscalar sinexpbc(a)
    @test c_res ≈ r_res

    @test @jit(sinexpbc(a)) ≈ r_res
end

sumexp(x) = sum(exp, x)

sum_compare(x) = sum(x) > 0

@testset "Basic mapreduce" begin
    x = rand(Float32, 10)
    a = Reactant.ConcreteRArray(x)
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

    a = Reactant.ConcreteRArray(x)

    f_res = @jit mysoftmax!(a)
    @test f_res ≈ r_res
end

bcast_cos(x) = cos.(x)

@testset "Basic cos" begin
    x = rand(3, 2)
    c = Reactant.ConcreteRArray(x)

    @test @jit(bcast_cos(c)) ≈ cos.(x)
end

f_var(args...) = sum(args)

@testset "Vararg" begin
    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(3 * ones(3))
    z = Reactant.to_rarray(2.6 * ones(3))

    @test @jit(f_var(x, y, z)) ≈ [6.6, 6.6, 6.6]
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

    @test @jit(grad_ip(c)) ≈ -sin.(ones(3, 2))

    orig, r = @jit(resgrad_ip(c))

    @test orig[2] ≈ sum(cos.(ones(3, 2)))
    @test r ≈ -sin.(ones(3, 2))
end

function mul(A, B)
    return A * B
end
@testset "matmul" begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    d = Reactant.ConcreteRArray(ones(70, 30))

    @test @jit(mul(c, d)) ≈ mul(ones(50, 70), ones(70, 30))
end

@testset "ConcreteRArray" begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    similar(c)
end

@testset "Reactant.@code_hlo" begin
    W = Reactant.ConcreteRArray(randn(Float32, 10, 20))
    x = Reactant.ConcreteRArray(randn(Float32, 20, 5))
    res = Reactant.@code_hlo W * x
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.dot_general")
end

@testset "Reactant.@code_hlo broadcasting" begin
    x = Reactant.ConcreteRArray(randn(Float32, 2, 2))
    y = Reactant.ConcreteRArray(randn(Float32, 2, 2))
    res = Reactant.@code_hlo (.+)(x, y)
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.add")
end

@testset "Statistics: `mean` & `var`" begin
    x = randn(2, 3, 4)
    x_ca = Reactant.ConcreteRArray(x)

    # XXX: @jit doesn't work with `;`
    # @test @jit(mean(x_ca)) ≈ mean(x)
    # @test @jit(mean(x_ca; dims=1)) ≈ mean(x; dims=1)
    # @test @jit(mean(x_ca; dims=(1, 2))) ≈ mean(x; dims=(1, 2))
    # @test @jit(mean(x_ca; dims=(1, 3))) ≈ mean(x; dims=(1, 3))

    mean_fn1(x) = mean(x)
    mean_fn2(x) = mean(x; dims=1)
    mean_fn3(x) = mean(x; dims=(1, 2))
    mean_fn4(x) = mean(x; dims=(1, 3))

    mean_fn1_compiled = @compile mean_fn1(x_ca)
    mean_fn2_compiled = @compile mean_fn2(x_ca)
    mean_fn3_compiled = @compile mean_fn3(x_ca)
    mean_fn4_compiled = @compile mean_fn4(x_ca)

    @test mean_fn1(x) ≈ mean_fn1_compiled(x_ca)
    @test mean_fn2(x) ≈ mean_fn2_compiled(x_ca)
    @test mean_fn3(x) ≈ mean_fn3_compiled(x_ca)
    @test mean_fn4(x) ≈ mean_fn4_compiled(x_ca)

    # XXX: @jit doesn't work with `;`
    # @test @jit(var(x_ca)) ≈ var(x)
    # @test @jit(var(x_ca; dims=1)) ≈ var(x; dims=1)
    # @test @jit(var(x_ca; dims=(1, 2), corrected=false)) ≈
    #     var(x; dims=(1, 2), corrected=false)
    # @test @jit(var(x_ca; dims=(1, 3), corrected=false)) ≈
    #     var(x; dims=(1, 3), corrected=false)

    var_fn1(x) = var(x)
    var_fn2(x) = var(x; dims=1)
    var_fn3(x) = var(x; dims=(1, 2), corrected=false)
    var_fn4(x) = var(x; dims=(1, 3), corrected=false)

    var_fn1_compiled = @compile var_fn1(x_ca)
    var_fn2_compiled = @compile var_fn2(x_ca)
    var_fn3_compiled = @compile var_fn3(x_ca)
    var_fn4_compiled = @compile var_fn4(x_ca)

    @test var_fn1(x) ≈ var_fn1_compiled(x_ca)
    @test var_fn2(x) ≈ var_fn2_compiled(x_ca)
    @test var_fn3(x) ≈ var_fn3_compiled(x_ca)
    @test var_fn4(x) ≈ var_fn4_compiled(x_ca)
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
end

function update_on_copy(x)
    y = x[1:2, 2:4, :]
    y[1:1, 1:1, :] = ones(1, 1, 3)
    return y
end

@testset "view / setindex" begin
    x = rand(2, 4, 3)
    y = copy(x)
    x_concrete = Reactant.to_rarray(x)
    y_concrete = Reactant.to_rarray(y)

    y1 = update_on_copy(x)
    y2 = @jit update_on_copy(x_concrete)
    @test x == y
    @test x_concrete == y_concrete
    @test y1 == y2

    # function update_inplace(x)
    #     y = view(x, 1:2, 1:2, :)
    #     y[1, 1, :] .= 1
    #     return y
    # end

    # get_indices(x) = x[1:2, 1:2, :]
    # get_view(x) = view(x, 1:2, 1:2, :)

    # get_indices_compiled = @compile get_indices(x_concrete)
    # get_view_compiled = @compile get_view(x_concrete)
end

function masking(x)
    y = similar(x)
    y[1:2, :] .= 0
    y[3:4, :] .= 1
    return y
end

function masking!(x)
    x[1:2, :] .= 0
    x[3:4, :] .= 1
    return x
end

@testset "setindex! with views" begin
    x = rand(4, 4) .+ 2.0
    x_ra = Reactant.to_rarray(x)

    y = masking(x)
    y_ra = @jit(masking(x_ra))
    @test y ≈ y_ra

    x_ra_array = Array(x_ra)
    @test !(any(iszero, x_ra_array[1, :]))
    @test !(any(iszero, x_ra_array[2, :]))
    @test !(any(isone, x_ra_array[3, :]))
    @test !(any(isone, x_ra_array[4, :]))

    y_ra = @jit(masking!(x_ra))
    @test y ≈ y_ra

    x_ra_array = Array(x_ra)
    @test @allowscalar all(iszero, x_ra_array[1, :])
    @test @allowscalar all(iszero, x_ra_array[2, :])
    @test @allowscalar all(isone, x_ra_array[3, :])
    @test @allowscalar all(isone, x_ra_array[4, :])
end

tuple_byref(x) = (; a=(; b=x))
tuple_byref2(x) = abs2.(x), tuple_byref2(x)

@testset "Tuple byref" begin
    x = Reactant.to_rarray([1.0 -2.0; -3.0 4.0])
    @test @jit(tuple_byref(x)).a.b.data === x.data

    # TODO this seems to hang during compile
    # f2 = @compile tuple_byref2(x)
    # r2 = f2(x)
    # @test r2[2].a.b.data === x.data
    # @test r2[1] == abs2.([1.0 -2.0; -3.0 4.0])
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

        x_ra = Reactant.to_rarray(x; track_numbers=(Number,))
        f2 = @compile f1(x_ra)
        @test f2(Reactant.to_rarray((5, 5.2); track_numbers=(Number,))) ≈ 5 * 5.2
        @test f2(Reactant.to_rarray((5, 5.2); track_numbers=(Number,))) isa ConcreteRNumber

        x_ra = Reactant.to_rarray(x)
        f3 = @compile f1(x_ra)
        @test f3(Reactant.to_rarray((5, 5.2))) ≈ f1(x)
        @test !(f3(Reactant.to_rarray((5, 5.2))) isa ConcreteRNumber)
        @test f3(Reactant.to_rarray((5, 5.2))) isa Number

        x_ra = Reactant.to_rarray(x; track_numbers=(Int,))
        f4 = @compile f1(x_ra)
        @test f4(Reactant.to_rarray((5, 5.2); track_numbers=(Int,))) ≈ 5 * 3.14
        @test f4(Reactant.to_rarray((5, 5.2); track_numbers=(Int,))) isa ConcreteRNumber
    end

    @testset "Mixed" begin
        x = (3, [3.14])

        f1(x) = x[1] * x[2]

        x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

        f2 = @compile f1(x_ra)
        res2 = f2(Reactant.to_rarray((5, [3.14]); track_numbers=(Number,)))
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
    x = ConcreteRNumber(3.14)
    @test Float32(x) isa Float32
    @test Float64(x) isa Float64
    @test_throws InexactError Int(x)

    x = ConcreteRNumber(3)
    @test Float32(x) isa Float32
    @test Float64(x) isa Float64
    @test Int(x) isa Int
    @test float(x) isa ConcreteRNumber{Float64}
end

@testset "concrete number with fill" begin
    x = ConcreteRNumber(10)
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

@testset "dynamic indexing" begin
    x = randn(5, 3)
    x_ra = Reactant.to_rarray(x)

    idx = [1, 2, 3]
    idx_ra = Reactant.to_rarray(idx)

    fn(x, idx) = @allowscalar x[idx, :]

    y = @jit(fn(x_ra, idx_ra))
    @test y ≈ x[idx, :]
end

@testset "aos_to_soa" begin
    using ArrayInterface

    x_res = collect(reshape(1.0:4.0, 2, 1, 2))
    x_ca = ConcreteRNumber.(x_res)

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

    @testset "ConcreteRArray" begin
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
        @test y isa ConcreteRArray{Int,0}
        @test y == x
    end

    @testset "TracedRArray" begin
        y = @jit(collect(x_ra))
        @test y isa ConcreteRArray{Int,0}
        @test y == x
    end
end

function f_row_major(x)
    y = [1 2; 3 4; 5 6]
    if x isa Reactant.TracedRArray
        y = Reactant.promote_to(Reactant.TracedRArray{eltype(x),2}, y)
    end
    return x .+ y
end

@testset "array attributes: row major" begin
    x = zeros(Int, 3, 2)
    x_ra = Reactant.to_rarray(x)

    @test @jit(f_row_major(x_ra)) ≈ f_row_major(x)
end

@testset "PermutedDimsArray" begin
    x = randn(2, 3)
    x_re = Reactant.to_rarray(x)

    f(u) = PermutedDimsArray(u, (2, 1))
    @test f(x) == @jit f(x_re)
end
