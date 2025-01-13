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
    sim_c = similar(c)
    @test typeof(sim_c) == typeof(c) && size(sim_c) == size(sim_c)
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
    mean_f1abs2(x) = mean(abs2, x)
    mean_f2abs2(x) = mean(abs2, x; dims=1)

    mean_fn1_compiled = @compile mean_fn1(x_ca)
    mean_fn2_compiled = @compile mean_fn2(x_ca)
    mean_fn3_compiled = @compile mean_fn3(x_ca)
    mean_fn4_compiled = @compile mean_fn4(x_ca)
    mean_f1abs2_compiled = @compile mean_f1abs2(x_ca)
    mean_f2abs2_compiled = @compile mean_f2abs2(x_ca)

    @test mean_fn1(x) ≈ mean_fn1_compiled(x_ca)
    @test mean_fn2(x) ≈ mean_fn2_compiled(x_ca)
    @test mean_fn3(x) ≈ mean_fn3_compiled(x_ca)
    @test mean_fn4(x) ≈ mean_fn4_compiled(x_ca)
    @test mean_f1abs2(x) ≈ mean_f1abs2_compiled(x_ca)
    @test mean_f2abs2(x) ≈ mean_f2abs2_compiled(x_ca)

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

    @testset "Number and RArray" for a in [1.0f0, 1.0e0]
        typeof_a = typeof(a)
        _b = [2.0, 3.0, 4.0] .|> typeof_a
        _c = [2.0 3.0 4.0] .|> typeof_a
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
    fn_inner(x, counts) = repeat(x; inner=counts)

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
            @test (@jit fn_inner(Reactant.to_rarray(x), counts)) == fn_inner(x, counts)
        end
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

function write_with_broadcast1!(x, y)
    x[1, :, :] .= reshape(y, 4, 3)
    return x
end
function write_with_broadcast2!(x, y)
    x[:, 1, :] .= view(y, :, 1:3)
    return x
end

@testset "write_with_broadcast" begin
    x_ra = Reactant.to_rarray(zeros(3, 4, 3))
    y_ra = Reactant.to_rarray(rand(3, 4))

    res = @jit write_with_broadcast1!(x_ra, y_ra)

    @test res.data === x_ra.data

    res = Array(res)
    y = Array(y_ra)
    @test res[1, :, :] ≈ reshape(y, 4, 3)

    x_ra = Reactant.to_rarray(zeros(3, 4, 3))
    y_ra = Reactant.to_rarray(rand(3, 4))

    res = @jit write_with_broadcast2!(x_ra, y_ra)

    @test res.data === x_ra.data

    res = Array(res)
    y = Array(y_ra)
    @test res[:, 1, :] ≈ view(y, :, 1:3)
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

function non_contiguous_setindex!(x)
    x[[1, 3, 2], [1, 2, 3, 4]] .= 1.0
    return x
end

@testset "non-contiguous setindex!" begin
    x = rand(6, 6)
    x_ra = Reactant.to_rarray(x)

    y = @jit(non_contiguous_setindex!(x_ra))
    y = Array(y)
    x_ra = Array(x_ra)
    @test all(isone, y[1:3, 1:4])
    @test all(isone, x_ra[1:3, 1:4])
    @test !all(isone, y[4:end, :])
    @test !all(isone, x_ra[4:end, :])
    @test !all(isone, y[:, 5:end])
    @test !all(isone, x_ra[:, 5:end])
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

@testset "$op" for op in [:round, :ceil, :floor]
    for x in (rand(Float32, (3, 3)), rand(Float64))
        @eval @test @jit($op.(ConcreteRNumber.($x))) == $op.($x)
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
    @test 1.0 ==
        @jit ifelse(ConcreteRNumber(true), ConcreteRNumber(1.0), ConcreteRNumber(0.0f0))
    @test @jit(
        ifelse(ConcreteRNumber(false), ConcreteRNumber(1.0), ConcreteRNumber(0.0f0))
    ) isa ConcreteRNumber{Float64}
    @test 0.0f0 ==
        @jit ifelse(ConcreteRNumber(false), ConcreteRNumber(1.0), ConcreteRNumber(0.0f0))
    @test @jit(
        ifelse(ConcreteRNumber(false), ConcreteRNumber(1.0f0), ConcreteRNumber(0.0f0))
    ) isa ConcreteRNumber{Float32}
end

@testset "fill! and zero on ConcreteRArray" begin
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
        Ptr{Float64}, Reactant.XLA.UnsafeBufferPointer(x.data.buffer)
    )
    ptr_res = Base.unsafe_convert(
        Ptr{Float64}, Reactant.XLA.UnsafeBufferPointer(res.data.buffer)
    )
    ptr_T1 = Base.unsafe_convert(
        Ptr{Float64}, Reactant.XLA.UnsafeBufferPointer(T[1].data.buffer)
    )

    @test ptr_x == ptr_res == ptr_T1
end

@testset "eltype conversion inside interpreter" begin
    function test_convert(x::AbstractArray{T}, eta) where {T}
        eta = T(eta)
        return x .* eta, eta
    end

    res = @jit test_convert(ConcreteRArray(rand(4, 2)), ConcreteRNumber(3.0f0))

    @test res[1] isa ConcreteRArray{Float64,2}
    @test res[2] isa ConcreteRNumber{Float64}
end

@testset "non-contiguous indexing" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing1(x) = x[[1, 3, 2], :, :]
    non_contiguous_indexing2(x) = x[:, [1, 2, 1, 3], [1, 3]]

    @test @jit(non_contiguous_indexing1(x_ra)) ≈ non_contiguous_indexing1(x)
    @test @jit(non_contiguous_indexing2(x_ra)) ≈ non_contiguous_indexing2(x)

    x = rand(4, 2)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing3(x) = x[[1, 3, 2], :]
    non_contiguous_indexing4(x) = x[:, [1, 2, 2]]

    @test @jit(non_contiguous_indexing3(x_ra)) ≈ non_contiguous_indexing3(x)
    @test @jit(non_contiguous_indexing4(x_ra)) ≈ non_contiguous_indexing4(x)

    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing1!(x) = x[[1, 3, 2], :, :] .= 2
    non_contiguous_indexing2!(x) = x[:, [1, 2, 1, 3], [1, 3]] .= 2

    @jit(non_contiguous_indexing1!(x_ra))
    non_contiguous_indexing1!(x)
    @test x_ra ≈ x

    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    @jit(non_contiguous_indexing2!(x_ra))
    non_contiguous_indexing2!(x)
    @test x_ra ≈ x

    x = rand(4, 2)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing3!(x) = x[[1, 3, 2], :] .= 2
    non_contiguous_indexing4!(x) = x[:, [1, 2, 2]] .= 2

    @jit(non_contiguous_indexing3!(x_ra))
    non_contiguous_indexing3!(x)
    @test x_ra ≈ x

    x = rand(4, 2)
    x_ra = Reactant.to_rarray(x)

    @jit(non_contiguous_indexing4!(x_ra))
    non_contiguous_indexing4!(x)
    @test x_ra ≈ x
end

@testset "indexing with traced arrays" begin
    x = rand(4, 4, 3)
    idx1 = [1, 3, 2]
    idx3 = [1, 2, 1, 3]

    x_ra = Reactant.to_rarray(x)
    idx1_ra = Reactant.to_rarray(idx1)
    idx3_ra = Reactant.to_rarray(idx3)

    getindex1(x, idx1) = x[idx1, :, :]
    getindex2(x, idx1) = x[:, idx1, :]
    getindex3(x, idx3) = x[:, :, idx3]
    getindex4(x, idx1, idx3) = x[idx1, :, idx3]

    @test @jit(getindex1(x_ra, idx1_ra)) ≈ getindex1(x, idx1)
    @test @jit(getindex2(x_ra, idx1_ra)) ≈ getindex2(x, idx1)
    @test @jit(getindex3(x_ra, idx3_ra)) ≈ getindex3(x, idx3)
    @test @jit(getindex4(x_ra, idx1_ra, idx3_ra)) ≈ getindex4(x, idx1, idx3)
end

@testset "linear indexing" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    getindex_linear_scalar(x, idx) = @allowscalar x[idx]

    @testset for i in 1:length(x)
        @test @jit(getindex_linear_scalar(x_ra, i)) ≈ getindex_linear_scalar(x, i)
        @test @jit(
            getindex_linear_scalar(x_ra, Reactant.to_rarray(i; track_numbers=(Number,)))
        ) ≈ getindex_linear_scalar(x, i)
    end

    idx = rand(1:length(x), 8)
    idx_ra = Reactant.to_rarray(idx)

    getindex_linear_vector(x, idx) = x[idx]

    @test @jit(getindex_linear_vector(x_ra, idx_ra)) ≈ getindex_linear_vector(x, idx)
    @test @jit(getindex_linear_vector(x_ra, idx)) ≈ getindex_linear_vector(x, idx)
end

@testset "stack" begin
    x = rand(4, 4)
    y = rand(4, 4)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    s1(x) = stack((x, x))
    s2(x) = stack((x, x); dims=2)
    s3(x, y) = stack((x, y); dims=2)
    s4(x, y) = stack((x, y, x); dims=1)

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

@testset "Boolean Indexing" begin
    x_ra = Reactant.to_rarray(rand(Float32, 4, 16))
    idxs_ra = Reactant.to_rarray(rand(Bool, 16))

    fn(x, idxs) = x[:, idxs]

    @test_throws ErrorException @jit(fn(x_ra, idxs_ra))

    res = @jit fn(x_ra, Array(idxs_ra))
    @test res ≈ fn(Array(x_ra), Array(idxs_ra))
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
    x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

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
