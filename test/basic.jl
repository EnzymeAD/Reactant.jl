using Reactant, Test, Enzyme, Statistics, InteractiveUtils

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))

@testset "2D sum" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)

    r_res = sum(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar sum(a)
    @test c_res ≈ r_res

    @test @jit(sum(a)) ≈ r_res
end

@testset "Julia Compilation cache" begin
    x = @compile -(Reactant.to_rarray(ones(2)))
    y = @compile -(Reactant.to_rarray(ones(2)))

    @test typeof(x) == typeof(y)
    # TODO, currently x and y are not equal as x.exec != y.exec
    # as the executable we generate is itself not cached
    # (which clearly we should do to improve jit time)
end

@testset "Basic reduce max" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)

    r_res = fastmax(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar fastmax(a)
    @test c_res ≈ r_res

    @test @jit(fastmax(a)) ≈ r_res
end

sinexp(x) = sin(exp(x))
sinexpbc(x) = sinexp.(x)

@testset "Broadcast combined" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)

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
    x = Reactant.TestUtils.construct_test_array(Float32, 10)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)
    r_res = mysoftmax!(x)

    a = Reactant.to_rarray(x)

    f_res = @jit mysoftmax!(a)
    @test f_res ≈ r_res
end

bcast_cos(x) = cos.(x)

@testset "Basic cos" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 3, 2)
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
    W = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 10, 20))
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 20, 5))
    res = @code_hlo W * x
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.dot_general")
end

@testset "@code_hlo broadcasting" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 2))
    y = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 2))
    res = @code_hlo (.+)(x, y)
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.add")
end

@testset "Statistics: `mean` & `var`" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 4)
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
        x = Reactant.TestUtils.construct_test_array(Float64, size...)

        @testset "outer repeat" begin
            @test (@jit repeat(Reactant.to_rarray(x), counts...)) ≈ repeat(x, counts...)
        end

        length(counts) < length(size) && continue

        @testset "inner repeat" begin
            @test (@jit repeat(Reactant.to_rarray(x); inner=counts)) ≈
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
    @testset "size(x): $(size(x))" for x in (
        Reactant.TestUtils.construct_test_array(Float64, 4, 4),
        Reactant.TestUtils.construct_test_array(Float64, 4),
    )
        x_ca = Reactant.to_rarray(x)

        @test @jit(sum_xxᵀ(x_ca)) ≈ sum_xxᵀ(x)
    end
end

function similar_from_type(x)
    sim_x = similar(typeof(x), (4, 5))
    return sim_x
end

@testset "similar" begin
    x = zeros(2, 3)
    y = Reactant.to_rarray(x)
    f = @compile similar(y)
    @test size(f(y)) == size(x)
    @test eltype(f(y)) == eltype(x)

    f_from_type = @compile similar_from_type(y)
    @test size(f_from_type(y)) == (4, 5)
    @test eltype(f_from_type(y)) == eltype(x)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    x_ra = Reactant.to_rarray(x)

    y = @jit(clamp!(x_ra, 0.0, 0.25))
    @allowscalar begin
        @test maximum(y) ≤ 0.25
        @test minimum(y) ≥ 0.0
        @test maximum(x_ra) == maximum(y)
        @test minimum(x_ra) == minimum(y)
    end

    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    x_ra = Reactant.to_rarray(x)

    y = @jit(clamp.(x_ra, 0.0, 0.25))
    @allowscalar begin
        @test maximum(y) ≤ 0.25
        @test minimum(y) ≥ 0.0
        @test x_ra ≈ x
    end

    x_ra = ConcreteRNumber(3.0)
    y = @jit(clamp(x_ra, 0.0, 0.25))
    @test y isa ConcreteRNumber{Float64}
end

@testset for op in [round, ceil, floor]
    @testset "$(typeof(x)) : $(size(x))" for x in (
        Reactant.TestUtils.construct_test_array(Float32, 3, 3),
        Reactant.TestUtils.construct_test_array(Float64, 1),
    )
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
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    x_ra = Reactant.to_rarray(x)

    @testset "Reactant.to_rarray" begin
        y = collect(x_ra)
        @test y ≈ x
        @test y !== x_ra
    end

    @testset "TracedRArray" begin
        y = @jit(collect(x_ra))
        @test y ≈ x
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
        y = Reactant.promote_to(Reactant.TracedRArray{Reactant.unwrapped_eltype(T),2}, y)
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
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3, 4))

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
        Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 4, 2)),
        Reactant.to_rarray(3.0f0; track_numbers=Number),
    )

    @test res[1] isa ConcreteRArray{Float64,2}
    @test res[2] isa ConcreteRNumber{Float64}
end

@testset "stack" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    y = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    y = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
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
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 2, 2))
    res = @jit first_arg(x_ra, x_ra)
    @test res ≈ x_ra
end

@testset "Common Trig Functions" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 4, 16)[:, 1:7]
    x_ra = Reactant.to_rarray(x)

    @testset for fn in (sinpi, cospi, tanpi, sin, cos, tan)
        @test @jit(fn.(x_ra)) ≈ fn.(x)
        @test @jit(fn.(x_ra)) isa ConcreteRArray{Float32,2}
    end

    x2 = inv.(x)
    x2_ra = Reactant.to_rarray(x2)

    @testset for fn in (acscd, asecd)
        @test @jit(fn.(x2_ra)) ≈ fn.(x2)
        @test @jit(fn.(x2_ra)) isa ConcreteRArray{Float32,2}
    end

    xrad = deg2rad.(x)
    xrad_ra = Reactant.to_rarray(xrad)

    @testset for fn in (sind, cosd, tand, cscd, secd, cotd, asind, acosd, atand, acotd)
        @test @jit(fn.(xrad_ra)) ≈ fn.(xrad)
        @test @jit(fn.(xrad_ra)) isa ConcreteRArray{Float32,2}
    end

    yrad = Reactant.TestUtils.construct_test_array(Float32, 4, 16)[:, 3:9]
    yrad_ra = Reactant.to_rarray(yrad)

    @testset for fn in (atan, atand)
        @test @jit(fn.(yrad_ra, xrad_ra)) ≈ fn.(yrad, xrad)
        @test @jit(fn.(yrad_ra, xrad_ra)) isa ConcreteRArray{Float32,2}
    end

    x = 0.235f0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    @testset for fn in (sinpi, cospi, tanpi, sin, cos, tan, asind, acosd, atand, acotd)
        @test @jit(fn.(x_ra)) ≈ fn.(x)
        @test @jit(fn.(x_ra)) isa ConcreteRNumber{Float32}
    end

    x2 = inv(x)
    x2_ra = Reactant.to_rarray(x2; track_numbers=Number)

    @testset for fn in (acscd, asecd)
        @test @jit(fn.(x2_ra)) ≈ fn.(x2)
        @test @jit(fn.(x2_ra)) isa ConcreteRNumber{Float32}
    end

    xrad = deg2rad(x)
    xrad_ra = Reactant.to_rarray(xrad; track_numbers=Number)

    @testset for fn in (sind, cosd, tand, cscd, secd, cotd)
        @test @jit(fn.(xrad_ra)) ≈ fn.(xrad)
        @test @jit(fn.(xrad_ra)) isa ConcreteRNumber{Float32}
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

    @test begin
        x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN] .* im)
        @jit(isfinite.(x)) == [true, false, false, false, false]
    end skip = RunningOnTPU
end

@testset "isnan" begin
    x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN])
    @test @jit(isnan.(x)) == [false, true, false, false, true]

    @test begin
        x = Reactant.to_rarray([1.0, NaN, Inf, -Inf, NaN] .* im)
        @jit(isnan.(x)) == [false, true, false, false, true]
    end skip = RunningOnTPU
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
    @test @jit(mod.(Reactant.to_rarray(a), Reactant.to_rarray(b))) ≈ expected_mod broken =
        RunningOnTPU
    @test @jit(mod.(a, Reactant.to_rarray(b))) ≈ expected_mod broken = RunningOnTPU
    @test @jit(mod.(Reactant.to_rarray(a), b)) ≈ expected_mod broken = RunningOnTPU

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
    @testset "$(typeof(x))" for x in (-4, -3.14, -0.0f0, 0.0, 0, 5, 6.28f0)
        @test @jit(signbit(ConcreteRNumber(x))) == signbit(x) broken =
            RunningOnTPU && eltype(x) == Float64
    end
end

@testset "copysign" begin
    @testset "$(typeof(a)) $(typeof(b))" for a in (-3.14, -2, 0.0, 2.71, 42),
        b in (-7, -0.57, -0.0, 1, 3.14)
        # Make sure also the return type is correct
        @test Reactant.to_number(@jit(copysign(ConcreteRNumber(a), ConcreteRNumber(b)))) ≈
            copysign(a, b) broken = RunningOnTPU && eltype(b) == Float64
    end
end

@testset "reduce integers" begin
    x = [isodd(i) for i in 1:100]
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra)) == sum(x)

    x = collect(Int16, 1:100)
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
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 10))
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
    fn(x) = Reactant.broadcast_to_size(x, (length(x),))

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
    @test ra[:a] ≈ (2.7 * 3.1) * ones(4)
end

@testset "@code_xla" begin
    x_ra = Reactant.to_rarray(ones(Float32, 4))
    hlo = repr(@code_xla(sin.(x_ra)))
    @test contains(hlo, "HloModule")
    @test contains(hlo, "sine")
end

@testset "Raise keyword" begin
    v = Reactant.TestUtils.construct_test_array(Float32, 16)
    rv = Reactant.to_rarray(v)
    @test sin.(v) ≈ @jit raise = true sin.(rv)
    @test cos.(v) ≈ @jit raise = false cos.(rv)
    @test exp.(v) ≈ @jit raise = "canonicalize" exp.(rv)
    @test_throws Reactant.MLIR.IR.AddPipelineException @jit raise = "this_pass-does_not_ExisT" exp.(
        rv
    )
end

@testset "map!" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3)
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

@testset "HLO Cost Analysis" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 4, 4))
    mul_comp = @compile x_ra * x_ra
    @test begin
        Reactant.XLA.cost_analysis(mul_comp) isa Reactant.XLA.HloCostAnalysisProperties
    end broken = RunningOnTPU
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
    @test times isa Base.StepRangeLen
    res = @jit fractional_idx(times, ConcreteRNumber(2.143))
    @test res[1] ≈ 0.29999999999997334
    @test res[2] == 215
    @test res[3] == 216
end

@testset "Traced fractional index" begin
    times = Reactant.to_rarray(0:0.01:4.5; track_numbers=Number)
    @test times isa Reactant.TracedStepRangeLen
    res = @jit fractional_idx(times, ConcreteRNumber(2.143))
    @test res[1] ≈ 0.29999999999997334
    @test res[2] == 215
    @test res[3] == 216
end

@testset "Unitrange" begin
    x = 2:10
    @test (@jit getindex(x, 3)) == 4
    @test (@jit getindex(x, Reactant.ConcreteRNumber(4))) == 5

    x = Reactant.to_rarray(2:10; track_numbers=Number)
    @test (@jit getindex(x, 3)) == 4
    @test (@jit getindex(x, Reactant.ConcreteRNumber(4))) == 5
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

@testset "copyto! ConcreteArray Array" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    y_ra = view(zeros(4, 4), 1:2, 1:2)
    copyto!(view(x_ra, 1:2, 1:2), y_ra)
    @test Array(x_ra) ==
        [0.0 0.0 1.0 1.0; 0.0 0.0 1.0 1.0; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
end

@testset "copyto! TracedRArray" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    y_ra = Reactant.to_rarray(zeros(2, 2))
    @jit copyto!(x_ra, 6, y_ra, 3, 2)

    x = ones(4, 4)
    y = zeros(2, 2)
    copyto!(x, 6, y, 3, 2)
    @test Array(x_ra) == x
end

function reshapecopy!(x, y)
    Base.copyto!(x, reshape(y, size(x)))
    return nothing
end
@testset "copyto! Reshaped TracedRArray" begin
    x = zeros(3, 4, 5)
    y = collect(reshape(1:60, (3, 20)))

    xr = Reactant.to_rarray(x)
    yr = Reactant.to_rarray(y)

    @jit reshapecopy!(xr, yr)

    reshapecopy!(x, y)
    @test Array(xr) == x
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

@testset "Module printing" begin
    for opt in (true, false, :before_jit), debug in (true, false)
        v = collect(Float32(1):Float32(64))
        vr = Reactant.to_rarray(v)
        mod = @code_hlo optimize = opt log.(vr)

        # Store the module as a string with different debug options.
        io = IOBuffer()
        show(IOContext(io, :debug => debug), mod)
        mod_string = String(take!(io))

        # Test that we can parse back the string as an MLIR module, compile it
        # and get correct results.
        res = @jit(Reactant.Ops.hlo_call(mod_string, vr))[1]
        @test res ≈ log.(v)
    end
end

@testset "Dump MLIR modules" begin
    always_old = Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[]
    dir_old = Reactant.MLIR.IR.DUMP_MLIR_DIR[]

    mktempdir() do dir
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir
        @compile sin.(Reactant.to_rarray(Float32[1.0]))
        for mod in readdir(dir; join=true)
            @test contains(read(mod, String), "hlo.sine")
        end
    end

    mktempdir() do dir
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = false
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir
        @compile exp.(Reactant.to_rarray(Float32[1.0]))
        # Make sure we don't save anything to file when compilation is
        # successful and `DUMP_MLIR_ALWAYS=false`.
        @test isempty(readdir(dir; join=true))
    end

    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = always_old
    Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir_old
end

@testset "Allocator Stats" begin
    platform_name = lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    if platform_name != "cpu" # not supported on CPU
        @test Reactant.XLA.allocatorstats() isa Reactant.XLA.AllocatorStats
    else
        @test_throws Reactant.XLA.ReactantInternalError Reactant.XLA.allocatorstats()
    end
end

@testset "copy/deepcopy" begin
    for op in (copy, deepcopy)
        x = Reactant.to_rarray(ones(4, 4))
        if x isa Reactant.ConcretePJRTArray
            orig_ptr = only(x.data).buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcretePJRTArray
            @test only(y.data).buffer.buffer != orig_ptr
            @test only(x.data).buffer.buffer == orig_ptr
        else
            orig_ptr = x.data.buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcreteIFRTArray
            @test y.data.buffer.buffer != orig_ptr
            @test x.data.buffer.buffer == orig_ptr
        end

        x = Reactant.to_rarray(4.0; track_numbers=Number)
        if x isa Reactant.ConcretePJRTNumber
            orig_ptr = only(x.data).buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcretePJRTNumber
            @test only(y.data).buffer.buffer != orig_ptr
            @test only(x.data).buffer.buffer == orig_ptr
        else
            orig_ptr = x.data.buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcreteIFRTNumber
            @test y.data.buffer.buffer != orig_ptr
            @test x.data.buffer.buffer == orig_ptr
        end
    end
end

function test_aliased_numbers(ps, x)
    return map(Returns(x), ps)
end

@testset "Correct Aliasing" begin
    ps = Reactant.to_rarray((
        a=Reactant.TestUtils.construct_test_array(Float64, 4),
        b=Reactant.TestUtils.construct_test_array(Float64, 2),
        c=Reactant.TestUtils.construct_test_array(Float64, 4),
    ))
    x = ConcreteRNumber(3.14)
    res = @jit test_aliased_numbers(ps, x)

    @test res[1] === res[2] === res[3]
end

accum_fn(x, y) = abs2(x) + abs2(y)

@testset "accumulate" begin
    a = collect(Float32, 1:10) ./ 10
    a_ra = Reactant.to_rarray(a)

    b = reshape(collect(Float32, 1:60), (3, 4, 5)) ./ 60
    b_ra = Reactant.to_rarray(b)

    @testset "cumsum" begin
        @test @jit(cumsum(a_ra)) ≈ cumsum(a)

        @test @jit(cumsum(b_ra; dims=1)) ≈ cumsum(b; dims=1)
        @test @jit(cumsum(b_ra; dims=2)) ≈ cumsum(b; dims=2)
        @test @jit(cumsum(b_ra; dims=3)) ≈ cumsum(b; dims=3)

        @test begin
            z = similar(a_ra)
            @jit(cumsum!(z, a_ra))
            z
        end ≈ cumsum(a)

        @test begin
            z = similar(b_ra)
            @jit(cumsum!(z, b_ra; dims=1))
            z
        end ≈ cumsum(b; dims=1)
        @test begin
            z = similar(b_ra)
            @jit(cumsum!(z, b_ra; dims=2))
            z
        end ≈ cumsum(b; dims=2)
        @test begin
            z = similar(b_ra)
            @jit(cumsum!(z, b_ra; dims=3))
            z
        end ≈ cumsum(b; dims=3)
    end

    @testset "cumprod" begin
        @test @jit(cumprod(a_ra)) ≈ cumprod(a)

        @test @jit(cumprod(b_ra; dims=1)) ≈ cumprod(b; dims=1)
        @test @jit(cumprod(b_ra; dims=2)) ≈ cumprod(b; dims=2)
        @test @jit(cumprod(b_ra; dims=3)) ≈ cumprod(b; dims=3)

        @test begin
            z = similar(a_ra)
            @jit(cumprod!(z, a_ra))
            z
        end ≈ cumprod(a)
        @test begin
            z = similar(b_ra)
            @jit(cumprod!(z, b_ra; dims=1))
            z
        end ≈ cumprod(b; dims=1)
        @test begin
            z = similar(b_ra)
            @jit(cumprod!(z, b_ra; dims=2))
            z
        end ≈ cumprod(b; dims=2)
        @test begin
            z = similar(b_ra)
            @jit(cumprod!(z, b_ra; dims=3))
            z
        end ≈ cumprod(b; dims=3)
    end

    @testset "accumulate" begin
        @test @jit(accumulate(accum_fn, a_ra; init=0.0f0)) ≈
            accumulate(accum_fn, a; init=0.0f0) broken = RunningOnTPU

        @test @jit(accumulate(accum_fn, b_ra; init=0.0f0, dims=1)) ≈
            accumulate(accum_fn, b; dims=1, init=0.0f0) broken = RunningOnTPU
        @test @jit(accumulate(accum_fn, b_ra; init=0.0f0, dims=2)) ≈
            accumulate(accum_fn, b; dims=2, init=0.0f0) broken = RunningOnTPU
        @test @jit(accumulate(accum_fn, b_ra; init=0.0f0, dims=3)) ≈
            accumulate(accum_fn, b; dims=3, init=0.0f0) broken = RunningOnTPU

        @test begin
            z = similar(a_ra)
            @jit(accumulate!(accum_fn, z, a_ra; init=0.0f0))
            z
        end ≈ accumulate(accum_fn, a; init=0.0f0) broken = RunningOnTPU

        @test begin
            z = similar(b_ra)
            @jit(accumulate!(accum_fn, z, b_ra; init=0.0f0, dims=1))
            z
        end ≈ accumulate(accum_fn, b; dims=1, init=0.0f0) broken = RunningOnTPU
        @test begin
            z = similar(b_ra)
            @jit(accumulate!(accum_fn, z, b_ra; init=0.0f0, dims=2))
            z
        end ≈ accumulate(accum_fn, b; dims=2, init=0.0f0) broken = RunningOnTPU
        @test begin
            z = similar(b_ra)
            @jit(accumulate!(accum_fn, z, b_ra; init=0.0f0, dims=3))
            z
        end ≈ accumulate(accum_fn, b; dims=3, init=0.0f0) broken = RunningOnTPU
    end
end

sameunitrange(x, y) = first(x) == first(y) && last(x) == last(y)

@testset "searchsorted" begin
    x = [1, 2, 4, 5, 5, 7]
    x_ra = Reactant.to_rarray(x)

    @testset "searchsortedfirst" begin
        @testset for val in (4, 5, 3, 9, 0)
            @test @jit(searchsortedfirst(x_ra, val)) == searchsortedfirst(x, val)
            @test @jit(searchsortedfirst(x_ra, ConcreteRNumber(val))) ==
                searchsortedfirst(x, val)
        end
    end

    @testset "searchsortedlast" begin
        @testset for val in (4, 5, 3, 9, 0)
            @test @jit(searchsortedlast(x_ra, val)) == searchsortedlast(x, val)
            @test @jit(searchsortedlast(x_ra, ConcreteRNumber(val))) ==
                searchsortedlast(x, val)
        end
    end

    @testset "searchsorted" begin
        @testset for val in (4, 5, 3, 9, 0)
            @test sameunitrange(@jit(searchsorted(x_ra, val)), searchsorted(x, val))
            @test sameunitrange(
                @jit(searchsorted(x_ra, ConcreteRNumber(val))), searchsorted(x, val)
            )
        end
    end
end

@testset "circshift" begin
    x = reshape(collect(Float32, 1:36), 2, 6, 3)
    x_ra = Reactant.to_rarray(x)

    @test @jit(circshift(x_ra, (1, 2))) ≈ circshift(x, (1, 2))
    @test @jit(circshift(x_ra, (1, 2, 3))) ≈ circshift(x, (1, 2, 3))
    @test @jit(circshift(x_ra, (-3, 2))) ≈ circshift(x, (-3, 2))
    @test @jit(circshift(x_ra, (5, 2))) ≈ circshift(x, (5, 2))
end

linrange_mat(x1, x2) = Reactant.materialize_traced_array(LinRange(x1, x2, 10024))

@testset "LinRange" begin
    x1 = 0.0f0
    x2 = 1.0f0
    x1_ra = Reactant.to_rarray(x1; track_numbers=Number)
    x2_ra = Reactant.to_rarray(x2; track_numbers=Number)

    @test @jit(linrange_mat(x1_ra, x2_ra)) ≈ collect(LinRange(x1, x2, 10024))
    hlo = repr(@code_hlo(linrange_mat(x1_ra, x2_ra)))
    @test contains(hlo, "stablehlo.iota")
end

@testset "chlo legalize to stablehlo" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 4, 4)
    x_ra = Reactant.to_rarray(x)

    hlo1 = repr(@code_hlo Reactant.Ops.conj(x_ra))
    hlo2 = repr(@code_hlo legalize_chlo_to_stablehlo = true Reactant.Ops.conj(x_ra))

    @test contains(hlo1, "chlo.conj")
    @test !contains(hlo2, "chlo")
end

@testset "scalar indexing in any #1434" begin
    xr = Reactant.to_rarray(ones(4, 4))
    @test @jit(any(<(0), xr)) == any(<(0), Array(xr))
end

@testset "copyto!, no offsets" begin
    a = Float32[10, 20, 30, 40, 50]
    len = length(a)
    b = Float32[111, 222, 333, 444, 555]
    cpu = fill(0.0f0, len)
    gpu = (Reactant.@jit Reactant.Ops.fill(0.0f0, (len,)))

    cpu .= a
    gpu .= b
    copyto!(cpu, gpu)
    @test gpu == b
    @test cpu == b

    cpu .= a
    gpu .= b
    copyto!(gpu, cpu)
    @test gpu == a
    @test cpu == a
end

@testset "copyto!, with offsets" begin
    a = Float32[10, 20, 30, 40, 50, 60, 70]
    alen = length(a)
    b = Float32[111, 222, 333, 444, 555]
    blen = length(b)

    dest = fill(0.0f0, alen)
    src = Reactant.@jit Reactant.Ops.fill(0.0f0, (blen,))

    for desto in 1:alen, srco in 1:blen, l in 1:min(blen - srco + 1, alen - desto + 1)

        # TODO offset-enabled copy not implemented for IFRTArray
        if src isa ConcretePJRTArray
            expected = copyto!(copy(a), desto, b, srco, l)

            dest .= a
            src .= b
            copyto!(dest, desto, src, srco, l)
            @test dest == expected
        end
    end

    # TODO direct copy not implemented for IFRTArray
    dest = Reactant.@jit Reactant.Ops.fill(0.0f0, (alen,))
    if dest isa ConcretePJRTArray
        src = fill(0.0f0, blen)
        for desto in 1:alen, srco in 1:blen, l in 1:min(blen - srco + 1, alen - desto + 1)
            expected = copyto!(copy(a), desto, b, srco, l)

            dest .= a
            src .= b
            copyto!(dest, desto, src, srco, l)
            @test dest == expected
        end
    end
end

zip_iterator(a, b) = mapreduce(splat(*), +, zip(a, b))
zip_iterator2(a, b) = mapreduce(splat(.-), +, zip(a, b))
enumerate_iterator(a) = mapreduce(splat(*), +, enumerate(a))
enumerate_iterator2(a) = mapreduce(splat(.-), +, enumerate(a))
mapreduce_vector(a) = mapreduce(-, +, a)

function nested_mapreduce_zip(x, y)
    return mapreduce(+, zip(eachcol(x), eachcol(y)); init=0.0f0) do (x, y)
        return sum(abs2, x) + sum(abs2, y)
    end
end

function nested_mapreduce_hcat(x, y)
    return mapreduce(
        hcat, zip(eachcol(x), eachcol(y)); init=similar(x, size(x, 1), 0)
    ) do (x, y)
        return x .+ y
    end
end

function f_generator(points, params)
    return sum(params * point for point in points)
end

@testset "Base.Iterators" begin
    @testset "zip" begin
        N = 10
        a = collect(range(1.0, 5.0; length=N))
        x = collect(range(10.0, 15.0; length=N + 2))
        x_ra = Reactant.to_rarray(x)

        @test @jit(zip_iterator(a, x_ra)) ≈ zip_iterator(a, x)

        a = [Reactant.TestUtils.construct_test_array(Float32, 2, 3) for _ in 1:10]
        x = [Reactant.TestUtils.construct_test_array(Float32, 2, 3) for _ in 1:10]
        a_ra = Reactant.to_rarray(a)
        x_ra = Reactant.to_rarray(x)

        @test @jit(zip_iterator2(a_ra, x_ra)) ≈ zip_iterator2(a, x)
    end

    @testset "enumerate" begin
        x = collect(range(1.0, 5.0; length=10))
        x_ra = Reactant.to_rarray(x)

        @test @jit(enumerate_iterator(x_ra)) ≈ enumerate_iterator(x)

        x = [Reactant.TestUtils.construct_test_array(Float32, 2, 3) for _ in 1:10]
        x_ra = Reactant.to_rarray(x)

        @test @jit(enumerate_iterator2(x_ra)) ≈ enumerate_iterator2(x)
    end

    @testset "nested mapreduce" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)
        @test @jit(nested_mapreduce_zip(x_ra, y_ra)) ≈ nested_mapreduce_zip(x, y)
    end
    @testset "nested mapreduce hcat" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(nested_mapreduce_hcat(x_ra, y_ra)) ≈ nested_mapreduce_hcat(x, y)
    end
end

@testset "Base.Generator" begin
    points = eachcol(Reactant.TestUtils.construct_test_array(Float32, 2, 6))
    params = Reactant.TestUtils.construct_test_array(Float32, 4, 2)
    points_ra = Reactant.to_rarray(points)
    params_ra = Reactant.to_rarray(params)

    @test @jit(f_generator(points_ra, params_ra)) ≈ f_generator(points, params)
end

@testset "compilation cache" begin
    if Reactant.PersistentCompileCache.autotune_cache_enabled() &&
        contains(string(Reactant.devices()[1]), "CUDA")
        A = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 5))
        B = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 5, 1000))
        @jit A * B # This should populate the cache dir

        @test any(
            endswith(".textproto"),
            readdir(Reactant.PersistentCompileCache.get_autotune_cache_directory()),
        )
    end
end

@testset "mapreduce with unitrange dims" begin
    x = reshape(collect(Float32, 1:64), 2, 4, 8)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra; dims=1:2)) ≈ sum(x; dims=1:2)
end

stack_numbers(x) = stack([sum(x[:, i]) for i in axes(x, 2)])

@testset "stack numbers" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(stack_numbers(x_ra)) ≈ stack_numbers(x)
end

@testset "copysign/mod type check" begin
    x = ConcreteRNumber(Int32(5))
    y = ConcreteRNumber(Int32(3))
    @test @jit(copysign(x, y)) isa ConcreteRNumber{Int32}
    @test @jit(mod(x, y)) isa ConcreteRNumber{Int32}
end

@testset "mod1" begin
    x = collect(Int32, 1:12)
    y = Int32(10)

    @testset for xᵢ in x
        res = @jit mod1(ConcreteRNumber(xᵢ), ConcreteRNumber(y))
        @test res isa ConcreteRNumber{Int32}
        @test res == mod1(xᵢ, y)
    end
end

@testset "log10" begin
    x = collect(Float64, 1:10)
    x_ra = Reactant.to_rarray(x)
    @test Array(@jit(log10.(x_ra))) ≈ log10.(x)
end

@testset "log2" begin
    x = collect(Float64, 1:10)
    x_ra = Reactant.to_rarray(x)
    @test Array(@jit(log2.(x_ra))) ≈ log2.(x)
end

map_test_1(i, xᵢ, yᵢ) = xᵢ + yᵢ + max(xᵢ, yᵢ)

@testset "multi-argument map" begin
    x = collect(Float32, 1:10)
    y = collect(Float32, 31:40)

    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    gt = map(map_test_1, 1:length(x), x, y)
    @test @jit(map(map_test_1, 1:length(x), x_ra, y_ra)) ≈ gt

    z = similar(x)
    z_ra = Reactant.to_rarray(z)
    map!(map_test_1, z, 1:length(x), x, y)
    @jit map!(map_test_1, z_ra, 1:length(x), x_ra, y_ra)
    @test z ≈ z_ra
    @test z_ra ≈ gt
end

@testset "repeat specialize" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 3))

    hlo = repr(@code_hlo(repeat(x_ra, 2, 3)))
    @test !contains(hlo, "stablehlo.dynamic_update_slice")
end

@testset "call through inference barrier" begin
    points = [
        Reactant.TestUtils.construct_test_array(Float32, 2),
        Reactant.TestUtils.construct_test_array(Float32, 2),
    ]
    params = Reactant.TestUtils.construct_test_array(Float32, 4, 2)
    points_ra = Reactant.to_rarray(points)
    params_ra = Reactant.to_rarray(params)

    f(params, points) = mapreduce(Base.Fix1(*, params), +, points)

    @test @jit(f(params_ra, points_ra)) ≈ f(params, points)
end

@testset "clamp!" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 32, 32)
    x_ra = Reactant.to_rarray(x)
    @test @jit(clamp!(x_ra, 0.5, Inf32)) ≈ clamp!(x, 0.5, Inf32)
end

mapped_sub(xs...) = stack(map(-, xs...))

@testset "map of slices" begin
    # We shouldn't be using `elem_apply` in this case and instead unroll the map
    # our passes will fuse them backup if needed.
    @testset "Vector of Slices" begin
        x_full = Reactant.TestUtils.construct_test_array(Float32, 10, 5, 3)
        y_full = Reactant.TestUtils.construct_test_array(Float32, 10, 5, 3)
        x = [view(x_full, :, i, :) for i in 1:size(x_full, 2)]
        y = [view(y_full, :, i, :) for i in 1:size(y_full, 2)]
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(mapped_sub(x_ra, y_ra)) ≈ mapped_sub(x, y) atol = 1e-5 rtol = 1e-5
    end

    @testset "Slices" begin
        x_full = Reactant.TestUtils.construct_test_array(Float32, 10, 5)

        @testset "ColumnSlices" begin
            x_sliced = eachcol(x_full)
            x_ra = Reactant.to_rarray(x_sliced)

            @test @jit(mapped_sub(x_ra)) ≈ mapped_sub(x_sliced) atol = 1e-5 rtol = 1e-5
        end
    end
end

@testset "AbstractRange Unwanted Promotions" begin
    hlo1 = @code_hlo Reactant.promote_to(Reactant.TracedRArray, Base.OneTo(Int32(42)))
    @test !contains(repr(hlo1), "42xi64")
    @test contains(repr(hlo1), "42xi32")

    hlo2 = @code_hlo Reactant.promote_to(Reactant.TracedRArray, Int32(34):(Int32(42)))
    @test !contains(repr(hlo2), "9xi64")
    @test contains(repr(hlo2), "9xi32")

    hlo3 = @code_hlo Reactant.promote_to(
        Reactant.TracedRArray, Int16(34):Int16(3):(Int16(42))
    )
    @test !contains(repr(hlo3), "3xi64")
    @test !contains(repr(hlo3), "3xi32")
    @test contains(repr(hlo3), "3xi16")
end

init_zeros(x) = zeros(eltype(x), 2, 3, 4)
init_zeros_0d(x) = zeros(eltype(x))
init_ones(x) = ones(eltype(x), 2, 3, 4)
init_ones_0d(x) = ones(eltype(x))
init_fill(x) = fill(eltype(x)(5), 2, 3, 4)
init_fill_0d(x) = fill(eltype(x)(5))

@testset "zeros/ones/fill" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2)
    x_ra = Reactant.to_rarray(x)

    @test @jit(init_zeros(x_ra)) ≈ init_zeros(x)
    @test @jit(init_zeros_0d(x_ra)) ≈ init_zeros_0d(x)
    @test @jit(init_ones(x_ra)) ≈ init_ones(x)
    @test @jit(init_ones_0d(x_ra)) ≈ init_ones_0d(x)
    @test @jit(init_fill(x_ra)) ≈ init_fill(x)
    @test @jit(init_fill_0d(x_ra)) ≈ init_fill_0d(x)
end

@testset "Mismatched Thunk Error" begin
    x_ra1 = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2))
    x_ra2 = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 2))

    fn = @compile sum(x_ra1)
    @test_throws Reactant.Compiler.MisMatchedThunkTypeError fn(x_ra2)
end

@testset "Slices" begin
    @testset "drop=true" begin
        x = eachslice(
            Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5); dims=(3, 1)
        )
        x_ra = Reactant.to_rarray(x)

        @test @jit(sum(x_ra)) ≈ sum(x)

        @testset for dims in (1, 2, (1, 2), (2, 1))
            res_ra = @jit sum(x_ra; dims)
            res = sum(x; dims)
            @test size(res_ra) == size(res)
            for (gt, comp) in zip(res_ra, res)
                @test gt ≈ comp
            end
        end
    end

    @testset "drop=false" begin
        x = eachslice(
            Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5);
            dims=(3, 1),
            drop=false,
        )
        x_ra = Reactant.to_rarray(x)

        @test @jit(sum(x_ra)) ≈ sum(x)

        @testset for dims in (1, 2, 3, 4, (1, 2), (1, 2, 4), (3, 4, 1), (2, 1))
            res_ra = @jit sum(x_ra; dims)
            res = sum(x; dims)
            @test size(res_ra) == size(res)
            for (gt, comp) in zip(res_ra, res)
                @test gt ≈ comp
            end
        end
    end
end

function meshgrid(args::AbstractVector...)
    return let N = length(args)
        stack(enumerate(args)) do (i, arg)
            new_shape = ones(Int, N)
            new_shape[i] = length(arg)
            repeat_sizes = collect(Int, map(length, args))
            repeat_sizes[i] = 1
            return repeat(reshape(arg, new_shape...), repeat_sizes...)
        end
    end
end

function meshgrid(x::Number, y::Number)
    return meshgrid(range(eltype(x)(0), x; length=10), range(eltype(y)(0), y; length=10))
end

@testset "meshgrid" begin
    x = 10.0f0
    y = 20.0f0
    x_ra = ConcreteRNumber(x)
    y_ra = ConcreteRNumber(y)

    @test @jit(meshgrid(x_ra, y_ra)) ≈ meshgrid(x, y)
end
