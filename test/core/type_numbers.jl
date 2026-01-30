# Tests for type handling, scalars, and number operations
using Reactant, Test, Enzyme
using ArrayInterface

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

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

@testset "aos_to_soa" begin
    x_res = collect(reshape(1.0:4.0, 2, 1, 2))
    x_ca = Reactant.to_rarray.(x_res; track_numbers=Number)

    y_ca1 = @allowscalar ArrayInterface.aos_to_soa(x_ca)
    @test y_ca1 ≈ x_res
    @test y_ca1 isa ConcreteRArray

    y_ca2 = @jit(ArrayInterface.aos_to_soa(x_ca))
    @test y_ca2 ≈ x_res
    @test y_ca2 isa ConcreteRArray
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

@testset "typemin/typemax" begin
    fn(x) = [typemin(eltype(x)), typemax(eltype(x))]

    x_ra = Reactant.to_rarray(ones(4))
    @test @jit(fn(x_ra)) == fn(ones(4))

    x_ra = Reactant.to_rarray(ones(Int, 4))
    @test @jit(fn(x_ra)) == fn(ones(Int, 4))
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

@testset "Preserve Aliasing" begin
    x = Reactant.to_rarray([3])

    if x isa ConcretePJRTArray
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

bc_apply(t::NTuple{N,T}, x) where {N,T} = sum(ntuple(n -> t[n], Val(N))) * x
function tobc(nt, x)
    nt = Ref(nt)
    return bc_apply.(nt, x)
end

@testset "Broadcast Ref" begin
    x = [2.7, 3.1]
    xr = Reactant.to_rarray(x)
    nt = map(ConcreteRNumber, (10.0, 10.0))

    res = @jit tobc(nt, x)

    @test res ≈ tobc((10.0, 10.0), x)
end
