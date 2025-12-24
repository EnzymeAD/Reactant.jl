using Reactant, Test
using Reactant: Ops
using LinearAlgebra
using SpecialFunctions: SpecialFunctions

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")
const RunningOnAppleX86 = Sys.isapple() && Sys.ARCH === :x86_64

@testset "abs" begin
    x = Reactant.to_rarray([1, -1])
    @test [1, 1] ≈ @jit Ops.abs(x)

    x = Reactant.to_rarray([1.0, -1.0])
    @test [1.0, 1.0] ≈ @jit Ops.abs(x)

    x = Reactant.to_rarray(ComplexF32[
        3.0+4im -3.0+4im
        3.0-4im -3.0-4im
    ])
    @test [
        5.0 5.0
        5.0 5.0
    ] ≈ @jit(Ops.abs(x))
end

@testset "add" begin
    a = Reactant.to_rarray([false, false, true, true])
    b = Reactant.to_rarray([false, true, false, true])
    @test [false, true, true, true] ≈ @jit Ops.add(a, b)

    a = Reactant.to_rarray([1, 2, 3, 4])
    b = Reactant.to_rarray([5, 6, -7, -8])
    @test Array(a) .+ Array(b) ≈ @jit Ops.add(a, b)

    a = Reactant.to_rarray([1.1, 2.2, 3.3, 4.4])
    b = Reactant.to_rarray([5.5, 6.6, -7.7, -8.8])
    @test Array(a) .+ Array(b) ≈ @jit Ops.add(a, b)

    a = Reactant.to_rarray(ComplexF32[1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    b = Reactant.to_rarray(
        ComplexF32[9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im]
    )
    @test Array(a) .+ Array(b) ≈ @jit(Ops.add(a, b))
end

@testset "after_all" begin
    # TODO
end

@testset "and" begin
    a = Reactant.to_rarray([false, false, true, true])
    b = Reactant.to_rarray([false, true, false, true])
    @test [false, false, false, true] ≈ @jit Ops.and(a, b)

    a = Reactant.to_rarray([1, 2, 3, 4])
    b = Reactant.to_rarray([5, 6, -7, -8])
    @test [1, 2, 1, 0] == @jit Ops.and(a, b)
end

@testset "atan2" begin
    a = Reactant.to_rarray([1.1, 2.2, 3.3, 4.4])
    b = Reactant.to_rarray([5.5, 6.6, -7.7, -8.8])
    @test atan.(Array(a), Array(b)) ≈ @jit Ops.atan2(a, b)

    # TODO couldn't find the definition of complex atan2 to compare against, but it should be implemented
end

@testset "cbrt" begin
    x = Reactant.to_rarray([1.0, 8.0, 27.0, 64.0])
    @test [1.0, 2.0, 3.0, 4.0] ≈ @jit Ops.cbrt(x)

    # TODO currently crashes, reenable when #291 is fixed
    # x = Reactant.to_rarray([1.0 + 2.0im, 8.0 + 16.0im, 27.0 + 54.0im, 64.0 + 128.0im])
    # @test Array(x) .^ (1//3) ≈ @jit Ops.cbrt(x)
end

@testset "ceil" begin
    x = Reactant.to_rarray(
        [
            1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9 10.0
            -1.1 -2.2 -3.3 -4.4 -5.5 -6.6 -7.7 -8.8 -9.9 -10.0
        ],
    )
    @test ceil.(x) ≈ @jit Ops.ceil(x)
end

@testset "cholesky" begin
    # cholesky in stablehlo for the other triangle is implementation defined.
    # See https://github.com/EnzymeAD/Reactant.jl/issues/338 for more details.
    g1(x) = triu(Ops.cholesky(x))
    g2(x) = tril(Ops.cholesky(x; lower=true))

    x = Reactant.to_rarray([
        10.0 2.0 3.0
        2.0 5.0 6.0
        3.0 6.0 9.0
    ])
    @test cholesky(Array(x)).U ≈ @jit g1(x)
    @test transpose(cholesky(Array(x)).U) ≈ @jit g2(x)

    x = Reactant.to_rarray(
        ComplexF32[
            10.0+0.0im 2.0-3.0im 3.0-4.0im
            2.0+3.0im 5.0+0.0im 3.0-2.0im
            3.0+4.0im 3.0+2.0im 9.0+0.0im
        ],
    )
    @test cholesky(Array(x)).U ≈ @jit(g1(x))
    @test adjoint(cholesky(Array(x)).U) ≈ @jit(g2(x))
end

@testset "clamp" begin
    for (_min, _max) in [
        (3, 7),
        (
            Reactant.to_rarray(3; track_numbers=true),
            Reactant.to_rarray(7; track_numbers=true),
        ),
        (
            Reactant.to_rarray([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            Reactant.to_rarray([7, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
        ),
    ]
        x = Reactant.to_rarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        @test [3, 3, 3, 4, 5, 6, 7, 7, 7, 7] == @jit Ops.clamp(_min, x, _max)
    end

    for (_min, _max) in [
        (3.0, 7.0),
        (
            Reactant.to_rarray(3.0; track_numbers=true),
            Reactant.to_rarray(7.0; track_numbers=true),
        ),
        (
            Reactant.to_rarray([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            Reactant.to_rarray([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
        ),
    ]
        x = Reactant.to_rarray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0])
        @test [3.0, 3.0, 3.3, 4.4, 5.5, 6.6, 7.0, 7.0, 7.0, 7.0] ≈
            @jit Ops.clamp(_min, x, _max)
    end
end

@testset "complex" begin
    x = Reactant.to_rarray(1.1f0; track_numbers=true)
    y = Reactant.to_rarray(2.2f0; track_numbers=true)
    @test ComplexF32(1.1 + 2.2im) ≈ @jit(Ops.complex(x, y))

    x = Reactant.to_rarray([1.1f0, 2.2f0, 3.3f0, 4.4f0])
    y = Reactant.to_rarray([5.5f0, 6.6f0, -7.7f0, -8.8f0])
    @test ComplexF32[1.1 + 5.5im, 2.2 + 6.6im, 3.3 - 7.7im, 4.4 - 8.8im] ≈
        @jit(Ops.complex(x, y))
end

@testset "constant" begin
    for x in [[1, 2, 3], [1.1, 2.2, 3.3], [1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im]]
        if x isa AbstractArray{ComplexF64} && contains(string(Reactant.devices()[1]), "TPU")
            continue
        end
        @test x ≈ @jit Ops.constant(x)

        xscalar = x[1]
        @test xscalar ≈ @jit Ops.constant(xscalar)
    end
end

@testset "cosine" begin
    # it crashes in apple x86_64 and it's a deprecated platform so we don't need to care a lot...
    x = Reactant.to_rarray([0, π / 2, π, 3π / 2, 2π])
    @test [1, 0, -1, 0, 1] ≈ @jit(Ops.cosine(x)) broken = RunningOnAppleX86

    x = Reactant.to_rarray([0.0, π / 2, π, 3π / 2, 2π])
    @test [1.0, 0.0, -1.0, 0.0, 1.0] ≈ @jit(Ops.cosine(x)) broken = RunningOnAppleX86

    @test ComplexF32[1.0 + 0.0im, 0.0 + 0.0im, -1.0 + 0.0im, 0.0 + 0.0im, 1.0 + 0.0im] ≈
        @jit(
        Ops.cosine(
            Reactant.to_rarray(
                ComplexF32[
                    0.0 + 0.0im, π / 2 + 0.0im, π + 0.0im, 3π / 2 + 0.0im, 2π + 0.0im
                ],
            ),
        )
    ) skip = RunningOnAppleX86
end

@testset "count_leading_zeros" begin
    x = Reactant.to_rarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    @test [64, 63, 62, 62, 61, 61, 61, 61, 60, 60] ≈ @jit Ops.count_leading_zeros(x)
end

@testset "divide" begin
    a = Reactant.to_rarray([5, 6, -7, -8])
    b = Reactant.to_rarray([1, 2, 3, 4])
    @test Array(a) .÷ Array(b) ≈ @jit Ops.divide(a, b)

    for (a, b) in [
        ([1.1, 2.2, 3.3, 4.4], [5.5, 6.6, -7.7, -8.8]),
        (
            [1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im],
            [9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im],
        ),
    ]
        if a isa AbstractArray{ComplexF64} && contains(string(Reactant.devices()[1]), "TPU")
            continue
        end
        a = Reactant.to_rarray(a)
        b = Reactant.to_rarray(b)
        @test Array(a) ./ Array(b) ≈ @jit Ops.divide(a, b)
    end
end

@testset "dot_general" begin
    # dot product of first dim
    f1(x, y) = Ops.dot_general(x, y; contracting_dimensions=[[1], [1]])

    # outer product
    fouter(x, y) = Ops.dot_general(x, y; contracting_dimensions=[Int[], Int[]])

    # outer product, batch first dim
    fouter_batch1(x, y) = Ops.dot_general(
        x, y; contracting_dimensions=[Int[], Int[]], batching_dimensions=[[1], [1]]
    )

    for (a, b) in [
        (Int32[1, 2, 3, 4], Int32[5, 6, -7, -8]),
        ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, -7.0, -8.0]),
        (ComplexF32[1.0, 2.0im, 3.0, 4.0im], ComplexF32[5.0, 6.0im, -7.0im, -8.0]),
    ]
        a = Reactant.to_rarray(a)
        b = Reactant.to_rarray(b)
        # NOTE `LinearAlgebra.dot` is not equal to `sum(a .* b)` on complex numbers due to conjugation
        @test sum(a .* b) ≈ @jit f1(a, b)
        @test kron(reshape(Array(a), length(a), 1), reshape(Array(b), 1, length(b))) ≈
            @jit fouter(a, b)
        @test a .* b ≈ @jit fouter_batch1(a, b)
    end

    a = Reactant.to_rarray(Int32[1 2; 3 4])
    b = Reactant.to_rarray(Int32[5 6; -7 -8])
    @test Array(a)' * Array(b) == @jit(f1(a, b))
end

@testset "exponential" begin
    x = [1.0, 2.0, 3.0, 4.0]
    @test exp.(x) ≈ @jit Ops.exponential(Reactant.to_rarray(x))

    x = ComplexF32[1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im]
    @test exp.(x) ≈ @jit(Ops.exponential(Reactant.to_rarray(x))) skip = RunningOnAppleX86
end

@testset "exponential_minus_one" begin
    x = [1.0, 2.0, 3.0, 4.0]
    @test expm1.(x) ≈ @jit Ops.exponential_minus_one(Reactant.to_rarray(x))

    x = ComplexF32[1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im]
    @test expm1.(x) ≈ @jit(Ops.exponential_minus_one(Reactant.to_rarray(x))) skip =
        RunningOnAppleX86
end

@testset "fft" begin
    grfft(x) = Ops.fft(x; type="RFFT", length=[4])
    gfft(x) = Ops.fft(x; type="FFT", length=[4])

    x = Reactant.to_rarray(Float32[1.0, 1.0, 1.0, 1.0])
    @test ComplexF32[4.0, 0.0, 0.0] ≈ @jit(grfft(x))

    x = Reactant.to_rarray(Float32[0.0, 1.0, 0.0, -1.0])
    @test ComplexF32[0.0, -2.0im, 0.0] ≈ @jit(grfft(x))

    x = Reactant.to_rarray(Float32[1.0, -1.0, 1.0, -1.0])
    @test ComplexF32[0.0, 0.0, 4.0] ≈ @jit(grfft(x))

    x = Reactant.to_rarray(ComplexF32[1.0, 1.0, 1.0, 1.0])
    @test ComplexF32[4.0, 0.0, 0.0, 0.0] ≈ @jit(gfft(x))

    x = Reactant.to_rarray(ComplexF32[0.0, 1.0, 0.0, -1.0])
    @test ComplexF32[0.0, -2.0im, 0.0, 2.0im] ≈ @jit(gfft(x))

    x = Reactant.to_rarray(ComplexF32[1.0, -1.0, 1.0, -1.0])
    @test ComplexF32[0.0, 0.0, 4.0, 0.0] ≈ @jit(gfft(x))

    # TODO test with complex numbers and inverse FFT
end

@testset "floor" begin
    x = Reactant.to_rarray(
        [
            1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9 10.0
            -1.1 -2.2 -3.3 -4.4 -5.5 -6.6 -7.7 -8.8 -9.9 -10.0
        ],
    )
    @test floor.(x) ≈ @jit Ops.floor(x)
end

@testset "get_dimension_size" begin
    x = Reactant.to_rarray(fill(0, (1, 2, 3, 4)))
    for i in 1:4
        @test i == @jit Ops.get_dimension_size(x, i)
    end
end

@testset "imag" begin
    x = Reactant.to_rarray(ComplexF32[1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    @test Float32[2.2, 4.4, 6.6, 8.8] ≈ @jit(Ops.imag(x))
end

@testset "iota" begin
    g1(shape) = Ops.iota(Int, shape; iota_dimension=1)
    @test [
        0 0 0 0 0
        1 1 1 1 1
        2 2 2 2 2
        3 3 3 3 3
    ] ≈ @jit g1([4, 5])

    g2(shape) = Ops.iota(Int, shape; iota_dimension=2)
    @test [
        0 1 2 3 4
        0 1 2 3 4
        0 1 2 3 4
        0 1 2 3 4
    ] ≈ @jit g2([4, 5])
end

@testset "is_finite" begin
    x = Reactant.to_rarray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [false, false, false, true, true, true, true] ≈ @jit Ops.is_finite(x)
end

@testset "log" begin
    x = Reactant.to_rarray([1.0, 2.0, 3.0, 4.0])
    @test log.(Array(x)) ≈ @jit Ops.log(x)

    x = Reactant.to_rarray(ComplexF32[1.0 + 0.0im, 2.0 + 0.0im, -3.0 + 0.0im, -4.0 + 0.0im])
    @test log.(Array(x)) ≈ @jit(Ops.log(x))
end

@testset "log_plus_one" begin
    x = Reactant.to_rarray([1.0, 2.0, 3.0, 4.0])
    @test log.(Array(x)) ≈ @jit Ops.log(x)

    x = Reactant.to_rarray(ComplexF32[1.0 + 0.0im, 2.0 + 0.0im, -3.0 + 0.0im, -4.0 + 0.0im])
    @test log.(Array(x)) ≈ @jit(Ops.log(x))
end

@testset "logistic" begin
    x = Reactant.to_rarray([0.0, 1.0, 2.0, 3.0])
    l(x) = 1 / (1 + exp(-x))
    @test l.(Array(x)) ≈ @jit Ops.logistic(x)
end

@testset "maximum" begin
    x = Reactant.to_rarray([false, false, true, true])
    y = Reactant.to_rarray([false, true, false, true])
    @test [false, true, true, true] == @jit Ops.maximum(x, y)

    x = Reactant.to_rarray([-1, 0, 1, 10])
    y = Reactant.to_rarray([10, 1, 0, -1])
    @test [10, 1, 1, 10] == @jit Ops.maximum(x, y)

    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 10.0])
    y = Reactant.to_rarray([10.0, 1.0, 0.0, -1.0])
    @test [10.0, 1.0, 1.0, 10.0] == @jit Ops.maximum(x, y)
end

@testset "minimum" begin
    x = Reactant.to_rarray([false, false, true, true])
    y = Reactant.to_rarray([false, true, false, true])
    @test [false, false, false, true] == @jit Ops.minimum(x, y)

    x = Reactant.to_rarray([-1, 0, 1, 10])
    y = Reactant.to_rarray([10, 1, 0, -1])
    @test [-1, 0, 0, -1] == @jit Ops.minimum(x, y)

    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 10.0])
    y = Reactant.to_rarray([10.0, 1.0, 0.0, -1.0])
    @test [-1.0, 0.0, 0.0, -1.0] == @jit Ops.minimum(x, y)
end

@testset "multiply" begin
    x = Reactant.to_rarray([false, false, true, true])
    y = Reactant.to_rarray([false, true, false, true])
    @test [false, false, false, true] == @jit Ops.multiply(x, y)

    for (a, b) in [
        ([5, 6, -7, -8], [1, 2, 3, 4]),
        ([1.1, 2.2, 3.3, 4.4], [5.5, 6.6, -7.7, -8.8]),
        (
            [1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im],
            [9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im],
        ),
    ]
        if contains(string(Reactant.devices()[1]), "TPU")
            continue
        end
        a = Reactant.to_rarray(a)
        b = Reactant.to_rarray(b)
        @test Array(a) .* Array(b) ≈ @jit Ops.multiply(a, b)
    end
end

@testset "negate" begin
    x = Reactant.to_rarray([-1, 0, 1, 10])
    @test [1, 0, -1, -10] == @jit Ops.negate(x)

    # on unsigned integers: (1) bitcast, (2) change sign and (3) bitcast
    x = Reactant.to_rarray(UInt[0, 1, 10])
    @test reinterpret(UInt, Base.checked_neg.(reinterpret.(Int, Array(x)))) ==
        @jit Ops.negate(x)

    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 10.0])
    @test [1.0, 0.0, -1.0, -10.0] ≈ @jit Ops.negate(x)

    x = Reactant.to_rarray(ComplexF32[-1.0 + 2im, 0.0 - 3im, 1.0 + 4im, 10.0 - 5im])
    @test [1.0 - 2im, 0.0 + 3im, -1.0 - 4im, -10.0 + 5im] ≈ @jit(Ops.negate(x))
end

@testset "not" begin
    x = Reactant.to_rarray([false, true])
    @test [true, false] == @jit Ops.not(x)

    x = Reactant.to_rarray([1, 0])
    @test [~1, ~0] == @jit Ops.not(x)
end

@testset "optimization_barrier" begin
    # TODO is there a better way to test this? we're only testing for identify
    # TODO crashing for just 1 argument
    x = Reactant.to_rarray([1, 2, 3, 4])
    y = Reactant.to_rarray([5, 6, -7, -8])
    @test (x, y) == @jit Ops.optimization_barrier(x, y)
end

@testset "or" begin
    a = Reactant.to_rarray([false, false, true, true])
    b = Reactant.to_rarray([false, true, false, true])
    @test [false, true, true, true] ≈ @jit Ops.or(a, b)

    a = Reactant.to_rarray([1, 2, 3, 4])
    b = Reactant.to_rarray([5, 6, -7, -8])
    @test Array(a) .| Array(b) == @jit Ops.or(a, b)
end

@testset "outfeed" begin end

@testset "pad" begin
    x = Reactant.to_rarray([1, 2, 3, 4])
    v = Reactant.to_rarray(0; track_numbers=true)

    flow(x, v) = Ops.pad(x, v; low=[1])
    @test [0, 1, 2, 3, 4] == @jit flow(x, v)

    fhigh(x, v) = Ops.pad(x, v; high=[1])
    @test [1, 2, 3, 4, 0] == @jit fhigh(x, v)

    finterior(x, v) = Ops.pad(x, v; interior=[1])
    @test [1, 0, 2, 0, 3, 0, 4] == @jit finterior(x, v)

    x = Reactant.to_rarray([1 2; 3 4])

    glow(x, v) = Ops.pad(x, v; low=[1, 2])
    @test [0 0 0 0; 0 0 1 2; 0 0 3 4] == @jit glow(x, v)

    ghigh(x, v) = Ops.pad(x, v; high=[1, 2])
    @test [1 2 0 0; 3 4 0 0; 0 0 0 0] == @jit ghigh(x, v)

    ginterior(x, v) = Ops.pad(x, v; interior=[1, 2])
    @test [1 0 0 2; 0 0 0 0; 3 0 0 4] == @jit ginterior(x, v)
end

@testset "partition_id" begin
    @test @jit(Ops.partition_id()) isa ConcreteRNumber{UInt32}
end

@testset "popcnt" begin
    x = Reactant.to_rarray([0, 1, 2, 127])
    @test [0, 1, 1, 7] == @jit Ops.popcnt(x)
end

@testset "power" begin
    x = Reactant.to_rarray([-1, -1, -1, -1])
    p = Reactant.to_rarray([0, 1, 2, 3])
    @test Array(x) .^ Array(p) == @jit Ops.power(x, p)

    x = Reactant.to_rarray(ComplexF32[0.0 + 1.0im, 0.0 + 1.0im, 0.0 + 1.0im, 0.0 + 1.0im])
    p = Reactant.to_rarray(ComplexF32[0.0 + 0.0im, 1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im])
    @test Array(x) .^ Array(p) ≈ @jit(Ops.power(x, p))
end

@testset "real" begin
    x = Reactant.to_rarray(ComplexF32[1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    @test [1.1, 3.3, 5.5, 7.7] ≈ @jit(Ops.real(x))
end

@testset "recv" begin end

@testset "remainder" begin
    for (a, b) in [
        (Reactant.to_rarray([1, 2, 3, 4]), Reactant.to_rarray([5, 6, -7, -8])),
        (
            Reactant.to_rarray([1.1, 2.2, 3.3, 4.4]),
            Reactant.to_rarray([5.5, 6.6, -7.7, -8.8]),
        ),
    ]
        @test Array(a) .% Array(b) ≈ @jit Ops.remainder(a, b)
    end
end

@testset "replica_id" begin
    @test @jit(Ops.partition_id()) isa ConcreteRNumber{UInt32}
end

@testset "reshape" begin
    x = Reactant.to_rarray([1, 2, 3, 4])
    @test reshape(Array(x), 2, 2) == @jit Ops.reshape(x, 2, 2)

    x = Reactant.to_rarray(collect(reshape(1:12, 2, 2, 3)))
    @test reshape(Array(x), 3, 1, 4) == @jit Ops.reshape(x, 3, 1, 4)
end

@testset "reverse" begin
    x = Reactant.to_rarray([1, 2, 3, 4])
    g1(x) = Ops.reverse(x; dimensions=[1])
    @test [4, 3, 2, 1] == @jit g1(x)

    x = Reactant.to_rarray([1 2; 3 4])
    g2(x) = Ops.reverse(x; dimensions=[2])
    @test [3 4; 1 2] == @jit g1(x)
    @test [2 1; 4 3] == @jit g2(x)
end

@testset "rng_bit_generator" begin
    @testset for (alg, sz) in
                 [("DEFAULT", 2), ("PHILOX", 2), ("PHILOX", 3), ("THREE_FRY", 2)]
        seed = Reactant.to_rarray(zeros(UInt64, sz))

        res = @jit Ops.rng_bit_generator(Int32, seed, [2, 4]; algorithm=alg)
        @test res.output_state !== seed
        @test size(res.output_state) == (sz,)
        @test res.output isa ConcreteRArray{Int32,2}
        @test size(res.output) == (2, 4)
        seed = res.output_state

        res = @jit Ops.rng_bit_generator(Int64, seed, [2, 4]; algorithm=alg)
        @test res.output_state !== seed
        @test size(res.output_state) == (sz,)
        @test res.output isa ConcreteRArray{Int64,2}
        @test size(res.output) == (2, 4)
        seed = res.output_state

        res = @jit Ops.rng_bit_generator(UInt64, seed, [2, 4]; algorithm=alg)
        @test res.output_state !== seed
        @test size(res.output_state) == (sz,)
        @test res.output isa ConcreteRArray{UInt64,2}
        @test size(res.output) == (2, 4)
        seed = res.output_state

        res = @jit Ops.rng_bit_generator(Float32, seed, [2, 4]; algorithm=alg)
        @test res.output_state !== seed
        @test size(res.output_state) == (sz,)
        @test res.output isa ConcreteRArray{Float32,2}
        @test size(res.output) == (2, 4)
        seed = res.output_state

        res = @jit Ops.rng_bit_generator(Float64, seed, [2, 4]; algorithm=alg)
        @test res.output_state !== seed
        @test size(res.output_state) == (sz,)
        @test res.output isa ConcreteRArray{Float64,2}
        @test size(res.output) == (2, 4)
        seed = res.output_state

        res = @jit Ops.rng_bit_generator(Float32, seed, [2, 4]; algorithm=alg)
        @test res.output_state !== seed
        @test size(res.output_state) == (sz,)
        @test res.output isa ConcreteRArray{Float32,2}
        @test size(res.output) == (2, 4)
        seed = res.output_state
    end
end

@testset "round_nearest_afz" begin
    x = Reactant.to_rarray([-2.5, 0.4, 0.5, 0.6, 2.5])
    @test [-3.0, 0.0, 1.0, 1.0, 3.0] ≈ @jit Ops.round_nearest_afz(x)
end

@testset "round_nearest_even" begin
    x = Reactant.to_rarray([-2.5, 0.4, 0.5, 0.6, 2.5])
    @test [-2.0, 0.0, 0.0, 1.0, 2.0] ≈ @jit Ops.round_nearest_even(x)
end

@testset "rsqrt" begin
    x = Reactant.to_rarray([1.0 4.0; 9.0 25.0])
    @test 1 ./ sqrt.(Array(x)) ≈ @jit Ops.rsqrt(x)

    x = Reactant.to_rarray(ComplexF32[1.0+1im 4.0+2im; 9.0+3im 25.0+4im])
    @test 1 ./ sqrt.(Array(x)) ≈ @jit(Ops.rsqrt(x))
end

@testset "select" begin
    ontrue = Reactant.to_rarray([1, 2, 3, 4])
    onfalse = Reactant.to_rarray([5, 6, -7, -8])

    pred = Reactant.to_rarray([true, true, false, false])
    @test [1, 2, -7, -8] == @jit Ops.select(pred, ontrue, onfalse)

    pred = Reactant.to_rarray([false, false, true, true])
    @test [5, 6, 3, 4] == @jit Ops.select(pred, ontrue, onfalse)

    pred = Reactant.to_rarray(true; track_numbers=true)
    @test ontrue == @jit Ops.select(pred, ontrue, onfalse)

    pred = Reactant.to_rarray(false; track_numbers=true)
    @test onfalse == @jit Ops.select(pred, ontrue, onfalse)

    ontrue = Reactant.to_rarray(1; track_numbers=true)
    onfalse = Reactant.to_rarray(2; track_numbers=true)

    pred = Reactant.to_rarray(true; track_numbers=true)
    @test ontrue == @jit Ops.select(pred, ontrue, onfalse)

    pred = Reactant.to_rarray(false; track_numbers=true)
    @test onfalse == @jit Ops.select(pred, ontrue, onfalse)
end

@testset "send" begin end

@testset "set_dimension_size" begin end

@testset "shift_left" begin
    a = Reactant.to_rarray([-1, 0, 1])
    b = Reactant.to_rarray([1, 2, 3])
    @test [-2, 0, 8] == @jit Ops.shift_left(a, b)
end

@testset "shift_right_arithmetic" begin
    a = Reactant.to_rarray([-1, 0, 8])
    b = Reactant.to_rarray([1, 2, 3])
    @test [-1, 0, 1] == @jit Ops.shift_right_arithmetic(a, b)
end

@testset "shift_right_logical" begin
    a = Reactant.to_rarray([-1, 0, 8])
    b = Reactant.to_rarray([1, 2, 3])
    @test [9223372036854775807, 0, 1] == @jit Ops.shift_right_logical(a, b)
end

@testset "sign" begin
    x = Reactant.to_rarray([-1, 0, 1])
    @test [-1, 0, 1] == @jit Ops.sign(x)

    x = Reactant.to_rarray([Inf, -Inf, NaN, -NaN, -1.0, -0.0, +0.0, 1.0])
    @test [1.0, -1.0, NaN, NaN, -1.0, -0.0, 0.0, 1.0] ≈ @jit(Ops.sign(x)) nans = true

    x = Reactant.to_rarray(
        ComplexF32[
            NaN + 1.0im, 1.0 + NaN, 0.0 + 0.0im, -1.0 + 2.0im, 0.0 - 3.0im, 1.0 + 4.0im
        ],
    )
    @test ComplexF32[
        NaN + NaN * im,
        NaN + NaN * im,
        0.0 + 0.0im,
        -0.4472135954999579 + 0.8944271909999159im,
        0.0 - 1.0im,
        0.24253562503633297 + 0.9701425001453319im,
    ] ≈ @jit(Ops.sign(x)) nans = true
end

@testset "sine" begin
    x = Reactant.to_rarray([0, π / 2, π, 3π / 2, 2π])
    @test [0, 1, 0, -1, 0] ≈ @jit(Ops.sine(x)) broken = RunningOnAppleX86

    x = Reactant.to_rarray([0.0, π / 2, π, 3π / 2, 2π])
    @test [0.0, 1.0, 0.0, -1.0, 0.0] ≈ @jit(Ops.sine(x))

    x = Reactant.to_rarray(
        ComplexF32[0.0 + 0.0im, π / 2 + 0.0im, π + 0.0im, 3π / 2 + 0.0im, 2π + 0.0im]
    )
    @test ComplexF32[0.0 + 0.0im, 1.0 + 0.0im, 0.0 + 0.0im, -1.0 + 0.0im, 0.0 + 0.0im] ≈
        @jit(Ops.sine(x)) broken = RunningOnAppleX86
end

@testset "sort" begin
    basic_sort(x, dimension) = only(Ops.sort(x; comparator=(a, b) -> a < b, dimension))
    @testset for i in 1:3
        t_size = tuple(fill(10, (i,))...)
        x = Reactant.TestUtils.construct_test_array(Float32, t_size...)
        xa = Reactant.to_rarray(x)

        @testset for j in 1:i
            @test (i == 1 ? sort(x) : sort(x; dims=j)) ≈ @jit basic_sort(xa, j)
        end
    end
end

@testset "slice" begin
    x = Reactant.to_rarray([1, 2, 3, 4])
    @test [2, 3] == @jit Ops.slice(x, [2], [3])
    @test [1] == @jit Ops.slice(x, [1], [1])
end

@testset "sqrt" begin
    x = Reactant.to_rarray([1.0, 4.0, 9.0, 16.0])
    @test [1.0, 2.0, 3.0, 4.0] ≈ @jit Ops.sqrt(x)

    x = Reactant.to_rarray(ComplexF32[1.0 + 0im, 0.0 + 1im])
    @test ComplexF32[1.0 + 0im, 1 / √2 * (1 + im)] ≈ @jit(Ops.sqrt(x)) broken =
        RunningOnAppleX86
end

@testset "subtract" begin
    for (a, b) in [
        ([1, 2, 3, 4], [5, 6, -7, -8]),
        ([1.1, 2.2, 3.3, 4.4], [5.5, 6.6, -7.7, -8.8]),
        (
            [1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im],
            [9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im],
        ),
    ]
        if contains(string(Reactant.devices()[1]), "TPU")
            continue
        end
        a = Reactant.to_rarray(a)
        b = Reactant.to_rarray(b)
        @test Array(a) .- Array(b) == @jit Ops.subtract(a, b)
    end
end

@testset "tan" begin
    # TODO: tan(π/2) is not inf
    x = Reactant.to_rarray([0, π / 4, π / 3, 3π / 4, π])

    @test [0.0, 1.0, 1.73205, -1.0, 0.0] ≈ @jit(Ops.tan(x)) atol = 1e-5 rtol = 1e-3 broken =
        RunningOnAppleX86

    x = Reactant.to_rarray(
        ComplexF32[0.0 + 0.0im, π / 4 + 0.0im, π / 3 + 0.0im, 3π / 4 + 0.0im, π + 0.0im]
    )
    @test ComplexF32[0.0, 1.0, 1.73205, -1.0, 0.0] ≈ @jit(Ops.tan(x)) atol = 1e-5 rtol =
        1e-3 broken = RunningOnAppleX86
end

@testset "tanh" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test [-0.7615941559557649, 0.0, 0.7615941559557649] ≈ @jit Ops.tanh(x)

    x = Reactant.to_rarray(ComplexF32[-1.0, 0.0, 1.0])
    @test ComplexF32[-0.7615941559557649, 0.0, 0.7615941559557649] ≈ @jit(Ops.tanh(x)) skip =
        RunningOnAppleX86
end

@testset "transpose" begin
    x = Reactant.to_rarray(collect(reshape(1:12, 2, 2, 3)))
    @test [
        1 3; 5 7; 9 11;;;
        2 4; 6 8; 10 12
    ] == @jit Ops.transpose(x, [3, 2, 1])
end

# NOTE deprecated
# @testset "unary_einsum" begin
#     f1(a) = Ops.unary_einsum(a; equation="i->")
#     f4(a) = Ops.unary_einsum(a; equation="ij->")
#     f3(a) = Ops.unary_einsum(a; equation="ij->ji")
#     f4(a) = Ops.unary_einsum(a; equation="ij->j")
#     f5(a) = Ops.unary_einsum(a; equation="ij->i")
#     f6(a) = Ops.unary_einsum(a; equation="ii->i")

#     x = Reactant.to_rarray([1, 2, 3, 4])
#     @test sum(Array(x)) ≈ @jit f1(x)

#     x = Reactant.to_rarray([1 2; 3 4])
#     @test sum(Array(x)) ≈ @jit f4(x)
#     @test Base.transpose(Array(x)) ≈ @jit f3(x)
#     @test sum(Array(x); dims=1) ≈ @jit f4(x)
#     @test sum(Array(x); dims=2) ≈ @jit f5(x)
#     @test diag(Array(x)) ≈ @jit f6(x)
# end

@testset "xor" begin
    a = Reactant.to_rarray([false, false, true, true])
    b = Reactant.to_rarray([false, true, false, true])
    @test [false, true, true, false] ≈ @jit Ops.xor(a, b)

    a = Reactant.to_rarray([1, 2, 3, 4])
    b = Reactant.to_rarray([5, 6, -7, -8])
    @test Array(a) .⊻ Array(b) == @jit Ops.xor(a, b)
end

@testset "acos" begin
    x = Reactant.to_rarray(Float32[-1.0, 0.0, 1.0])
    @test acos.(Array(x)) ≈ @jit(Ops.acos(x))
end

@testset "acosh" begin
    x = Reactant.to_rarray(Float32[1.0, 10.0])
    @test acosh.(Array(x)) ≈ @jit(Ops.acosh(x))
end

@testset "asin" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test asin.(Array(x)) ≈ @jit Ops.asin(x)
end

@testset "asinh" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test asinh.(Array(x)) ≈ @jit Ops.asinh(x)
end

@testset "atan" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test atan.(Array(x)) ≈ @jit Ops.atan(x)
end

@testset "atanh" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test atanh.(Array(x)) ≈ @jit Ops.atanh(x)
end

@testset "bessel_i1e" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    @test SpecialFunctions.besselix.(1, Array(x)) ≈ @jit Ops.bessel_i1e(x)
end

@testset "conj" begin
    x = Reactant.to_rarray(ComplexF32[-1.0 + 2im, 0.0 - 1im, 1.0 + 4im])
    @test conj(Array(x)) ≈ @jit(Ops.conj(x))
end

@testset "cosh" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test cosh.(Array(x)) ≈ @jit Ops.cosh(x)
end

@testset "digamma" begin
    # small divergence between chlo.digamma and SpecialFunctions.digamma:
    # on <=0, chlo.digamma returns NaN, SpecialFunctions.digamma returns Inf
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test [NaN, NaN, SpecialFunctions.digamma(1.0)] ≈ @jit(Ops.digamma(x)) nans = true skip =
        RunningOnAppleX86
end

@testset "erf_inv" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test SpecialFunctions.erfinv.(Array(x)) ≈ @jit Ops.erf_inv(x)
end

@testset "erf" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test SpecialFunctions.erf.(Array(x)) ≈ @jit Ops.erf(x)
end

@testset "erfc" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test SpecialFunctions.erfc.(Array(x)) ≈ @jit Ops.erfc(x)
end

@testset "is_inf" begin
    x = Reactant.to_rarray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [true, true, false, false, false, false, false] ≈ @jit Ops.is_inf(x)
end

@testset "is_neg_inf" begin
    x = Reactant.to_rarray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [true, false, false, false, false, false, false] ≈ @jit Ops.is_neg_inf(x)
end

@testset "is_pos_inf" begin
    x = Reactant.to_rarray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [false, true, false, false, false, false, false] ≈ @jit Ops.is_pos_inf(x)
end

@testset "lgamma" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 2.5])
    lgamma(x) = (SpecialFunctions.logabsgamma(x))[1]
    @test lgamma.(Array(x)) ≈ @jit(Ops.lgamma(x)) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnAppleX86
end

@testset "next_after" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 1.0, 2.5, 1e18, 1e18, 3e-9, 3e-9])
    y = Reactant.to_rarray([-2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1e19, 0, 1])
    @test [
        prevfloat(-1.0),
        0.0,
        1.0,
        nextfloat(1.0),
        nextfloat(2.5),
        prevfloat(1e18),
        nextfloat(1e18),
        prevfloat(3e-9),
        nextfloat(3e-9),
    ] == @jit(Ops.next_after(x, y)) skip = RunningOnTPU
end

@testset "polygamma" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0, 1.0, 2.5])
    m = Reactant.to_rarray([3.0, 3.0, 2.0, 3.0, 4.0])
    @test SpecialFunctions.polygamma.(Int.(Array(m)), Array(x)) ≈ @jit(Ops.polygamma(m, x)) broken =
        RunningOnAppleX86
end

@testset "sinh" begin
    x = Reactant.to_rarray([-1.0, 0.0, 1.0])
    @test sinh.(Array(x)) ≈ @jit Ops.sinh(x)
end

@testset "top_k" begin
    x = Reactant.to_rarray([1, 2, 3, 4])
    @test (; values=[4, 3], indices=[4, 3]) == @jit Ops.top_k(x, 2)

    x = Reactant.to_rarray([NaN, 123, 456, 789, 121])
    res = @jit Ops.top_k(x, 2)
    true_res = (; values=[NaN, 789], indices=[1, 4])
    @test res.indices == true_res.indices
    @test @allowscalar isnan(res.values[1])
    @test @allowscalar res.values[2] == 789
end

@testset "zeta" begin
    s = Reactant.to_rarray([1.0, 2.0, 50.0])
    z = Reactant.to_rarray([1e-8, 0.001, 2.0])
    @test SpecialFunctions.zeta.(Array(s), Array(z)) ≈ @jit Ops.zeta(s, z)
end

@testset "hlo_call" begin
    x = Float32[1.0, 2.0, 50.0]
    y = Float32[-4.0, 0.001, 2.0]
    x_reactant = Reactant.to_rarray(x)
    y_reactant = Reactant.to_rarray(y)

    @test Reactant.@jit(
        Ops.hlo_call(
            """
            module {
              func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
                return %0 : tensor<3xf32>
              }
            }
            """,
            x_reactant,
            y_reactant,
        )
    )[1] ≈ x .+ y

    @test Float32(
        only(
            Reactant.@jit(
                Ops.hlo_call(
                    """
                    module {
                      func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                        %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
                        return %0 : tensor<f32>
                      }
                    }
                    """,
                    Reactant.ConcreteRNumber(2.0f0),
                    Reactant.ConcreteRNumber(2.0f0),
                )
            )
        ),
    ) == 4.0f0
end

function f_repeat(x, y)
    for _ in 1:3
        x, = Ops.hlo_call(
            """
            module {
              func.func @my_add(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
                return %0 : tensor<3xf32>
              }
            }
            """,
            x,
            y;
            func_name="my_add",
        )
    end
    return x
end

@testset "hlo_call: repeat" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))
    y = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))
    mod = Reactant.@code_hlo optimize = false f_repeat(x, y)
    hlo_ir = repr(mod)

    add_pos = findfirst("stablehlo.add", hlo_ir)
    @test !isnothing(add_pos)

    add_pos = findfirst("stablehlo.add", hlo_ir[last(add_pos):end])
    @test isnothing(add_pos)
end

@testset "hlo_call: multiple functions" begin
    @test Reactant.@jit(
        Ops.hlo_call(
            """
            module {
              func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                %0 = func.call @add(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
                return %0 : tensor<3xf32>
              }
              func.func @add(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
                return %0 : tensor<3xf32>
              }
            }
            """,
            Reactant.to_rarray(Float32[1, 2, 3]),
            Reactant.to_rarray(Float32[1, 2, 3]),
        )
    )[1] ≈ Float32[2, 4, 6]
end

function f_multiple_hlo_calls(x, y)
    x, = Ops.hlo_call(
        """
        module {
          func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """,
        x,
        y,
    )
    return Ops.hlo_call(
        """
        module {
          func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """,
        x,
        y,
    )
end

@testset "hlo_call: multiple hlo_calls" begin
    x = Float32[1.0, 2.0, 50.0]
    y = Float32[-4.0, 0.001, 2.0]
    x_reactant = Reactant.to_rarray(x)
    y_reactant = Reactant.to_rarray(y)

    @test Reactant.@jit(f_multiple_hlo_calls(x_reactant, y_reactant))[1] ≈ (x .+ y) .* y
end

@testset "reduce" begin
    # stablehlo reduce collapse the dimension so that (1,3) beomces (3, )
    # while Julia reduce retains (1, 3). The test will fail despite elements being equal
    function squeeze_dims(r)
        return dropdims(r; dims=tuple(findall(size(r) .== 1)...))
    end

    # Floating point operation is not associative
    A = Reactant.TestUtils.construct_test_array(Int64, 3, 4, 5)
    A_ra = Reactant.to_rarray(A)
    init = 1
    init_ra = @jit Reactant.Ops.constant(init)
    dims = [2]
    r_hlo = @jit Reactant.Ops.reduce(A_ra, init_ra, dims, *)
    r = reduce(*, A; dims=dims, init=init)
    @test r_hlo ≈ squeeze_dims(r)

    dims = [1, 3]
    init = 0
    init_ra = @jit Reactant.Ops.constant(init)
    r_hlo = @jit Reactant.Ops.reduce(A_ra, init_ra, dims, +)
    r = reduce(+, A; dims=dims, init=init)
    @test r_hlo ≈ squeeze_dims(r)

    dims = [1, 2, 3]
    r_hlo = @jit Reactant.Ops.reduce(A_ra, init_ra, dims, +)
    r = reduce(+, A; dims=dims, init=init)
    @test r_hlo ≈ squeeze_dims(r)
end

@testset "const dedup" begin
    x = Reactant.to_rarray([11, 12, 13, 14])
    function const_dedup(x)
        c1 = [1, 2, 3, 4]
        y1 = (x .+ c1)
        c2 = [1, 2, 3, 4]
        y2 = (x .+ c2)
        c1[1] = 6
        return y1 .* y2 .* c1
    end

    mod = @code_hlo optimize = false const_dedup(x)
    hlo_ir = repr(mod)
    csts = collect(x for x in eachsplit(hlo_ir, "\n") if occursin("stablehlo.constant", x))
    # calls to similar give rise to dense<0> constants (that are not deduplicated):
    csts = filter(x -> !occursin("dense<0>", x), csts)

    @test length(csts) == 2
    idx = findfirst(x -> occursin("1, 2, 3, 4", x), csts)
    @test idx !== nothing
    if idx == 1
        @test occursin("6, 2, 3, 4", csts[2])
    else
        @test occursin("6, 2, 3, 4", csts[1])
    end
end

@testset "Large constant" begin
    N = 5
    constant = Reactant.TestUtils.construct_test_array(Float64, N)
    v = Reactant.TestUtils.construct_test_array(Float64, N)
    vr = Reactant.to_rarray(v)
    # Function which would use the `constant` object
    f!(v) = v .+= constant
    default_threshold = Ops.LARGE_CONSTANT_THRESHOLD[]
    default_raise_error = Ops.LARGE_CONSTANT_RAISE_ERROR[]
    try
        Ops.LARGE_CONSTANT_THRESHOLD[] = N
        Ops.LARGE_CONSTANT_RAISE_ERROR[] = true
        @compile f!(vr)
    catch err
        @test err.msg ==
            "Generating a constant of 40 bytes, which larger than the $(N) bytes threshold"
    finally
        # Restore threshold
        Ops.LARGE_CONSTANT_THRESHOLD[] = default_threshold
        Ops.LARGE_CONSTANT_RAISE_ERROR[] = default_raise_error
    end
    # Make sure we can now compile the function
    fr! = @compile f!(vr)
    @test fr!(vr) ≈ f!(v)
end

fn_test_wrap(x) = Reactant.Ops.wrap(x, 2, 1; dimension=3)

@testset "Ops.wrap" begin
    x = Reactant.to_rarray(rand(2, 3, 4, 5))
    out = @jit fn_test_wrap(x)

    @test size(out) == (2, 3, 7, 5)
end

@testset "Ops.fill" begin
    @testset "Fill with TracedScalar" begin
        fn(x) = Ops.fill(x, [2, 3])
        x_ra = ConcreteRNumber(1.0f0)
        y_ra = @jit fn(x_ra)
        @test y_ra isa ConcreteRArray{Float32,2}
        @test Array(y_ra) == ones(Float32, 2, 3)
    end
end

function recon_from_lu(lu_res::AbstractArray{T,4}) where {T}
    y = similar(lu_res)
    for i in 1:size(lu_res, 1), j in 1:size(lu_res, 2)
        y[i, j, :, :] .= recon_from_lu(lu_res[i, j, :, :])
    end
    return y
end

function apply_permutation(x::AbstractArray{T,4}, perm) where {T}
    y = similar(x)
    for i in 1:size(x, 1), j in 1:size(x, 2)
        y[i, j, :, :] .= x[i, j, perm[i, j, :], :]
    end
    return y
end

function recon_from_lu(lu_res::AbstractMatrix)
    return UnitLowerTriangular(lu_res) * UpperTriangular(lu_res)
end

@testset "lu factorization" begin
    @testset "unbatched" begin
        x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 6, 6))
        lu_ra, ipiv, perm, info = @jit Ops.lu(x_ra)

        @test @jit(recon_from_lu(lu_ra)) ≈ @jit(getindex(x_ra, perm, :)) atol = 1e-5 rtol =
            1e-2
    end

    @testset "batched" begin
        x_ra = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 4, 3, 6, 6)
        )
        lu_ra, ipiv, perm, info = @jit Ops.lu(x_ra)
        @test size(lu_ra) == (4, 3, 6, 6)
        @test size(ipiv) == (4, 3, 6)
        @test size(perm) == (4, 3, 6)
        @test size(info) == (4, 3)

        @test @jit(recon_from_lu(lu_ra)) ≈ @jit(apply_permutation(x_ra, perm)) atol = 1e-5 rtol =
            1e-2
    end
end

@testset "batch norm" begin
    @testset "training" begin
        @testset for affine in [false, true]
            x = Reactant.to_rarray(
                Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5)
            )
            if affine
                scale = Reactant.to_rarray(
                    Reactant.TestUtils.construct_test_array(Float32, 3)
                )
                offset = Reactant.to_rarray(
                    Reactant.TestUtils.construct_test_array(Float32, 3)
                )
            else
                scale, offset = nothing, nothing
            end

            hlo = @code_hlo Ops.batch_norm_training(
                x, scale, offset; epsilon=1e-5, feature_index=2
            )
            @test occursin("stablehlo.batch_norm_training", repr(hlo))

            if !affine
                @test occursin(
                    "stablehlo.constant dense<0.000000e+00> : tensor<3xf32>", repr(hlo)
                )
                @test occursin(
                    "stablehlo.constant dense<1.000000e+00> : tensor<3xf32>", repr(hlo)
                )
            end

            res, m, v = @jit Ops.batch_norm_training(
                x, scale, offset; epsilon=1e-5, feature_index=2
            )
            @test size(res) == size(x)
            @test size(m) == (3,)
            @test size(v) == (3,)
        end
    end

    @testset "inference" begin
        @testset for affine in [false, true]
            x = Reactant.to_rarray(
                Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5)
            )
            if affine
                scale = Reactant.to_rarray(
                    Reactant.TestUtils.construct_test_array(Float32, 3)
                )
                offset = Reactant.to_rarray(
                    Reactant.TestUtils.construct_test_array(Float32, 3)
                )
            else
                scale, offset = nothing, nothing
            end

            rm = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))
            rv = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))

            hlo = @code_hlo Ops.batch_norm_inference(
                x, scale, offset, rm, rv; epsilon=1e-5, feature_index=2
            )
            @test occursin("stablehlo.batch_norm_inference", repr(hlo))
            if !affine
                @test occursin(
                    "stablehlo.constant dense<0.000000e+00> : tensor<3xf32>", repr(hlo)
                )
                @test occursin(
                    "stablehlo.constant dense<1.000000e+00> : tensor<3xf32>", repr(hlo)
                )
            end

            res = @jit Ops.batch_norm_inference(
                x, scale, offset, rm, rv; epsilon=1e-5, feature_index=2
            )
            @test size(res) == size(x)
        end
    end

    @testset "batch_norm_grad" begin
        @testset for affine in [false, true]
            x = Reactant.to_rarray(
                Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5)
            )
            scale = if affine
                Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))
            else
                nothing
            end
            rm = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))
            rv = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 3))
            gx = Reactant.to_rarray(
                Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5)
            )

            hlo = @code_hlo Ops.batch_norm_grad(
                x, scale, rm, rv, gx; epsilon=1e-5, feature_index=2
            )
            @test occursin("stablehlo.batch_norm_grad", repr(hlo))

            if !affine
                @test occursin(
                    "stablehlo.constant dense<1.000000e+00> : tensor<3xf32>", repr(hlo)
                )
            end

            gres, gscale, goffset = @jit Ops.batch_norm_grad(
                x, scale, rm, rv, gx; epsilon=1e-5, feature_index=2
            )
            @test size(gres) == size(x)
            if !affine
                @test gscale === nothing
                @test goffset === nothing
            else
                @test size(gscale) == (3,)
                @test size(goffset) == (3,)
            end
        end
    end
end
