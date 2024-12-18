using Reactant, Test
using Reactant: Ops
using LinearAlgebra
using SpecialFunctions: SpecialFunctions

@testset "abs" begin
    x = ConcreteRArray([1, -1])
    @test [1, 1] ≈ @jit Ops.abs(x)

    x = ConcreteRArray([1.0, -1.0])
    @test [1.0, 1.0] ≈ @jit Ops.abs(x)

    x = ConcreteRArray([
        3.0+4im -3.0+4im
        3.0-4im -3.0-4im
    ])
    @test [
        5.0 5.0
        5.0 5.0
    ] ≈ @jit Ops.abs(x)
end

@testset "add" begin
    a = ConcreteRArray([false, false, true, true])
    b = ConcreteRArray([false, true, false, true])
    @test [false, true, true, false] ≈ @jit Ops.add(a, b)

    a = ConcreteRArray([1, 2, 3, 4])
    b = ConcreteRArray([5, 6, -7, -8])
    @test Array(a) .+ Array(b) ≈ @jit Ops.add(a, b)

    a = ConcreteRArray([1.1, 2.2, 3.3, 4.4])
    b = ConcreteRArray([5.5, 6.6, -7.7, -8.8])
    @test Array(a) .+ Array(b) ≈ @jit Ops.add(a, b)

    a = ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    b = ConcreteRArray([
        9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im
    ])
    @test Array(a) .+ Array(b) ≈ @jit Ops.add(a, b)
end

@testset "after_all" begin
    # TODO
end

@testset "and" begin
    a = ConcreteRArray([false, false, true, true])
    b = ConcreteRArray([false, true, false, true])
    @test [false, false, false, true] ≈ @jit Ops.and(a, b)

    a = ConcreteRArray([1, 2, 3, 4])
    b = ConcreteRArray([5, 6, -7, -8])
    @test [1, 2, 1, 0] == @jit Ops.and(a, b)
end

@testset "atan2" begin
    a = ConcreteRArray([1.1, 2.2, 3.3, 4.4])
    b = ConcreteRArray([5.5, 6.6, -7.7, -8.8])
    @test atan.(Array(a), Array(b)) ≈ @jit Ops.atan2(a, b)

    # TODO couldn't find the definition of complex atan2 to compare against, but it should be implemented
end

@testset "cbrt" begin
    x = ConcreteRArray([1.0, 8.0, 27.0, 64.0])
    @test [1.0, 2.0, 3.0, 4.0] ≈ @jit Ops.cbrt(x)

    # TODO currently crashes, reenable when #291 is fixed
    # x = ConcreteRArray([1.0 + 2.0im, 8.0 + 16.0im, 27.0 + 54.0im, 64.0 + 128.0im])
    # @test Array(x) .^ (1//3) ≈ @jit Ops.cbrt(x)
end

@testset "ceil" begin
    x = ConcreteRArray(
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

    x = ConcreteRArray([
        10.0 2.0 3.0
        2.0 5.0 6.0
        3.0 6.0 9.0
    ])
    @test cholesky(Array(x)).U ≈ @jit g1(x)
    @test transpose(cholesky(Array(x)).U) ≈ @jit g2(x)

    x = ConcreteRArray(
        [
            10.0+0.0im 2.0-3.0im 3.0-4.0im
            2.0+3.0im 5.0+0.0im 3.0-2.0im
            3.0+4.0im 3.0+2.0im 9.0+0.0im
        ],
    )

    @test cholesky(Array(x)).U ≈ @jit g1(x)
    @test adjoint(cholesky(Array(x)).U) ≈ @jit g2(x)
end

@testset "clamp" begin
    for (_min, _max) in [
        (3, 7),
        (ConcreteRNumber(3), ConcreteRNumber(7)),
        (
            ConcreteRArray([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            ConcreteRArray([7, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
        ),
    ]
        x = ConcreteRArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        @test [3, 3, 3, 4, 5, 6, 7, 7, 7, 7] == @jit Ops.clamp(_min, x, _max)
    end

    for (_min, _max) in [
        (3.0, 7.0),
        (ConcreteRNumber(3.0), ConcreteRNumber(7.0)),
        (
            ConcreteRArray([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            ConcreteRArray([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
        ),
    ]
        x = ConcreteRArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0])
        @test [3.0, 3.0, 3.3, 4.4, 5.5, 6.6, 7.0, 7.0, 7.0, 7.0] ==
            @jit Ops.clamp(_min, x, _max)
    end
end

@testset "complex" begin
    x = ConcreteRNumber(1.1)
    y = ConcreteRNumber(2.2)
    @test 1.1 + 2.2im ≈ @jit Ops.complex(x, y)

    x = ConcreteRArray([1.1, 2.2, 3.3, 4.4])
    y = ConcreteRArray([5.5, 6.6, -7.7, -8.8])
    @test [1.1 + 5.5im, 2.2 + 6.6im, 3.3 - 7.7im, 4.4 - 8.8im] ≈ @jit Ops.complex(x, y)
end

@testset "constant" begin
    for x in [[1, 2, 3], [1.1, 2.2, 3.3], [1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im]]
        @test x ≈ @jit Ops.constant(x)

        xscalar = x[1]
        @test xscalar ≈ @jit Ops.constant(xscalar)
    end
end

@testset "cosine" begin
    # it crashes in apple x86_64 and it's a deprecated platform so we don't need to care a lot...
    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([0, π / 2, π, 3π / 2, 2π])
        @test [1, 0, -1, 0, 1] ≈ @jit Ops.cosine(x)

        x = ConcreteRArray([0.0, π / 2, π, 3π / 2, 2π])
        @test [1.0, 0.0, -1.0, 0.0, 1.0] ≈ @jit Ops.cosine(x)

        x = ConcreteRArray([
            0.0 + 0.0im, π / 2 + 0.0im, π + 0.0im, 3π / 2 + 0.0im, 2π + 0.0im
        ])
        @test [1.0 + 0.0im, 0.0 + 0.0im, -1.0 + 0.0im, 0.0 + 0.0im, 1.0 + 0.0im] ≈
            @jit Ops.cosine(x)
    end
end

@testset "count_leading_zeros" begin
    x = ConcreteRArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    @test [64, 63, 62, 62, 61, 61, 61, 61, 60, 60] ≈ @jit Ops.count_leading_zeros(x)
end

@testset "divide" begin
    a = ConcreteRArray([5, 6, -7, -8])
    b = ConcreteRArray([1, 2, 3, 4])
    @test Array(a) .÷ Array(b) ≈ @jit Ops.divide(a, b)

    for (a, b) in [
        (ConcreteRArray([1.1, 2.2, 3.3, 4.4]), ConcreteRArray([5.5, 6.6, -7.7, -8.8])),
        (
            ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im]),
            ConcreteRArray([
                9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im
            ]),
        ),
    ]
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
        (ConcreteRArray([1, 2, 3, 4]), ConcreteRArray([5, 6, -7, -8])),
        (ConcreteRArray([1.0, 2.0, 3.0, 4.0]), ConcreteRArray([5.0, 6.0, -7.0, -8.0])),
        (
            ConcreteRArray([1.0, 2.0im, 3.0, 4.0im]),
            ConcreteRArray([5.0, 6.0im, -7.0im, -8.0]),
        ),
    ]
        # NOTE `LinearAlgebra.dot` is not equal to `sum(a .* b)` on complex numbers due to conjugation
        @test sum(a .* b) ≈ @jit f1(a, b)
        @test kron(reshape(Array(a), length(a), 1), reshape(Array(b), 1, length(b))) ≈
            @jit fouter(a, b)
        @test a .* b ≈ @jit fouter_batch1(a, b)
    end

    a = ConcreteRArray([1 2; 3 4])
    b = ConcreteRArray([5 6; -7 -8])
    @test Array(a)' * Array(b) == @jit f1(a, b)
end

@testset "einsum" begin
    f1(a, b) = Ops.einsum(a, b; equation="i,i->i")
    f2(a, b) = Ops.einsum(a, b; equation="i,j->ij")
    f3(a, b) = Ops.einsum(a, b; equation="ij,ij->ij")
    f4(a, b) = Ops.einsum(a, b; equation="ik,kj->ij")

    for (a, b) in [
        (ConcreteRArray([1, 2, 3, 4]), ConcreteRArray([5, 6, -7, -8])),
        (ConcreteRArray([1.0, 2.0, 3.0, 4.0]), ConcreteRArray([5.0, 6.0, -7.0, -8.0])),
        (
            ConcreteRArray([1.0 + 1im, 2.0 + 2im, 3.0 - 3im, 4.0 - 4im]),
            ConcreteRArray([5.0 + 5im, 6.0 + 6im, -7.0 - 7im, -8.0 - 8im]),
        ),
    ]
        @test a .* b ≈ @jit f1(a, b)
        @test reshape(kron(Array(b), Array(a)), 4, 4) ≈ @jit f2(a, b)

        x = ConcreteRArray(reshape(a, (2, 2)))
        y = ConcreteRArray(reshape(b, (2, 2)))
        @test x .* y ≈ @jit f3(x, y)
        @test Array(x) * Array(y) ≈ @jit f4(x, y)
    end
end

@testset "exponential" begin
    x = ConcreteRArray([1.0, 2.0, 3.0, 4.0])
    @test exp.(Array(x)) ≈ @jit Ops.exponential(x)

    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im])
        @test exp.(Array(x)) ≈ @jit Ops.exponential(x)
    end
end

@testset "exponential_minus_one" begin
    x = ConcreteRArray([1.0, 2.0, 3.0, 4.0])
    @test expm1.(Array(x)) ≈ @jit Ops.exponential_minus_one(x)

    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im])
        @test expm1.(Array(x)) ≈ @jit Ops.exponential_minus_one(x)
    end
end

@testset "fft" begin
    grfft(x) = Ops.fft(x; type="RFFT", length=[4])
    gfft(x) = Ops.fft(x; type="FFT", length=[4])

    x = ConcreteRArray([1.0, 1.0, 1.0, 1.0])
    @test ComplexF64[4.0, 0.0, 0.0] ≈ @jit grfft(x)

    x = ConcreteRArray([0.0, 1.0, 0.0, -1.0])
    @test ComplexF64[0.0, -2.0im, 0.0] ≈ @jit grfft(x)

    x = ConcreteRArray([1.0, -1.0, 1.0, -1.0])
    @test ComplexF64[0.0, 0.0, 4.0] ≈ @jit grfft(x)

    x = ConcreteRArray(ComplexF64[1.0, 1.0, 1.0, 1.0])
    @test ComplexF64[4.0, 0.0, 0.0, 0.0] ≈ @jit gfft(x)

    x = ConcreteRArray(ComplexF64[0.0, 1.0, 0.0, -1.0])
    @test ComplexF64[0.0, -2.0im, 0.0, 2.0im] ≈ @jit gfft(x)

    x = ConcreteRArray(ComplexF64[1.0, -1.0, 1.0, -1.0])
    @test ComplexF64[0.0, 0.0, 4.0, 0.0] ≈ @jit gfft(x)

    # TODO test with complex numbers and inverse FFT
end

@testset "floor" begin
    x = ConcreteRArray(
        [
            1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9 10.0
            -1.1 -2.2 -3.3 -4.4 -5.5 -6.6 -7.7 -8.8 -9.9 -10.0
        ],
    )
    @test floor.(x) ≈ @jit Ops.floor(x)
end

@testset "get_dimension_size" begin
    x = ConcreteRArray(fill(0, (1, 2, 3, 4)))
    for i in 1:4
        @test i == @jit Ops.get_dimension_size(x, i)
    end
end

@testset "imag" begin
    x = ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    @test [2.2, 4.4, 6.6, 8.8] ≈ @jit Ops.imag(x)
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
    x = ConcreteRArray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [false, false, false, true, true, true, true] ≈ @jit Ops.is_finite(x)
end

@testset "log" begin
    x = ConcreteRArray([1.0, 2.0, 3.0, 4.0])
    @test log.(Array(x)) ≈ @jit Ops.log(x)

    x = ConcreteRArray([1.0 + 0.0im, 2.0 + 0.0im, -3.0 + 0.0im, -4.0 + 0.0im])
    @test log.(Array(x)) ≈ @jit Ops.log(x)
end

@testset "log_plus_one" begin
    x = ConcreteRArray([1.0, 2.0, 3.0, 4.0])
    @test log.(Array(x)) ≈ @jit Ops.log(x)

    x = ConcreteRArray([1.0 + 0.0im, 2.0 + 0.0im, -3.0 + 0.0im, -4.0 + 0.0im])
    @test log.(Array(x)) ≈ @jit Ops.log(x)
end

@testset "logistic" begin
    x = ConcreteRArray([0.0, 1.0, 2.0, 3.0])
    l(x) = 1 / (1 + exp(-x))
    @test l.(Array(x)) ≈ @jit Ops.logistic(x)
end

@testset "maximum" begin
    x = ConcreteRArray([false, false, true, true])
    y = ConcreteRArray([false, true, false, true])
    @test [false, true, true, true] == @jit Ops.maximum(x, y)

    x = ConcreteRArray([-1, 0, 1, 10])
    y = ConcreteRArray([10, 1, 0, -1])
    @test [10, 1, 1, 10] == @jit Ops.maximum(x, y)

    x = ConcreteRArray([-1.0, 0.0, 1.0, 10.0])
    y = ConcreteRArray([10.0, 1.0, 0.0, -1.0])
    @test [10.0, 1.0, 1.0, 10.0] == @jit Ops.maximum(x, y)
end

@testset "minimum" begin
    x = ConcreteRArray([false, false, true, true])
    y = ConcreteRArray([false, true, false, true])
    @test [false, false, false, true] == @jit Ops.minimum(x, y)

    x = ConcreteRArray([-1, 0, 1, 10])
    y = ConcreteRArray([10, 1, 0, -1])
    @test [-1, 0, 0, -1] == @jit Ops.minimum(x, y)

    x = ConcreteRArray([-1.0, 0.0, 1.0, 10.0])
    y = ConcreteRArray([10.0, 1.0, 0.0, -1.0])
    @test [-1.0, 0.0, 0.0, -1.0] == @jit Ops.minimum(x, y)
end

@testset "multiply" begin
    x = ConcreteRArray([false, false, true, true])
    y = ConcreteRArray([false, true, false, true])
    @test [false, false, false, true] == @jit Ops.multiply(x, y)

    for (a, b) in [
        (ConcreteRArray([5, 6, -7, -8]), ConcreteRArray([1, 2, 3, 4])),
        (ConcreteRArray([1.1, 2.2, 3.3, 4.4]), ConcreteRArray([5.5, 6.6, -7.7, -8.8])),
        (
            ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im]),
            ConcreteRArray([
                9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im
            ]),
        ),
    ]
        @test Array(a) .* Array(b) ≈ @jit Ops.multiply(a, b)
    end
end

@testset "negate" begin
    x = ConcreteRArray([-1, 0, 1, 10])
    @test [1, 0, -1, -10] == @jit Ops.negate(x)

    # on unsigned integers: (1) bitcast, (2) change sign and (3) bitcast
    x = ConcreteRArray(UInt[0, 1, 10])
    @test reinterpret(UInt, Base.checked_neg.(reinterpret.(Int, Array(x)))) ==
        @jit Ops.negate(x)

    x = ConcreteRArray([-1.0, 0.0, 1.0, 10.0])
    @test [1.0, 0.0, -1.0, -10.0] ≈ @jit Ops.negate(x)

    x = ConcreteRArray([-1.0 + 2im, 0.0 - 3im, 1.0 + 4im, 10.0 - 5im])
    @test [1.0 - 2im, 0.0 + 3im, -1.0 - 4im, -10.0 + 5im] ≈ @jit Ops.negate(x)
end

@testset "not" begin
    x = ConcreteRArray([false, true])
    @test [true, false] == @jit Ops.not(x)

    x = ConcreteRArray([1, 0])
    @test [~1, ~0] == @jit Ops.not(x)
end

@testset "optimization_barrier" begin
    # TODO is there a better way to test this? we're only testing for identify
    # TODO crashing for just 1 argument
    x = ConcreteRArray([1, 2, 3, 4])
    y = ConcreteRArray([5, 6, -7, -8])
    @test (x, y) == @jit Ops.optimization_barrier(x, y)
end

@testset "or" begin
    a = ConcreteRArray([false, false, true, true])
    b = ConcreteRArray([false, true, false, true])
    @test [false, true, true, true] ≈ @jit Ops.or(a, b)

    a = ConcreteRArray([1, 2, 3, 4])
    b = ConcreteRArray([5, 6, -7, -8])
    @test Array(a) .| Array(b) == @jit Ops.or(a, b)
end

@testset "outfeed" begin end

@testset "pad" begin
    x = ConcreteRArray([1, 2, 3, 4])
    v = ConcreteRNumber(0)

    flow(x, v) = Ops.pad(x, v; low=[1])
    @test [0, 1, 2, 3, 4] == @jit flow(x, v)

    fhigh(x, v) = Ops.pad(x, v; high=[1])
    @test [1, 2, 3, 4, 0] == @jit fhigh(x, v)

    finterior(x, v) = Ops.pad(x, v; interior=[1])
    @test [1, 0, 2, 0, 3, 0, 4] == @jit finterior(x, v)

    x = ConcreteRArray([1 2; 3 4])

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
    x = ConcreteRArray([0, 1, 2, 127])
    @test [0, 1, 1, 7] == @jit Ops.popcnt(x)
end

@testset "power" begin
    x = ConcreteRArray([-1, -1, -1, -1])
    p = ConcreteRArray([0, 1, 2, 3])
    @test Array(x) .^ Array(p) == @jit Ops.power(x, p)

    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([0.0 + 1.0im, 0.0 + 1.0im, 0.0 + 1.0im, 0.0 + 1.0im])
        p = ConcreteRArray([0.0 + 0.0im, 1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im])
        @test Array(x) .^ Array(p) ≈ @jit Ops.power(x, p)
    end
end

@testset "real" begin
    x = ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    @test [1.1, 3.3, 5.5, 7.7] ≈ @jit Ops.real(x)
end

@testset "recv" begin end

@testset "remainder" begin
    for (a, b) in [
        (ConcreteRArray([1, 2, 3, 4]), ConcreteRArray([5, 6, -7, -8])),
        (ConcreteRArray([1.1, 2.2, 3.3, 4.4]), ConcreteRArray([5.5, 6.6, -7.7, -8.8])),
    ]
        @test Array(a) .% Array(b) ≈ @jit Ops.remainder(a, b)
    end
end

@testset "replica_id" begin
    @test @jit(Ops.partition_id()) isa ConcreteRNumber{UInt32}
end

@testset "reshape" begin
    x = ConcreteRArray([1, 2, 3, 4])
    @test reshape(Array(x), 2, 2) == @jit Ops.reshape(x, 2, 2)

    x = ConcreteRArray(collect(reshape(1:12, 2, 2, 3)))
    @test reshape(Array(x), 3, 1, 4) == @jit Ops.reshape(x, 3, 1, 4)
end

@testset "reverse" begin
    x = ConcreteRArray([1, 2, 3, 4])
    g1(x) = Ops.reverse(x; dimensions=[1])
    @test [4, 3, 2, 1] == @jit g1(x)

    x = ConcreteRArray([1 2; 3 4])
    g2(x) = Ops.reverse(x; dimensions=[2])
    @test [3 4; 1 2] == @jit g1(x)
    @test [2 1; 4 3] == @jit g2(x)
end

@testset "rng_bit_generator" begin
    # seed = ConcreteRArray([0, 0])
    # @jit Ops.rng_bit_generator(seed, [2])
end

@testset "round_nearest_afz" begin
    x = ConcreteRArray([-2.5, 0.4, 0.5, 0.6, 2.5])
    @test [-3.0, 0.0, 1.0, 1.0, 3.0] ≈ @jit Ops.round_nearest_afz(x)
end

@testset "round_nearest_even" begin
    x = ConcreteRArray([-2.5, 0.4, 0.5, 0.6, 2.5])
    @test [-2.0, 0.0, 0.0, 1.0, 2.0] ≈ @jit Ops.round_nearest_even(x)
end

@testset "rsqrt" begin
    x = ConcreteRArray([1.0 4.0; 9.0 25.0])
    @test 1 ./ sqrt.(Array(x)) ≈ @jit Ops.rsqrt(x)

    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([1.0+1im 4.0+2im; 9.0+3im 25.0+4im])
        @test 1 ./ sqrt.(Array(x)) ≈ @jit Ops.rsqrt(x)
    end
end

@testset "select" begin
    ontrue = ConcreteRArray([1, 2, 3, 4])
    onfalse = ConcreteRArray([5, 6, -7, -8])

    pred = ConcreteRArray([true, true, false, false])
    @test [1, 2, -7, -8] == @jit Ops.select(pred, ontrue, onfalse)

    pred = ConcreteRArray([false, false, true, true])
    @test [5, 6, 3, 4] == @jit Ops.select(pred, ontrue, onfalse)

    pred = ConcreteRNumber(true)
    @test ontrue == @jit Ops.select(pred, ontrue, onfalse)

    pred = ConcreteRNumber(false)
    @test onfalse == @jit Ops.select(pred, ontrue, onfalse)

    ontrue = ConcreteRNumber(1)
    onfalse = ConcreteRNumber(2)

    pred = ConcreteRNumber(true)
    @test ontrue == @jit Ops.select(pred, ontrue, onfalse)

    pred = ConcreteRNumber(false)
    @test onfalse == @jit Ops.select(pred, ontrue, onfalse)
end

@testset "send" begin end

@testset "set_dimension_size" begin end

@testset "shift_left" begin
    a = ConcreteRArray([-1, 0, 1])
    b = ConcreteRArray([1, 2, 3])
    @test [-2, 0, 8] == @jit Ops.shift_left(a, b)
end

@testset "shift_right_arithmetic" begin
    a = ConcreteRArray([-1, 0, 8])
    b = ConcreteRArray([1, 2, 3])
    @test [-1, 0, 1] == @jit Ops.shift_right_arithmetic(a, b)
end

@testset "shift_right_logical" begin
    a = ConcreteRArray([-1, 0, 8])
    b = ConcreteRArray([1, 2, 3])
    @test [9223372036854775807, 0, 1] == @jit Ops.shift_right_logical(a, b)
end

@testset "sign" begin
    x = ConcreteRArray([-1, 0, 1])
    @test [-1, 0, 1] == @jit Ops.sign(x)

    x = ConcreteRArray([Inf, -Inf, NaN, -NaN, -1.0, -0.0, +0.0, 1.0])
    @test [1.0, -1.0, NaN, NaN, -1.0, -0.0, 0.0, 1.0] ≈ @jit(Ops.sign(x)) nans = true

    x = ConcreteRArray([
        NaN + 1.0im, 1.0 + NaN, 0.0 + 0.0im, -1.0 + 2.0im, 0.0 - 3.0im, 1.0 + 4.0im
    ])
    @test [
        NaN + NaN * im,
        NaN + NaN * im,
        0.0 + 0.0im,
        -0.4472135954999579 + 0.8944271909999159im,
        0.0 - 1.0im,
        0.24253562503633297 + 0.9701425001453319im,
    ] ≈ @jit(Ops.sign(x)) nans = true
end

@testset "sine" begin
    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([0, π / 2, π, 3π / 2, 2π])
        @test [0, 1, 0, -1, 0] ≈ @jit Ops.sine(x)

        x = ConcreteRArray([0.0, π / 2, π, 3π / 2, 2π])
        @test [0.0, 1.0, 0.0, -1.0, 0.0] ≈ @jit Ops.sine(x)

        x = ConcreteRArray([
            0.0 + 0.0im, π / 2 + 0.0im, π + 0.0im, 3π / 2 + 0.0im, 2π + 0.0im
        ])
        @test [0.0 + 0.0im, 1.0 + 0.0im, 0.0 + 0.0im, -1.0 + 0.0im, 0.0 + 0.0im] ≈
            @jit Ops.sine(x)
    end
end

@testset "slice" begin
    x = ConcreteRArray([1, 2, 3, 4])
    @test [2, 3] == @jit Ops.slice(x, [2], [3])
    @test [1] == @jit Ops.slice(x, [1], [1])
end

@testset "sqrt" begin
    x = ConcreteRArray([1.0, 4.0, 9.0, 16.0])
    @test [1.0, 2.0, 3.0, 4.0] ≈ @jit Ops.sqrt(x)

    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([1.0 + 0im, 0.0 + 1im])
        @test [1.0 + 0im, 1 / √2 * (1 + im)] ≈ @jit Ops.sqrt(x)
    end
end

@testset "subtract" begin
    for (a, b) in [
        (ConcreteRArray([1, 2, 3, 4]), ConcreteRArray([5, 6, -7, -8])),
        (ConcreteRArray([1.1, 2.2, 3.3, 4.4]), ConcreteRArray([5.5, 6.6, -7.7, -8.8])),
        (
            ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im]),
            ConcreteRArray([
                9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im
            ]),
        ),
    ]
        @test Array(a) .- Array(b) == @jit Ops.subtract(a, b)
    end
end

@testset "tan" begin
    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        # TODO tan(π/2) is Inf but it returns 1.633123935319537e16
        x = ConcreteRArray([0, π / 4, π / 2, 3π / 4, π])
        @test [0.0, 1.0, 1.633123935319537e16, -1.0, 0.0] ≈ @jit Ops.tan(x)

        x = ConcreteRArray([
            0.0 + 0.0im, π / 4 + 0.0im, π / 2 + 0.0im, 3π / 4 + 0.0im, π + 0.0im
        ])
        @test ComplexF64[0.0, 1.0, 1.633123935319537e16, -1.0, 0.0] ≈ @jit Ops.tan(x)
    end
end

@testset "tanh" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test [-0.7615941559557649, 0.0, 0.7615941559557649] ≈ @jit Ops.tanh(x)

    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray(ComplexF64[-1.0, 0.0, 1.0])
        @test ComplexF64[-0.7615941559557649, 0.0, 0.7615941559557649] ≈ @jit Ops.tanh(x)
    end
end

@testset "transpose" begin
    x = ConcreteRArray(collect(reshape(1:12, 2, 2, 3)))
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

#     x = ConcreteRArray([1, 2, 3, 4])
#     @test sum(Array(x)) ≈ @jit f1(x)

#     x = ConcreteRArray([1 2; 3 4])
#     @test sum(Array(x)) ≈ @jit f4(x)
#     @test Base.transpose(Array(x)) ≈ @jit f3(x)
#     @test sum(Array(x); dims=1) ≈ @jit f4(x)
#     @test sum(Array(x); dims=2) ≈ @jit f5(x)
#     @test diag(Array(x)) ≈ @jit f6(x)
# end

@testset "xor" begin
    a = ConcreteRArray([false, false, true, true])
    b = ConcreteRArray([false, true, false, true])
    @test [false, true, true, false] ≈ @jit Ops.xor(a, b)

    a = ConcreteRArray([1, 2, 3, 4])
    b = ConcreteRArray([5, 6, -7, -8])
    @test Array(a) .⊻ Array(b) == @jit Ops.xor(a, b)
end

@testset "acos" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test acos.(Array(x)) ≈ @jit Ops.acos(x)
end

@testset "acosh" begin
    x = ConcreteRArray([1.0, 10.0])
    @test acosh.(Array(x)) ≈ @jit Ops.acosh(x)
end

@testset "asin" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test asin.(Array(x)) ≈ @jit Ops.asin(x)
end

@testset "asinh" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test asinh.(Array(x)) ≈ @jit Ops.asinh(x)
end

@testset "atan" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test atan.(Array(x)) ≈ @jit Ops.atan(x)
end

@testset "atanh" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test atanh.(Array(x)) ≈ @jit Ops.atanh(x)
end

@testset "bessel_i1e" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    @test SpecialFunctions.besselix.(1, Array(x)) ≈ @jit Ops.bessel_i1e(x)
end

@testset "conj" begin
    x = ConcreteRArray([-1.0 + 2im, 0.0 - 1im, 1.0 + 4im])
    @test conj(Array(x)) ≈ @jit Ops.conj(x)
end

@testset "cosh" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test cosh.(Array(x)) ≈ @jit Ops.cosh(x)
end

@testset "digamma" begin
    # small divergence between chlo.digamma and SpecialFunctions.digamma:
    # on <=0, chlo.digamma returns NaN, SpecialFunctions.digamma returns Inf
    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([-1.0, 0.0, 1.0])
        @test [NaN, NaN, SpecialFunctions.digamma(1.0)] ≈ @jit(Ops.digamma(x)) nans = true
    end
end

@testset "erf_inv" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test SpecialFunctions.erfinv.(Array(x)) ≈ @jit Ops.erf_inv(x)
end

@testset "erf" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test SpecialFunctions.erf.(Array(x)) ≈ @jit Ops.erf(x)
end

@testset "erfc" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test SpecialFunctions.erfc.(Array(x)) ≈ @jit Ops.erfc(x)
end

@testset "is_inf" begin
    x = ConcreteRArray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [true, true, false, false, false, false, false] ≈ @jit Ops.is_inf(x)
end

@testset "is_neg_inf" begin
    x = ConcreteRArray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [true, false, false, false, false, false, false] ≈ @jit Ops.is_neg_inf(x)
end

@testset "is_pos_inf" begin
    x = ConcreteRArray([-Inf, Inf, NaN, -10.0, -0.0, 0.0, 10.0])
    @test [false, true, false, false, false, false, false] ≈ @jit Ops.is_pos_inf(x)
end

@testset "lgamma" begin
    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([-1.0, 0.0, 1.0, 2.5])
        @test SpecialFunctions.lgamma.(Array(x)) ≈ @jit Ops.lgamma(x)
    end
end

@testset "next_after" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0, 1.0, 2.5, 1e18, 1e18, 3e-9, 3e-9])
    y = ConcreteRArray([-2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1e19, 0, 1])
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
    ] == @jit Ops.next_after(x, y)
end

@testset "polygamma" begin
    if !(Sys.isapple() && Sys.ARCH === :x86_64)
        x = ConcreteRArray([-1.0, 0.0, 1.0, 1.0, 2.5])
        m = ConcreteRArray([3.0, 3.0, 2.0, 3.0, 4.0])
        @test SpecialFunctions.polygamma.(Int.(Array(m)), Array(x)) ≈
            @jit Ops.polygamma(m, x)
    end
end

@testset "sinh" begin
    x = ConcreteRArray([-1.0, 0.0, 1.0])
    @test sinh.(Array(x)) ≈ @jit Ops.sinh(x)
end

@testset "top_k" begin
    x = ConcreteRArray([1, 2, 3, 4])
    @test (; values=[4, 3], indices=[3, 2]) == @jit Ops.top_k(x, 2)
end

@testset "zeta" begin
    s = ConcreteRArray([1.0, 2.0, 50.0])
    z = ConcreteRArray([1e-8, 0.001, 2.0])
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
    x = Reactant.to_rarray(randn(Float32, 3))
    y = Reactant.to_rarray(randn(Float32, 3))
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
