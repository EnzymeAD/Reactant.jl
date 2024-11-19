using Reactant, Test
using Reactant: Ops

@testset "abs" begin
    x = ConcreteRArray([1, -1])
    @test [1, 1] ≈ @jit Ops.abs(x)

    x = ConcreteRArray([1.0, -1.0])
    @test [1.0, 1.0] ≈ @jit Ops.abs(y)

    x = ConcreteRArray([3.0 + 4im, -3.0 + 4im; 3.0 - 4im, -3.0 - 4im])
    @test [5.0, 5.0; 5.0, 5.0] ≈ @jit Ops.abs(y)
end

@testset "add" begin
    a = ConcreteRArray([false, false, true, true])
    b = ConcreteRArray([false, true, false, true])
    @test [false, true, true, true] ≈ @jit Ops.add(a, b)

    a = ConcreteRArray([1, 2, 3, 4])
    b = ConcreteRArray([5, 6, -7, -8])
    @test [6, 8, -4, -4] == @jit Ops.add(a, b)

    a = ConcreteRArray([1.1, 2.2, 3.3, 4.4])
    b = ConcreteRArray([5.5, 6.6, -7.7, -8.8])
    @test [6.6, 8.8, -4.4, -4.4] ≈ @jit Ops.add(a, b)

    a = ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    b = ConcreteRArray([
        9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im
    ])
    @test [11.0 + 12.2im, 14.4 + 16.52im, -7.63 + -7.54im, -7.38 + -7.36im] ≈
        @jit Ops.add(a, b)
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
    @test [1, 2, 1, 1] == @jit Ops.and(a, b)
end

@testset "atan2" begin
    a = ConcreteRArray([1, 2, 3, 4])
    b = ConcreteRArray([5, 6, -7, -8])
    @test atan.(Array(a), Array(b)) ≈ @jit Ops.atan2(a, b)

    a = ConcreteRArray([1.1, 2.2, 3.3, 4.4])
    b = ConcreteRArray([5.5, 6.6, -7.7, -8.8])
    @test atan.(Array(a), Array(b)) ≈ @jit Ops.atan2(a, b)

    a = ConcreteRArray([1.1 + 2.2im, 3.3 + 4.4im, 5.5 + 6.6im, 7.7 + 8.8im])
    b = ConcreteRArray([
        9.9 + 10.10im, 11.11 + 12.12im, -13.13 + -14.14im, -15.15 + -16.16im
    ])
    @test atan.(Array(a), Array(b)) ≈ @jit Ops.atan2(a, b)
end

@testset "cbrt" begin
    x = ConcreteRArray([1.0, 8.0, 27.0, 64.0])
    @test [1.0, 2.0, 3.0, 4.0] ≈ @jit Ops.cbrt(x)

    x = ConcreteRArray([1.0 + 2.0im, 8.0 + 16.0im, 27.0 + 54.0im, 64.0 + 128.0im])
    @test Array(x) .^ (1//3) ≈ @jit Ops.cbrt(x)
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
    g(x) = Ops.cholesky(x; lower=true)
    x = ConcreteRArray([
        10.0 2.0 3.0
        2.0 5.0 6.0
        3.0 6.0 9.0
    ])
    @test cholesky(Array(x)).U ≈ @jit Ops.cholesky(x)
    @test transpose(cholesky(Array(x)).U) ≈ @jit g(x)

    x = ConcreteRArray(
        [
            10.0+0.0im 2.0-3.0im 3.0-4.0im
            2.0+3.0im 5.0+0.0im 3.0-2.0im
            3.0+4.0im 3.0+2.0im 9.0+0.0im
        ],
    )
    @test cholesky(Array(x)).U ≈ @jit Ops.cholesky(x)
    @test transpose(cholesky(Array(x)).U) ≈ @jit g(x)
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

        y = ConcreteRArray(x)
        @test x ≈ @jit Ops.constant(y)

        xscalar = x[1]
        @test xscalar ≈ @jit Ops.constant(xscalar)

        yscalar = ConcreteRNumber(xscalar)
        @test xscalar ≈ @jit Ops.constant(yscalar)
    end
end

@testset "cosine" begin
    x = ConcreteRArray([0, π / 2, π, 3π / 2, 2π])
    @test [1, 0, -1, 0, 1] ≈ @jit Ops.cosine(x)

    x = ConcreteRArray([0.0, π / 2, π, 3π / 2, 2π])
    @test [1.0, 0.0, -1.0, 0.0, 1.0] ≈ @jit Ops.cosine(x)

    x = ConcreteRArray([0.0 + 0.0im, π / 2 + 0.0im, π + 0.0im, 3π / 2 + 0.0im, 2π + 0.0im])
    @test [1.0 + 0.0im, 0.0 + 0.0im, -1.0 + 0.0im, 0.0 + 0.0im, 1.0 + 0.0im] ≈
        @jit Ops.cosine(x)
end

@testset "count_leading_zeros" begin
    x = ConcreteRArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    @test [64, 63, 62, 61, 60, 59, 58, 57, 56, 55] ≈ @jit Ops.count_leading_zeros(x)
end

@testset "divide" begin
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
        @test Array(a) ./ Array(b) ≈ @jit Ops.divide(a, b)
    end
end

@testset "einsum" begin
    for (a, b) in [
        (ConcreteRArray([1, 2, 3, 4]), ConcreteRArray([5, 6, -7, -8])),
        (ConcreteRArray([1.0, 2.0, 3.0, 4.0]), ConcreteRArray([5.0, 6.0, -7.0, -8.0])),
        (
            ConcreteRArray([1 + 1im, 2 + 2im, 3 - 3im, 4 - 4im]),
            ConcreteRArray([5 + 5im, 6 + 6im, -7 - 7im, -8 - 8im]),
        ),
        (
            ConcreteRArray([1.0 + 1im, 2.0 + 2im, 3.0 - 3im, 4.0 - 4im]),
            ConcreteRArray([5.0 + 5im, 6.0 + 6im, -7.0 - 7im, -8.0 - 8im]),
        ),
    ]
        @test a .* b ≈ @jit Ops.einsum("i,i->i", a, b)
        @test kron(Array(a), Array(b)) ≈ @jit Ops.einsum("i,j->ij", a, b)

        x = reshape(a, (2, 2))
        y = reshape(b, (2, 2))
        @test x .* y ≈ @jit Ops.einsum("ij,ij->ij", x, y)
        @test x * y ≈ @jit Ops.einsum("ik,kj->ij", x, y)
    end
end

@testset "exponential" begin
    x = ConcreteRArray([1.0, 2.0, 3.0, 4.0])
    @test exp.(Array(x)) ≈ @jit Ops.exponential(x)

    x = ConcreteRArray([1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im])
    @test exp.(Array(x)) ≈ @jit Ops.exponential(x)
end

@testset "exponential_minus_one" begin
    x = ConcreteRArray([1.0, 2.0, 3.0, 4.0])
    @test expm1.(Array(x)) ≈ @jit Ops.exponential_minus_one(x)

    x = ConcreteRArray([1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im])
    @test expm1.(Array(x)) ≈ @jit Ops.exponential_minus_one(x)
end

@testset "fft" begin
    x = ConcreteRArray([1.0, 1.0, 1.0, 1.0])
    @test ComplexF64[4.0, 0.0, 0.0] ≈ @jit Ops.fft(x, type="RFFT")

    x = ConcreteRArray([0.0, 1.0, 0.0, -1.0])
    @test ComplexF64[0.0, -2.0, 0.0] ≈ @jit Ops.fft(x, type="RFFT")

    x = ConcreteRArray([1.0, -1.0, 1.0, -1.0])
    @test ComplexF64[0.0, 0.0, 4.0] ≈ @jit Ops.fft(x, type="RFFT")

    x = ConcreteRArray([1.0, 1.0, 1.0, 1.0])
    @test ComplexF64[4.0, 0.0, 0.0, 0.0] ≈ @jit Ops.fft(x, type="FFT")

    x = ConcreteRArray([0.0, 1.0, 0.0, -1.0])
    @test ComplexF64[0.0, -2.0, 0.0, 2.0] ≈ @jit Ops.fft(x, type="FFT")

    x = ConcreteRArray([1.0, -1.0, 1.0, -1.0])
    @test ComplexF64[0.0, 0.0, 4.0, 0.0] ≈ @jit Ops.fft(x, type="FFT")

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
    @test [
        0 0 0 0 0
        1 1 1 1 1
        2 2 2 2 2
        3 3 3 3 3
    ] ≈ @jit Ops.iota(Int, [4, 5]; iota_dimension=1)

    @test [
        0 1 2 3 4
        0 1 2 3 4
        0 1 2 3 4
        0 1 2 3 4
    ] ≈ @jit Ops.iota(Int, [4, 5]; iota_dimension=2)
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

@testset "logistic" begin end

@testset "maximum" begin end

@testset "minimum" begin end

@testset "multiply" begin end

@testset "negate" begin end

@testset "not" begin end

@testset "optimization_barrier" begin end

@testset "or" begin end

@testset "outfeed" begin end

@testset "partition_id" begin end

@testset "popcnt" begin end

@testset "power" begin end

@testset "real" begin end

@testset "recv" begin end

@testset "remainder" begin end

@testset "replica_id" begin end

@testset "reshape" begin end

@testset "reverse" begin end

@testset "rng_bit_generator" begin end

@testset "round_nearest_even" begin end

@testset "round_nearest_afz" begin end

@testset "rsqrt" begin end

@testset "send" begin end

@testset "set_dimension_size" begin end

@testset "shift_left" begin end

@testset "shift_right_arithmetic" begin end

@testset "shift_right_logical" begin end

@testset "sign" begin end

@testset "sine" begin end

@testset "sort" begin end

@testset "sqrt" begin end

@testset "subtract" begin end

@testset "tan" begin end

@testset "tanh" begin end

@testset "transpose" begin end

@testset "unary_einsum" begin end

@testset "xor" begin end

@testset "acos" begin end

@testset "acosh" begin end

@testset "asin" begin end

@testset "asinh" begin end

@testset "atan" begin end

@testset "atanh" begin end

@testset "bessel_i1e" begin end

@testset "conj" begin end

@testset "cosh" begin end

@testset "digamma" begin end

@testset "erf_inv" begin end

@testset "erf" begin end

@testset "erfc" begin end

@testset "is_inf" begin end

@testset "is_neg_inf" begin end

@testset "is_pos_inf" begin end

@testset "lgamma" begin end

@testset "next_after" begin end

@testset "polygamma" begin end

@testset "sinh" begin end

@testset "tan" begin end

@testset "top_k" begin end

@testset "zeta" begin end
