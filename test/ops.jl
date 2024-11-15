using Reactant, Test
using Reactant: Ops

@testset "abs" begin
    x = ConcreteRArray([1.0, -1.0])
    @test [1.0, 1.0] ≈ @jit Ops.abs(y)

    x = ConcreteRArray([3.0 + 4im, -3.0 + 4im; 3.0 - 4im, -3.0 - 4im])
    @test [5.0, 5.0; 5.0, 5.0] ≈ @jit Ops.abs(y)
end

@testset "add" begin end

@testset "after_all" begin end

@testset "and" begin end

@testset "atan2" begin end

@testset "cbrt" begin end

@testset "ceil" begin end

@testset "cholesky" begin end

@testset "clamp" begin end

@testset "count_leading_zeros" begin end

@testset "complex" begin end

@testset "constant" begin end

@testset "cosine" begin end

@testset "divide" begin end

@testset "einsum" begin end

@testset "exponential" begin end

@testset "exponential_minus_one" begin end

@testset "fft" begin end

@testset "floor" begin end

@testset "get_dimension_size" begin end

@testset "imag" begin end

@testset "iota" begin end

@testset "is_finite" begin end

@testset "log_plus_one" begin end

@testset "log" begin end

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
