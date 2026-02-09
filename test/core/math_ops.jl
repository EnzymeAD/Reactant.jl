# Tests for mathematical operations
using Reactant, Test

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

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

bcast_cos(x) = cos.(x)

@testset "Basic cos" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 3, 2)
    c = Reactant.to_rarray(x)

    @test @jit(bcast_cos(c)) ≈ cos.(x)
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

@testset "rem2pi" begin
    a = [-1.1, 7.7, -3.3, 9.9, -5.5]

    @testset "$T" for (convfn, T) in [
        (identity, Float64), (x -> Float32.(x), Float32), (x -> floor.(Int32, x), Int32)
    ]
        if RunningOnTPU
            @warn "Skipping rem2pi test on TPU. F64 bitcast not supported on TPU"
            break
        end

        @testset for round_mode in
                     (Base.RoundUp, Base.RoundDown, Base.RoundNearest, Base.RoundToZero)
            a_ = convfn(a)
            expected_mod2pi = rem2pi.(a_, round_mode)
            reactant_mod2pi = @jit(rem2pi.(Reactant.to_rarray(a_), round_mode))
            @test reactant_mod2pi ≈ expected_mod2pi
            @test Reactant.unwrapped_eltype(reactant_mod2pi) == eltype(expected_mod2pi)
        end
    end
end

@testset "mod2pi" begin
    a = [-1.1, 7.7, -3.3, 9.9, -5.5]

    @testset "$T" for (convfn, T) in [
        (identity, Float64), (x -> Float32.(x), Float32), (x -> floor.(Int32, x), Int32)
    ]
        if RunningOnTPU
            @warn "Skipping rem2pi test on TPU. F64 bitcast not supported on TPU"
            break
        end

        a_ = convfn(a)
        expected_mod2pi = mod2pi.(a_)
        reactant_mod2pi = @jit(mod2pi.(Reactant.to_rarray(a_)))
        @test reactant_mod2pi ≈ expected_mod2pi
        @test Reactant.unwrapped_eltype(reactant_mod2pi) == eltype(expected_mod2pi)
    end
end

@testset "xor" begin
    for a in (true, false), b in (true, false)
        @test @jit(xor(ConcreteRNumber(a), ConcreteRNumber(b))) == xor(a, b)
    end

    for (a,b) in Iterators.product((3, 0), (true, false))
        at = Reactant.to_rarray(a; track_numbers=Number)
        bt = Reactant.to_rarray(b; track_numbers=Number)
    
        @test @jit(xor(at, b)) == xor(a, b) 
        @test @jit(xor(a, bt)) == xor(a, b)
        @test @jit(xor(at, bt)) == xor(a, b)
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

        @test Reactant.to_number(@jit(copysign(ConcreteRNumber(a), ConcreteRNumber(b)))) ≈
            copysign(a, b) broken = RunningOnTPU && eltype(b) == Float64
    end
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

@testset "sign" begin
    x = collect(Float64, 0:0.01:1) .- 0.5
    x_ra = Reactant.to_rarray(x)
    @test Array(@jit(sign.(x_ra))) ≈ sign.(x)
end

@testset "/ on integers" begin
    @test @jit(/(ConcreteRNumber(2), ConcreteRNumber(4))) ≈ 0.5
    @test @jit(/(ConcreteRNumber(2), 4)) ≈ 0.5
    @test @jit(/(2, ConcreteRNumber(4))) ≈ 0.5
    @test @jit(/(2, ConcreteRNumber(Int32(4)))) ≈ 0.5
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

@testset "round with RoundToZero" begin
    # Test with Float32 arrays
    x32 = Float32[-2.7, -2.3, -1.5, -0.9, 0.0, 0.9, 1.5, 2.3, 2.7]
    x32_ra = Reactant.to_rarray(x32)
    roundtozero32(arr) = round.(arr, RoundToZero)
    result32 = @jit(roundtozero32(x32_ra))
    expected32 = round.(x32, RoundToZero)
    @test result32 ≈ expected32
    @test result32 isa ConcreteRArray{Float32,1}
    @test sign.(Array(result32)) == sign.(expected32)

    # Test with Float64 arrays
    if !RunningOnTPU
        x64 = Float64[-2.7, -2.3, -1.5, -0.9, 0.0, 0.9, 1.5, 2.3, 2.7]
        x64_ra = Reactant.to_rarray(x64)
        roundtozero64(arr) = round.(arr, RoundToZero)
        result64 = @jit(roundtozero64(x64_ra))
        expected64 = round.(x64, RoundToZero)
        @test result64 ≈ expected64
        @test result64 isa ConcreteRArray{Float64,1}
        @test sign.(Array(result64)) == sign.(expected64)
    end

    # Test with scalar TracedRNumber (Float32)
    s32 = -3.7f0
    s32_ra = Reactant.to_rarray(s32; track_numbers=Number)
    roundtozero_scalar(x) = round(x, RoundToZero)
    result_s32 = @jit(roundtozero_scalar(s32_ra))
    expected_s32 = round(s32, RoundToZero)
    @test result_s32 ≈ expected_s32
    @test result_s32 isa ConcreteRNumber{Float32}
    @test sign(Reactant.to_number(result_s32)) == sign(expected_s32)

    # Test with scalar TracedRNumber (Float64)
    if !RunningOnTPU
        s64 = 5.9
        s64_ra = Reactant.to_rarray(s64; track_numbers=Number)
        result_s64 = @jit(roundtozero_scalar(s64_ra))
        expected_s64 = round(s64, RoundToZero)
        @test result_s64 ≈ expected_s64
        @test result_s64 isa ConcreteRNumber{Float64}
        @test sign(Reactant.to_number(result_s64)) == sign(expected_s64)
    end

    # Test edge case: values already integers
    already_int = Float32[-3.0, 0.0, 4.0]
    already_int_ra = Reactant.to_rarray(already_int)
    result_int = @jit(roundtozero32(already_int_ra))
    expected_int = round.(already_int, RoundToZero)
    @test result_int ≈ expected_int
    @test sign.(Array(result_int)) == sign.(expected_int)

    # Test edge case: very small values close to zero
    small_vals = Float32[-0.1, -0.001, 0.001, 0.1]
    small_vals_ra = Reactant.to_rarray(small_vals)
    result_small = @jit(roundtozero32(small_vals_ra))
    expected_small = round.(small_vals, RoundToZero)
    @test result_small ≈ expected_small
    @test sign.(Array(result_small)) == sign.(expected_small)
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

@testset "clamp!" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 32, 32)
    x_ra = Reactant.to_rarray(x)
    @test @jit(clamp!(x_ra, 0.5, Inf32)) ≈ clamp!(x, 0.5, Inf32)
end

mulpi(x) = π * x

@testset "Irrational promotion" begin
    x = Reactant.to_rarray(ones(2))
    y = @jit mulpi(x)
    @test all(Array(y) .≈ π)
end
