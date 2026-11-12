using Reactant, Test

@testset "ranges" begin
    i = Reactant.to_rarray(5; track_numbers=true)
    @test Array{Int64}(@jit(1:i)) == collect(1:5)
    @test Array{Int64}(@jit(i:10)) == collect(5:10)
    j = Reactant.to_rarray(10; track_numbers=true)
    @test Array{Int64}(@jit(i:j)) == collect(5:10)
end

broadcast_over_range(a, kx, ky) = a .* (kx .^ 2 .+ ky' .^ 2)

@testset "broadcast over ranges" begin
    a = Reactant.TestUtils.construct_test_array(Float32, 32, 32)
    a_ra = Reactant.to_rarray(a)
    kx = 1:32
    ky = 30:61
    kx_ra = Reactant.to_rarray(kx; track_numbers=true)
    ky_ra = Reactant.to_rarray(ky; track_numbers=true)

    @test kx_ra isa Reactant.TracedUnitRange
    @test ky_ra isa Reactant.TracedUnitRange
    @test @jit(broadcast_over_range(a_ra, kx_ra, ky_ra)) ≈ broadcast_over_range(a, kx, ky)

    kx_2 = 1.0f0:32.0f0
    ky_2 = 30.0f0:61.0f0
    kx_2_ra = Reactant.to_rarray(kx_2; track_numbers=true)
    ky_2_ra = Reactant.to_rarray(ky_2; track_numbers=true)

    @test kx_2_ra isa Reactant.TracedStepRangeLen
    @test ky_2_ra isa Reactant.TracedStepRangeLen
    @test @jit(broadcast_over_range(a_ra, kx_2_ra, ky_2_ra)) ≈
        broadcast_over_range(a, kx_2, ky_2)
end

@testset "TwicePrecision to ConcreteRNumber" begin
    tp = Base.TwicePrecision(2.5, 0.5)
    @test Reactant.to_number(ConcreteRNumber{Float64}(tp)) === Float64(tp)

    r = 0:0.1:1
    r_ra = Reactant.to_rarray(r; track_numbers=Number)
    # On TPU, Float64 buffers are stored as a pair of Float32 values (48-bit
    # mantissa), so 0.1 round-trips through a ConcreteRNumber as
    # 0.09999999999999998 rather than 0.1.
    RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")
    @test Reactant.to_number(step(r_ra)) === Reactant.to_number(ConcreteRNumber(step(r)))
    @test Reactant.to_number(step(r_ra)) === step(r) skip = RunningOnTPU
end

function _nested_trace_eachindex_2083!(out, _ar, k1, k2)
    @trace for i in eachindex(k1)
        @trace for j in eachindex(k2)
            Reactant.@allowscalar begin
                out[i, j] = k1[i] + k2[j]
            end
        end
    end
    return out
end

function _sum_mixed_int_colon_2083(i32)
    s = zero(Int64)
    @trace for k in i32:10
        s += k
    end
    return s
end

@testset "Base.oneto / eachindex on TracedStepRangeLen and mixed-int Colon (#2083)" begin
    ar = ConcreteRNumber(1.0)
    out = Reactant.to_rarray(zeros(8, 8))
    k1 = range(1.0, 4.0; length=8)
    k2 = range(2.0, 5.0; length=8)
    res = @jit _nested_trace_eachindex_2083!(out, ar, k1, k2)
    ref = zeros(8, 8)
    for i in eachindex(k1), j in eachindex(k2)
        ref[i, j] = k1[i] + k2[j]
    end
    @test Array(res) ≈ ref

    i32 = ConcreteRNumber(Int32(3))
    j64 = ConcreteRNumber(Int64(10))
    @test Array{Int64}(@jit(i32:10)) == collect(3:10)
    @test Array{Int64}(@jit(1:i32)) == collect(1:3)
    @test Array{Int64}(@jit(i32:j64)) == collect(3:10)
    @test Reactant.to_number(@jit _sum_mixed_int_colon_2083(i32)) == sum(3:10)
end
