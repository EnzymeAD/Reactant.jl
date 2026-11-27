using Reactant, Test, Random, StableRNGs, Statistics, FileCheck

sampler_hlo_unit_range() = rand(1:10, 4, 5)
sampler_hlo_step_range() = rand(1:2:11, 4, 5)
sampler_hlo_vec() = rand(Float32[10, 20, 30], 4, 5)
sampler_hlo_tuple3() = rand((10, 20, 30), 4, 5)
sampler_hlo_bool() = rand(Bool, 4, 5)

function sampler_hlo_rand_bang!(x)
    rng = MersenneTwister(0)
    Random.rand!(rng, x, 1:10)
    return x
end

function sampler_hlo_host_rand_bang()
    rng = MersenneTwister(0)
    x = zeros(Int, 2, 3)
    Random.rand!(rng, x, 1:10)
    return x
end

function sampler_explicit_scalar_and_array(seed)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    sp_f32 = Random.Sampler(rng, Float32)
    sp_i32 = Random.Sampler(rng, Int32)
    sp_rng = Random.Sampler(rng, -5:5)
    sp_arr = Random.Sampler(rng, Float32[1.5, 2.5, 3.5])
    sp_co12 = Random.Sampler(rng, Random.CloseOpen12(Float64))
    s1 = rand(rng, sp_f32)
    s2 = rand(rng, sp_i32)
    s3 = rand(rng, sp_rng)
    s4 = rand(rng, sp_arr)
    s5 = rand(rng, sp_co12)
    a1 = rand(rng, sp_f32, 3, 4)
    a2 = rand(rng, sp_rng, (3, 4))
    a3 = rand(rng, sp_arr, 3, 4)
    a4 = rand(rng, sp_co12, 16)
    return s1, s2, s3, s4, s5, a1, a2, a3, a4
end

function sampler_rand_bang_convert!(buf_f32, buf_i64)
    rng = StableRNG(7)
    Random.rand!(rng, buf_f32, 1:5)
    Random.rand!(buf_i64, Int64[100, 200, 300])
    return buf_f32, buf_i64
end

sampler_unit_range_i64() = rand(StableRNG(1), 1:4, 4000)
sampler_unit_range_i8() = rand(StableRNG(2), Int8(-5):Int8(5), 256)
sampler_unit_range_u8() = rand(StableRNG(3), 0x00:0xff, 256)
sampler_unit_range_oneto() = rand(Base.OneTo(7), 128)
sampler_unit_range_bool() = rand(false:true, 256)
sampler_unit_range_full_u32() = rand(0x00000000:0xffffffff, 64)
sampler_unit_range_full_i64() = rand(typemin(Int64):typemax(Int64), 64)
sampler_traced_unit_range(a, b) = rand(a:b, 256)
sampler_traced_unit_range_scalar(a, b) = rand(a:b)

sampler_step_range_pos() = rand(1:2:11, 256)
sampler_step_range_neg() = rand(10:-3:1, 256)
sampler_step_range_huge() = rand(1:2:10_000_000_001, 32)
sampler_step_range_len(r) = rand(r, 128)
sampler_lin_range() = rand(LinRange(1.0f0, 5.0f0, 5), 128)

sampler_traced_vec(v) = rand(v, 128)
sampler_tuple1() = rand((42,), 8)
sampler_tuple2() = rand((10, 20), 128)
sampler_tuple3() = rand((10, 20, 30), 256)
sampler_tuple4() = rand((1.0f0, 2.0f0, 3.0f0, 4.0f0), 128)
sampler_tuple_scalar() = rand((2, 3))
sampler_traced_tuple(x, y, z) = rand((x, y, z), 128)
sampler_set() = rand(Set([10, 20, 30]), 128)
sampler_bitset() = rand(BitSet([2, 4, 6]), 128)

sampler_bool_scalar() = rand(Bool)
sampler_bool_array() = rand(StableRNG(42), Bool, 2000)

function sampler_tuple_type()
    rng = Random.default_rng()
    return rand(rng, Random.Sampler(rng, Tuple{Float32,Int32}))
end

function sampler_ntuple_type()
    rng = Random.default_rng()
    return rand(rng, Random.Sampler(rng, NTuple{3,Float32}))
end

function sampler_pair_type()
    rng = Random.default_rng()
    return rand(rng, Random.Sampler(rng, Pair{Float32,Int64}))
end

function sampler_seeded_range(seed)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    return rand(rng, 10:20, 64)
end

function check_all_in_set(arr, allowed)
    for x in arr
        x in allowed || return false
    end
    return true
end

@testset "Random.Sampler API (#2256)" begin
    @testset "Single rng_bit_generator without dynamic_update_slice" begin
        for fn in (
            sampler_hlo_unit_range,
            sampler_hlo_step_range,
            sampler_hlo_vec,
            sampler_hlo_tuple3,
            sampler_hlo_bool,
        )
            hlo = repr(@code_hlo fn())
            @test @filecheck begin
                @check "stablehlo.rng_bit_generator"
                @check_not "stablehlo.rng_bit_generator"
                hlo
            end
            @test !occursin("dynamic_update_slice", hlo)
        end

        x_dev = Reactant.to_rarray(zeros(Int, 2, 3))
        hlo_bang = repr(@code_hlo sampler_hlo_rand_bang!(x_dev))
        @test occursin("stablehlo.rng_bit_generator", hlo_bang)
        @test !occursin("dynamic_update_slice", hlo_bang)

        hlo_host = repr(@code_hlo sampler_hlo_host_rand_bang())
        @test !occursin("stablehlo.rng_bit_generator", hlo_host)
    end

    @testset "Explicit Sampler objects and in-place rand!" begin
        seed = Reactant.to_rarray(UInt64[11, 29])
        s1, s2, s3, s4, s5, a1, a2, a3, a4 = @jit sampler_explicit_scalar_and_array(seed)
        @test 0.0f0 <= Reactant.to_number(s1) < 1.0f0
        @test Reactant.to_number(s2) isa Int32
        @test -5 <= Reactant.to_number(s3) <= 5
        @test Reactant.to_number(s4) in (1.5f0, 2.5f0, 3.5f0)
        @test 1.0 <= Reactant.to_number(s5) < 2.0

        a1_h = Array(a1)
        a2_h = Array(a2)
        a3_h = Array(a3)
        a4_h = Array(a4)
        @test size(a1_h) == (3, 4) && eltype(a1_h) == Float32
        @test all(0.0f0 .<= a1_h .< 1.0f0)
        @test size(a2_h) == (3, 4) && eltype(a2_h) == Int
        @test all(-5 .<= a2_h .<= 5)
        @test check_all_in_set(a3_h, (1.5f0, 2.5f0, 3.5f0))
        @test all(1.0 .<= a4_h .< 2.0)

        buf_f32 = Reactant.to_rarray(zeros(Float32, 3, 4))
        buf_i64 = Reactant.to_rarray(zeros(Int64, 3, 4))
        @jit sampler_rand_bang_convert!(buf_f32, buf_i64)
        @test check_all_in_set(Array(buf_f32), (1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0))
        @test check_all_in_set(Array(buf_i64), (100, 200, 300))
    end

    @testset "Unit ranges and traced unit ranges" begin
        u_i64 = Array(@jit sampler_unit_range_i64())
        @test eltype(u_i64) == Int64
        @test all(1 .<= u_i64 .<= 4)
        for v in 1:4
            cnt = count(==(v), u_i64)
            @test 800 <= cnt <= 1200
        end

        u_i8 = Array(@jit sampler_unit_range_i8())
        @test eltype(u_i8) == Int8
        @test all(Int8(-5) .<= u_i8 .<= Int8(5))
        @test minimum(u_i8) == Int8(-5)
        @test maximum(u_i8) == Int8(5)

        u_u8 = Array(@jit sampler_unit_range_u8())
        @test eltype(u_u8) == UInt8 && length(u_u8) == 256

        u_oneto = Array(@jit sampler_unit_range_oneto())
        @test all(1 .<= u_oneto .<= 7)

        u_bool = Array(@jit sampler_unit_range_bool())
        @test eltype(u_bool) == Bool && any(u_bool) && any(.!u_bool)

        u_full32 = Array(@jit sampler_unit_range_full_u32())
        @test eltype(u_full32) == UInt32 && length(unique(u_full32)) > 50

        u_full64 = Array(@jit sampler_unit_range_full_i64())
        @test eltype(u_full64) == Int64 && length(unique(u_full64)) > 50

        a_c = ConcreteRNumber(3)
        b_c = ConcreteRNumber(9)
        u_tr = Array(@jit sampler_traced_unit_range(a_c, b_c))
        @test all(3 .<= u_tr .<= 9)
        @test minimum(u_tr) == 3 && maximum(u_tr) == 9
        s_tr = Reactant.to_number(@jit sampler_traced_unit_range_scalar(a_c, b_c))
        @test 3 <= s_tr <= 9
    end

    @testset "Step ranges, float ranges, arrays, tuples, sets, Bool, and Tuple/Pair types" begin
        @test check_all_in_set(Array(@jit sampler_step_range_pos()), (1, 3, 5, 7, 9, 11))
        @test check_all_in_set(Array(@jit sampler_step_range_neg()), (10, 7, 4, 1))
        huge = Array(@jit sampler_step_range_huge())
        @test all(isodd, huge) && all(1 .<= huge .<= 10_000_000_001)

        srl = Array(@jit sampler_step_range_len(0.0:0.25:1.0))
        @test check_all_in_set(srl, (0.0, 0.25, 0.5, 0.75, 1.0))

        lr = Array(@jit sampler_lin_range())
        @test check_all_in_set(lr, (1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0))

        v_dev = Reactant.to_rarray(Int32[11, 22, 33])
        @test check_all_in_set(
            Array(@jit sampler_traced_vec(v_dev)), (Int32(11), Int32(22), Int32(33))
        )

        @test all(==(42), Array(@jit sampler_tuple1()))
        @test check_all_in_set(Array(@jit sampler_tuple2()), (10, 20))
        t3 = Array(@jit sampler_tuple3())
        @test check_all_in_set(t3, (10, 20, 30))
        @test length(unique(t3)) == 3
        @test check_all_in_set(Array(@jit sampler_tuple4()), (1.0f0, 2.0f0, 3.0f0, 4.0f0))
        @test Reactant.to_number(@jit sampler_tuple_scalar()) in (2, 3)

        t_tr = Array(
            @jit sampler_traced_tuple(
                ConcreteRNumber(5), ConcreteRNumber(15), ConcreteRNumber(25)
            )
        )
        @test check_all_in_set(t_tr, (5, 15, 25))
        @test length(unique(t_tr)) == 3

        @test check_all_in_set(Array(@jit sampler_set()), (10, 20, 30))
        @test check_all_in_set(Array(@jit sampler_bitset()), (2, 4, 6))

        @test Reactant.to_number(@jit sampler_bool_scalar()) isa Bool
        b_arr = Array(@jit sampler_bool_array())
        @test eltype(b_arr) == Bool
        @test 0.4 <= mean(b_arr) <= 0.6

        tt = @jit sampler_tuple_type()
        @test 0.0f0 <= Reactant.to_number(tt[1]) < 1.0f0
        @test Reactant.to_number(tt[2]) isa Int32

        nt = @jit sampler_ntuple_type()
        @test length(nt) == 3
        @test 0.0f0 <= Reactant.to_number(nt[1]) < 1.0f0

        pt = @jit sampler_pair_type()
        @test 0.0f0 <= Reactant.to_number(pt.first) < 1.0f0
        @test Reactant.to_number(pt.second) isa Int64
    end

    @testset "Deterministic seeding" begin
        seed1 = Reactant.to_rarray(UInt64[7, 13])
        seed2 = Reactant.to_rarray(UInt64[7, 19])
        fn_c = @compile sampler_seeded_range(seed1)
        @test Array(fn_c(seed1)) == Array(fn_c(seed1))
        @test Array(fn_c(seed1)) != Array(fn_c(seed2))
    end
end
