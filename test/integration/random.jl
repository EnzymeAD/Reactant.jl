using Reactant, Test, Random, Random123, StableRNGs, Statistics
using StatsBase, Statistics, HypothesisTests, Distributions

# First Testing overlay works correctly
@testset "Random.jl Overlay" begin
    hlo = @code_hlo rand(Float32, 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    hlo = @code_hlo rand(MersenneTwister(), Float32, 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    hlo = @code_hlo rand(2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    hlo = @code_hlo rand(MersenneTwister(), 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    hlo = @code_hlo rand(MersenneTwister(), Float64, (2, 3))
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    hlo = @code_hlo rand(MersenneTwister(), Float64)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    hlo = @code_hlo rand(MersenneTwister())
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    fn(x) = begin
        # XXX: MersenneTwister without seed leads to illegal instructions
        rng = MersenneTwister(0)
        Random.rand!(rng, x)
        return x
    end
    hlo = @code_hlo fn(Reactant.to_rarray(rand(Float64, 2, 3)))
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    fn2() = begin
        # XXX: MersenneTwister without seed leads to illegal instructions
        rng = MersenneTwister(0)
        x = zeros(Float64, 2, 3)
        Random.rand!(rng, x)
        return x
    end
    hlo = @code_hlo fn2()
    @test !contains(repr(hlo), "stablehlo.rng_bit_generator")
end

@testset "Random123" begin
    hlo = @code_hlo rand(Random123.Threefry4x(), Float32, 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")
    @test contains(repr(hlo), "THREE_FRY")

    hlo = @code_hlo rand(Random123.Threefry2x(), Float64, 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")
    @test contains(repr(hlo), "THREE_FRY")

    hlo = @code_hlo rand(Random123.Philox4x(), Float64, 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")
    @test contains(repr(hlo), "PHILOX")

    hlo = @code_hlo rand(Random123.Philox2x(), Float64, 2, 3)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")
    @test contains(repr(hlo), "PHILOX")
end

# Next we test that the random number generators actually generate data from the correct
# distributions
@testset "Uniform Random" begin
    @testset "Deterministic Seed" begin
        seed1 = Reactant.to_rarray(UInt64[1, 3])
        seed2 = Reactant.to_rarray(UInt64[1, 5])

        fn(seed) = begin
            rng = Random.default_rng()
            Random.seed!(rng, seed)
            return rand(rng, 10000)
        end

        fn_compiled = @compile fn(seed1)
        @test fn_compiled(seed1) ≈ fn_compiled(seed1)
        @test !(all(Array(fn_compiled(seed1)) .≈ Array(fn_compiled(seed2))))
    end

    @testset "Correct Distribution" begin
        X = Array(@jit(rand(StableRNG(0), 10000)))
        ks_test = ExactOneSampleKSTest(X, Uniform(0.0, 1.0))
        @test pvalue(ks_test) > 0.05
    end

    @testset "AutoCorrelation" begin
        X = Array(@jit(rand(StableRNG(0), 10000)))
        autocorr = cor(X[1:(end - 1)], X[2:end])
        @test abs(autocorr) < 0.05
    end

    @testset "Correct Range" begin
        X = Array(@jit(rand(StableRNG(0), 10000)))
        X_min, X_max = extrema(X)
        @test X_min ≥ 0.0
        @test X_max ≤ 1.0
    end

    @testset "Mean & Variance" begin
        X = Array(@jit(rand(StableRNG(0), 10000)))
        μ = mean(X)
        σ² = var(X)
        @test μ ≈ 0.5 atol = 0.05 rtol = 0.05
        @test σ² ≈ (1//12) atol = 0.05 rtol = 0.05
    end
end

@testset "Normal Distribution" begin
    @testset "Deterministic Seed" begin
        seed1 = Reactant.to_rarray(UInt64[1, 3])
        seed2 = Reactant.to_rarray(UInt64[1, 5])

        fn(seed) = begin
            rng = Random.default_rng()
            Random.seed!(rng, seed)
            return randn(rng, 10000)
        end

        fn_compiled = @compile fn(seed1)
        @test fn_compiled(seed1) ≈ fn_compiled(seed1)
        @test !(all(Array(fn_compiled(seed1)) .≈ Array(fn_compiled(seed2))))
    end

    @testset "Correct Distribution" begin
        X = Array(@jit(randn(StableRNG(0), 10000)))
        sw_test = ShapiroWilkTest(X)
        @test pvalue(sw_test) > 0.05
    end

    @testset "AutoCorrelation" begin
        X = Array(@jit(randn(StableRNG(0), 10000)))
        autocorr = cor(X[1:(end - 1)], X[2:end])
        @test abs(autocorr) < 0.05
    end

    @testset "Mean & Variance" begin
        X = Array(@jit(randn(StableRNG(0), 10000)))
        μ = mean(X)
        σ² = var(X)
        @test μ ≈ 0.0 atol = 0.05 rtol = 0.05
        @test σ² ≈ 1.0 atol = 0.05 rtol = 0.05
    end
end

@testset "Exponential Distribution" begin
    @testset "Deterministic Seed" begin
        seed1 = Reactant.to_rarray(UInt64[1, 3])
        seed2 = Reactant.to_rarray(UInt64[1, 5])

        fn(seed) = begin
            rng = Random.default_rng()
            Random.seed!(rng, seed)
            return randexp(rng, 10000)
        end

        fn_compiled = @compile fn(seed1)
        @test fn_compiled(seed1) ≈ fn_compiled(seed1)
        @test !(all(Array(fn_compiled(seed1)) .≈ Array(fn_compiled(seed2))))
    end

    @testset "Correct Distribution" begin
        X = Array(@jit(randexp(StableRNG(0), 10000)))
        ks_test = ExactOneSampleKSTest(X, Exponential(1.0))
        @test pvalue(ks_test) > 0.05
    end

    @testset "AutoCorrelation" begin
        X = Array(@jit(randexp(StableRNG(0), 10000)))
        autocorr = cor(X[1:(end - 1)], X[2:end])
        @test abs(autocorr) < 0.05
    end

    @testset "Correct Range" begin
        X = Array(@jit(randexp(StableRNG(0), 10000)))
        X_min, X_max = extrema(X)
        @test X_min ≥ 0.0
    end

    @testset "Mean" begin
        X = Array(@jit(randexp(StableRNG(0), 10000)))
        μ = mean(X)
        @test μ ≈ 1.0 atol = 0.05 rtol = 0.05
    end
end

rand_sample(rng, x) = rand(rng, eltype(x), size(x))

function rand_on_device()
    x = Reactant.@opcall fill(0.0f0, (3, 4, 5))
    rand!(x)
    return x
end

@testset "TracedTypes in Sampling" begin
    @test @jit(rand_sample(Reactant.ReactantRNG(), rand(3, 4))) isa
        ConcreteRArray{Float64,2}
    @test @jit(rand_sample(Reactant.ReactantRNG(), Reactant.to_rarray(rand(3, 4)))) isa
        ConcreteRArray{Float64,2}
    @test @jit(rand_on_device()) isa ConcreteRArray{Float32,3}
end

@testset "Tracing of Random" begin
    struct RandomContainer{RNG}
        rng::RNG
    end

    rng_st = RandomContainer(MersenneTwister(0))
    rng_st_ra = Reactant.to_rarray(rng_st)
    @test rng_st_ra.rng isa Reactant.ReactantRNG

    fn(st) = rand(st.rng, 10000)

    hlo = @code_hlo fn(rng_st_ra)
    @test contains(repr(hlo), "stablehlo.rng_bit_generator")

    @testset "natively supported RNGs" begin
        rng = Threefry2x()
        rng_ra = Reactant.to_rarray(rng)
        @test rng_ra isa Reactant.ReactantRNG
        @test rng_ra.seed ≈ UInt64[rng.key1, rng.key2]
        @test rng_ra.algorithm == "THREE_FRY"

        rng = Philox2x()
        rng_ra = Reactant.to_rarray(rng)
        @test rng_ra isa Reactant.ReactantRNG
        @test rng_ra.seed ≈ UInt64[rng.ctr1, rng.ctr2, rng.key]
        @test rng_ra.algorithm == "PHILOX"
    end
end
