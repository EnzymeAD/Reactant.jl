using Reactant, Test
using Reactant:
    TracedRArray, TracedRNumber, MLIR, TracedUtils, ConcreteRArray, ConcreteRNumber
using Reactant.MLIR: IR
using Reactant.MLIR.Dialects: enzyme
using Statistics

# `enzyme.randomSplit` op is not intended to be emitted directly in Reactant-land.
# It is solely an intermediate representation within the `enzyme.mcmc` op lowering.
function random_split(rng_state::TracedRArray{UInt64,1}, ::Val{N}) where {N}
    rng_mlir = TracedUtils.get_mlir_data(rng_state)
    rng_state_type = IR.TensorType([2], IR.Type(UInt64))
    output_types = [rng_state_type for _ in 1:N]
    op = enzyme.randomSplit(rng_mlir; output_rng_states=output_types)
    return ntuple(i -> TracedRArray{UInt64,1}((), IR.result(op, i), (2,)), Val(N))
end

@testset "enzyme.randomSplit op" begin
    @testset "N=2, Seed [0, 42]" begin
        seed = ConcreteRArray(UInt64[0, 42])
        k1, k2 = @jit optimize = :probprog random_split(seed, Val(2))

        @test Array(k1) == [0x99ba4efe6b200159, 0x4f6cc618de79f4b9]
        @test Array(k2) == [0xcddb151d375f238f, 0xf67a601be6bdada3]
    end

    @testset "N=2, Seed [42, 0]" begin
        seed = ConcreteRArray(UInt64[42, 0])
        k1, k2 = @jit optimize = :probprog random_split(seed, Val(2))

        @test Array(k1) == [0x4f6cc618de79f4b9, 0x99ba4efe6b200159]
        @test Array(k2) == [0xf67a601be6bdada3, 0xcddb151d375f238f]
    end

    @testset "N=3, Seed [0, 42]" begin
        seed = ConcreteRArray(UInt64[0, 42])
        k1, k2, k3 = @jit optimize = :probprog random_split(seed, Val(3))

        @test Array(k1) == [0x99ba4efe6b200159, 0x4f6cc618de79f4b9]
        @test Array(k2) == [0xcddb151d375f238f, 0xf67a601be6bdada3]
        @test Array(k3) == [0xa20e4081f71f4ea9, 0x2f36b83d4e83f1ba]
    end

    @testset "N=4, Seed [0, 42]" begin
        seed = ConcreteRArray(UInt64[0, 42])
        k1, k2, k3, k4 = @jit optimize = :probprog random_split(seed, Val(4))

        @test Array(k1) == [0x99ba4efe6b200159, 0x4f6cc618de79f4b9]
        @test Array(k2) == [0xcddb151d375f238f, 0xf67a601be6bdada3]
        @test Array(k3) == [0xa20e4081f71f4ea9, 0x2f36b83d4e83f1ba]
        @test Array(k4) == [0xe4e8dfbe9312778b, 0x982ff5502e6ccb51]
    end
end

# Similarly, `enzyme.random` op is not intended to be emitted directly in Reactant-land.
# It is solely an intermediate representation within the `enzyme.mcmc` op lowering.
function rng_distribution_attr(distribution::Int32)
    return @ccall MLIR.API.mlir_c.enzymeRngDistributionAttrGet(
        MLIR.IR.context()::MLIR.API.MlirContext, distribution::Int32
    )::MLIR.IR.Attribute
end

const RNG_UNIFORM = Int32(0)
const RNG_NORMAL = Int32(1)
const RNG_MULTINORMAL = Int32(2)

function uniform_batch(
    rng_state::TracedRArray{UInt64,1},
    a::TracedRNumber{Float64},
    b::TracedRNumber{Float64},
    ::Val{BatchSize},
) where {BatchSize}
    rng_mlir = TracedUtils.get_mlir_data(rng_state)
    a_mlir = TracedUtils.get_mlir_data(a)
    b_mlir = TracedUtils.get_mlir_data(b)

    rng_state_type = IR.TensorType([2], IR.Type(UInt64))
    result_type = IR.TensorType([BatchSize], IR.Type(Float64))
    dist_attr = rng_distribution_attr(RNG_UNIFORM)

    op = enzyme.random(
        rng_mlir,
        a_mlir,
        b_mlir;
        output_rng_state=rng_state_type,
        result=result_type,
        rng_distribution=dist_attr,
    )

    final_rng = TracedRArray{UInt64,1}((), IR.result(op, 1), (2,))
    samples = TracedRArray{Float64,1}((), IR.result(op, 2), (BatchSize,))
    return final_rng, samples
end

function normal_batch(
    rng_state::TracedRArray{UInt64,1},
    μ::TracedRNumber{Float64},
    σ::TracedRNumber{Float64},
    ::Val{BatchSize},
) where {BatchSize}
    rng_mlir = TracedUtils.get_mlir_data(rng_state)
    μ_mlir = TracedUtils.get_mlir_data(μ)
    σ_mlir = TracedUtils.get_mlir_data(σ)

    rng_state_type = IR.TensorType([2], IR.Type(UInt64))
    result_type = IR.TensorType([BatchSize], IR.Type(Float64))
    dist_attr = rng_distribution_attr(RNG_NORMAL)

    op = enzyme.random(
        rng_mlir,
        μ_mlir,
        σ_mlir;
        output_rng_state=rng_state_type,
        result=result_type,
        rng_distribution=dist_attr,
    )

    final_rng = TracedRArray{UInt64,1}((), IR.result(op, 1), (2,))
    samples = TracedRArray{Float64,1}((), IR.result(op, 2), (BatchSize,))
    return final_rng, samples
end

function multinormal_sample(
    rng_state::TracedRArray{UInt64,1},
    μ::TracedRArray{Float64,1},
    Σ::TracedRArray{Float64,2},
    ::Val{Dim},
) where {Dim}
    rng_mlir = TracedUtils.get_mlir_data(rng_state)
    μ_mlir = TracedUtils.get_mlir_data(μ)
    Σ_mlir = TracedUtils.get_mlir_data(Σ)

    rng_state_type = IR.TensorType([2], IR.Type(UInt64))
    result_type = IR.TensorType([Dim], IR.Type(Float64))
    dist_attr = rng_distribution_attr(RNG_MULTINORMAL)

    op = enzyme.random(
        rng_mlir,
        μ_mlir,
        Σ_mlir;
        output_rng_state=rng_state_type,
        result=result_type,
        rng_distribution=dist_attr,
    )

    final_rng = TracedRArray{UInt64,1}((), IR.result(op, 1), (2,))
    sample = TracedRArray{Float64,1}((), IR.result(op, 2), (Dim,))
    return final_rng, sample
end

# https://en.wikipedia.org/wiki/Standard_error#Exact_value
se_mean(σ, n) = σ / sqrt(n)
# https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance
se_var(σ², n) = σ² * sqrt(2 / (n - 1))
se_std(σ, n) = σ / sqrt(2 * (n - 1))
se_cov(σᵢ, σⱼ, ρ, n) = sqrt((σᵢ^2 * σⱼ^2 + (ρ * σᵢ * σⱼ)^2) / (n - 1))  # ρ = correlation

const N_SIGMA = 5

@testset "enzyme.random op - UNIFORM distribution" begin
    batch_size = 10000
    n_batches = 10
    n_samples = batch_size * n_batches

    @testset "Uniform[0, 1)" begin
        seed = ConcreteRArray(UInt64[42, 123])
        a = ConcreteRNumber(0.0)
        b = ConcreteRNumber(1.0)

        compiled_fn = @compile optimize = :probprog uniform_batch(
            seed, a, b, Val(batch_size)
        )

        all_samples = Float64[]
        rng = seed
        for _ in 1:n_batches
            rng, samples = compiled_fn(rng, a, b, Val(batch_size))
            append!(all_samples, Array(samples))
        end

        expected_mean = 0.5
        expected_var = 1.0 / 12.0
        expected_std = sqrt(expected_var)

        @test all(all_samples .>= 0.0)
        @test all(all_samples .< 1.0)
        @test mean(all_samples) ≈ expected_mean atol =
            N_SIGMA * se_mean(expected_std, n_samples)
        @test var(all_samples) ≈ expected_var atol =
            N_SIGMA * se_var(expected_var, n_samples)
    end

    @testset "Uniform[-5, 5)" begin
        seed = ConcreteRArray(UInt64[99, 77])
        a = ConcreteRNumber(-5.0)
        b = ConcreteRNumber(5.0)

        compiled_fn = @compile optimize = :probprog uniform_batch(
            seed, a, b, Val(batch_size)
        )

        all_samples = Float64[]
        rng = seed
        for _ in 1:n_batches
            rng, samples = compiled_fn(rng, a, b, Val(batch_size))
            append!(all_samples, Array(samples))
        end

        expected_mean = 0.0
        expected_var = 100.0 / 12.0
        expected_std = sqrt(expected_var)

        @test all(all_samples .>= -5.0)
        @test all(all_samples .< 5.0)
        @test mean(all_samples) ≈ expected_mean atol =
            N_SIGMA * se_mean(expected_std, n_samples)
        @test var(all_samples) ≈ expected_var atol =
            N_SIGMA * se_var(expected_var, n_samples)
    end

    @testset "Uniform[10, 20)" begin
        seed = ConcreteRArray(UInt64[11, 22])
        a = ConcreteRNumber(10.0)
        b = ConcreteRNumber(20.0)

        compiled_fn = @compile optimize = :probprog uniform_batch(
            seed, a, b, Val(batch_size)
        )

        all_samples = Float64[]
        rng = seed
        for _ in 1:n_batches
            rng, samples = compiled_fn(rng, a, b, Val(batch_size))
            append!(all_samples, Array(samples))
        end

        expected_mean = 15.0
        expected_var = 100.0 / 12.0
        expected_std = sqrt(expected_var)

        @test all(all_samples .>= 10.0)
        @test all(all_samples .< 20.0)
        @test mean(all_samples) ≈ expected_mean atol =
            N_SIGMA * se_mean(expected_std, n_samples)
        @test var(all_samples) ≈ expected_var atol =
            N_SIGMA * se_var(expected_var, n_samples)
    end
end

@testset "enzyme.random op - NORMAL distribution" begin
    batch_size = 10000
    n_batches = 10
    n_samples = batch_size * n_batches

    @testset "Standard Gaussian" begin
        seed = ConcreteRArray(UInt64[42, 42])
        μ = ConcreteRNumber(0.0)
        σ = ConcreteRNumber(1.0)

        compiled_fn = @compile optimize = :probprog normal_batch(
            seed, μ, σ, Val(batch_size)
        )

        all_samples = Float64[]
        rng = seed
        for _ in 1:n_batches
            rng, samples = compiled_fn(rng, μ, σ, Val(batch_size))
            append!(all_samples, Array(samples))
        end

        expected_std = 1.0
        @test mean(all_samples) ≈ 0.0 atol = N_SIGMA * se_mean(expected_std, n_samples)
        @test std(all_samples) ≈ expected_std atol =
            N_SIGMA * se_std(expected_std, n_samples)
    end

    @testset "Normal(5, 2)" begin
        seed = ConcreteRArray(UInt64[100, 200])
        μ = ConcreteRNumber(5.0)
        σ = ConcreteRNumber(2.0)

        compiled_fn = @compile optimize = :probprog normal_batch(
            seed, μ, σ, Val(batch_size)
        )

        all_samples = Float64[]
        rng = seed
        for _ in 1:n_batches
            rng, samples = compiled_fn(rng, μ, σ, Val(batch_size))
            append!(all_samples, Array(samples))
        end

        expected_std = 2.0
        @test mean(all_samples) ≈ 5.0 atol = N_SIGMA * se_mean(expected_std, n_samples)
        @test std(all_samples) ≈ expected_std atol =
            N_SIGMA * se_std(expected_std, n_samples)
    end

    @testset "Normal(-3, 0.5)" begin
        seed = ConcreteRArray(UInt64[333, 444])
        μ = ConcreteRNumber(-3.0)
        σ = ConcreteRNumber(0.5)

        compiled_fn = @compile optimize = :probprog normal_batch(
            seed, μ, σ, Val(batch_size)
        )

        all_samples = Float64[]
        rng = seed
        for _ in 1:n_batches
            rng, samples = compiled_fn(rng, μ, σ, Val(batch_size))
            append!(all_samples, Array(samples))
        end

        expected_std = 0.5
        @test mean(all_samples) ≈ -3.0 atol = N_SIGMA * se_mean(expected_std, n_samples)
        @test std(all_samples) ≈ expected_std atol =
            N_SIGMA * se_std(expected_std, n_samples)
    end
end

@testset "enzyme.random op - MULTINORMAL distribution" begin
    n_samples = 2000

    @testset "2D Standard Multivariate Normal" begin
        seed = ConcreteRArray(UInt64[55, 66])
        μ = ConcreteRArray([0.0, 0.0])
        Σ = ConcreteRArray([1.0 0.0; 0.0 1.0])

        σ₁, σ₂, ρ₁₂ = 1.0, 1.0, 0.0

        compiled_fn = @compile optimize = :probprog multinormal_sample(seed, μ, Σ, Val(2))

        samples_matrix = zeros(n_samples, 2)
        rng = seed
        for i in 1:n_samples
            rng, sample = compiled_fn(rng, μ, Σ, Val(2))
            samples_matrix[i, :] = Array(sample)
        end

        sample_means = vec(mean(samples_matrix; dims=1))
        @test sample_means[1] ≈ 0.0 atol = N_SIGMA * se_mean(σ₁, n_samples)
        @test sample_means[2] ≈ 0.0 atol = N_SIGMA * se_mean(σ₂, n_samples)

        sample_cov = cov(samples_matrix)
        @test sample_cov[1, 1] ≈ 1.0 atol = N_SIGMA * se_cov(σ₁, σ₁, 1.0, n_samples)
        @test sample_cov[2, 2] ≈ 1.0 atol = N_SIGMA * se_cov(σ₂, σ₂, 1.0, n_samples)
        @test sample_cov[1, 2] ≈ 0.0 atol = N_SIGMA * se_cov(σ₁, σ₂, ρ₁₂, n_samples)
        @test sample_cov[2, 1] ≈ 0.0 atol = N_SIGMA * se_cov(σ₁, σ₂, ρ₁₂, n_samples)
    end

    @testset "2D Correlated Multivariate Normal" begin
        seed = ConcreteRArray(UInt64[77, 88])
        μ = ConcreteRArray([2.0, -1.0])
        Σ = ConcreteRArray([4.0 1.5; 1.5 2.0])

        σ₁, σ₂ = 2.0, sqrt(2.0)
        ρ₁₂ = 1.5 / (σ₁ * σ₂)

        compiled_fn = @compile optimize = :probprog multinormal_sample(seed, μ, Σ, Val(2))

        samples_matrix = zeros(n_samples, 2)
        rng = seed
        for i in 1:n_samples
            rng, sample = compiled_fn(rng, μ, Σ, Val(2))
            samples_matrix[i, :] = Array(sample)
        end

        sample_means = vec(mean(samples_matrix; dims=1))
        @test sample_means[1] ≈ 2.0 atol = N_SIGMA * se_mean(σ₁, n_samples)
        @test sample_means[2] ≈ -1.0 atol = N_SIGMA * se_mean(σ₂, n_samples)

        sample_cov = cov(samples_matrix)
        @test sample_cov[1, 1] ≈ 4.0 atol = N_SIGMA * se_cov(σ₁, σ₁, 1.0, n_samples)
        @test sample_cov[2, 2] ≈ 2.0 atol = N_SIGMA * se_cov(σ₂, σ₂, 1.0, n_samples)
        @test sample_cov[1, 2] ≈ 1.5 atol = N_SIGMA * se_cov(σ₁, σ₂, ρ₁₂, n_samples)
        @test sample_cov[2, 1] ≈ 1.5 atol = N_SIGMA * se_cov(σ₁, σ₂, ρ₁₂, n_samples)
    end

    @testset "3D Multivariate Normal with diagonal covariance" begin
        seed = ConcreteRArray(UInt64[111, 222])
        μ = ConcreteRArray([1.0, 2.0, 3.0])
        Σ = ConcreteRArray([1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 9.0])

        σ₁, σ₂, σ₃ = 1.0, 2.0, 3.0

        compiled_fn = @compile optimize = :probprog multinormal_sample(seed, μ, Σ, Val(3))

        samples_matrix = zeros(n_samples, 3)
        rng = seed
        for i in 1:n_samples
            rng, sample = compiled_fn(rng, μ, Σ, Val(3))
            samples_matrix[i, :] = Array(sample)
        end

        sample_means = vec(mean(samples_matrix; dims=1))
        @test sample_means[1] ≈ 1.0 atol = N_SIGMA * se_mean(σ₁, n_samples)
        @test sample_means[2] ≈ 2.0 atol = N_SIGMA * se_mean(σ₂, n_samples)
        @test sample_means[3] ≈ 3.0 atol = N_SIGMA * se_mean(σ₃, n_samples)

        sample_cov = cov(samples_matrix)
        @test sample_cov[1, 1] ≈ 1.0 atol = N_SIGMA * se_cov(σ₁, σ₁, 1.0, n_samples)
        @test sample_cov[2, 2] ≈ 4.0 atol = N_SIGMA * se_cov(σ₂, σ₂, 1.0, n_samples)
        @test sample_cov[3, 3] ≈ 9.0 atol = N_SIGMA * se_cov(σ₃, σ₃, 1.0, n_samples)

        @test sample_cov[1, 2] ≈ 0.0 atol = N_SIGMA * se_cov(σ₁, σ₂, 0.0, n_samples)
        @test sample_cov[1, 3] ≈ 0.0 atol = N_SIGMA * se_cov(σ₁, σ₃, 0.0, n_samples)
        @test sample_cov[2, 3] ≈ 0.0 atol = N_SIGMA * se_cov(σ₂, σ₃, 0.0, n_samples)
    end
end
