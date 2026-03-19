using Reactant, Test, Statistics, LinearAlgebra

@testset "Statistics: `mean` & `var`" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 4)
    x_ca = Reactant.to_rarray(x)

    @test @jit(mean(x_ca)) ≈ mean(x)
    @test @jit(mean(x_ca; dims=1)) ≈ mean(x; dims=1)
    @test @jit(mean(x_ca; dims=(1, 2))) ≈ mean(x; dims=(1, 2))
    @test @jit(mean(x_ca; dims=(1, 3))) ≈ mean(x; dims=(1, 3))

    @test @jit(var(x_ca)) ≈ var(x)
    @test @jit(var(x_ca, dims=1)) ≈ var(x; dims=1)
    @test @jit(var(x_ca, dims=(1, 2); corrected=false)) ≈
        var(x; dims=(1, 2), corrected=false)
    @test @jit(var(x_ca; dims=(1, 3), corrected=false)) ≈
        var(x; dims=(1, 3), corrected=false)
end

@testset "middle" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(middle(x_ra)) ≈ middle(x)
end

@testset "cov2cor!" begin
    @testset "Symmetric" begin
        C = [16.0 4.0; 4.0 4.0]
        xsd = [4.0, 2.0]

        C_cpu = copy(C)
        Statistics.cov2cor!(C_cpu, xsd)

        C_r = Reactant.to_rarray(copy(C))
        xsd_r = Reactant.to_rarray(xsd)

        @jit(Statistics.cov2cor!(C_r, xsd_r))

        @test Array(C_r) ≈ C_cpu
        @test diag(Array(C_r)) ≈ ones(2)
        @test issymmetric(Array(C_r))
    end

    @testset "Asymmetric" begin
        C = [16.0 4.0; 4.0 4.0]
        xsd = [4.0, 2.0]
        ysd = [2.0, 4.0]

        C_cpu = copy(C)
        Statistics.cov2cor!(C_cpu, xsd, ysd)

        C_r = Reactant.to_rarray(copy(C))
        xsd_r = Reactant.to_rarray(xsd)
        ysd_r = Reactant.to_rarray(ysd)

        @jit(Statistics.cov2cor!(C_r, xsd_r, ysd_r))

        @test Array(C_r) ≈ C_cpu
    end

    @testset "Asymmetric with Scalars" begin
        C = [16.0 4.0; 4.0 4.0]
        xsd_val = 4.0
        ysd_val = 2.0
        xsd_arr = [4.0, 2.0]
        ysd_arr = [2.0, 4.0]

        # Scalar xsd
        C_cpu = copy(C)
        Statistics.cov2cor!(C_cpu, xsd_val, ysd_arr)
        C_r = Reactant.to_rarray(copy(C))
        ysd_r = Reactant.to_rarray(ysd_arr)
        @jit(Statistics.cov2cor!(C_r, xsd_val, ysd_r))
        @test Array(C_r) ≈ C_cpu

        # Scalar ysd
        C_cpu = copy(C)
        Statistics.cov2cor!(C_cpu, xsd_arr, ysd_val)
        C_r = Reactant.to_rarray(copy(C))
        xsd_r = Reactant.to_rarray(xsd_arr)
        @jit(Statistics.cov2cor!(C_r, xsd_r, ysd_val))
        @test Array(C_r) ≈ C_cpu
    end

    @testset "Symmetric - Complex" begin
        A = ComplexF32[1.0+0im 0.5+0.5im; 0.5-0.5im 1.0+0im]
        xsd = Float32[1.0, 1.0]

        C = A
        C_cpu = copy(C)
        Statistics.cov2cor!(C_cpu, xsd)

        C_r = Reactant.to_rarray(copy(C))
        xsd_r = Reactant.to_rarray(xsd)

        @jit(Statistics.cov2cor!(C_r, xsd_r))

        @test Array(C_r) ≈ C_cpu
        @test ishermitian(Array(C_r))
    end
end

@testset "Statistics: cor, cov, median" begin
    @testset "median" begin
        x = Reactant.TestUtils.construct_test_array(Float64, 10)
        x_ra = Reactant.to_rarray(x)

        @test @jit(median(x_ra)) ≈ median(x)

        x_odd = Reactant.TestUtils.construct_test_array(Float64, 11)
        x_odd_ra = Reactant.to_rarray(x_odd)
        @test @jit(median(x_odd_ra)) ≈ median(x_odd)

        # With NaNs
        x_nan = copy(x)
        x_nan[1] = NaN
        x_nan_ra = Reactant.to_rarray(x_nan)
        # Median with NaN should be NaN
        @test isnan(@jit(median(x_nan_ra)))
    end

    @testset "cov" begin
        x = Reactant.TestUtils.construct_test_array(Float64, 10, 5)
        x_ra = Reactant.to_rarray(x)

        @test @jit(cov(x_ra)) ≈ cov(x)
        @test @jit(cov(x_ra; dims=1)) ≈ cov(x; dims=1)
        @test @jit(cov(x_ra; dims=2)) ≈ cov(x; dims=2)
        @test @jit(cov(x_ra; corrected=false)) ≈ cov(x; corrected=false)

        y = Reactant.TestUtils.construct_test_array(Float64, 10, 5)
        y_ra = Reactant.to_rarray(y)

        @test @jit(cov(x_ra, y_ra)) ≈ cov(x, y)
        @test @jit(cov(x_ra, y_ra; dims=2)) ≈ cov(x, y; dims=2)

        # Vectors
        v1 = Reactant.TestUtils.construct_test_array(Float64, 10)
        v2 = Reactant.TestUtils.construct_test_array(Float64, 10)
        v1_ra = Reactant.to_rarray(v1)
        v2_ra = Reactant.to_rarray(v2)

        @test @jit(cov(v1_ra, v2_ra)) ≈ cov(v1, v2)
    end

    @testset "cor" begin
        x = Reactant.TestUtils.construct_test_array(Float64, 10, 5)
        x_ra = Reactant.to_rarray(x)

        @test @jit(cor(x_ra)) ≈ cor(x)
        @test @jit(cor(x_ra; dims=2)) ≈ cor(x; dims=2)

        v1 = Reactant.TestUtils.construct_test_array(Float64, 10)
        v2 = Reactant.TestUtils.construct_test_array(Float64, 10)
        v1_ra = Reactant.to_rarray(v1)
        v2_ra = Reactant.to_rarray(v2)

        @test @jit(cor(v1_ra, v2_ra)) ≈ cor(v1, v2)
    end
end

@testset "quantile" begin
    # Basic Vector
    x = [1.0, 2.0, 3.0, 4.0]
    x_ra = Reactant.to_rarray(x)

    @test @jit(quantile(x_ra, 0.5)) ≈ 2.5

    # Vector of probabilities
    probs = [0.5]
    @test @jit(quantile(x_ra, probs)) ≈ [2.5]

    # Vector of probabilities - multiple
    probs2 = [0.25, 0.5, 0.75]
    x2 = [1.0, 3.0]
    x2_ra = Reactant.to_rarray(x2)
    @test Array(@jit(quantile(x2_ra, probs2)))[2] ≈ median(x2)

    # Range
    v = collect(100.0:-1.0:0.0)
    p = collect(0.0:0.1:1.0)
    v_ra = Reactant.to_rarray(v)
    @test @jit(quantile(v_ra, p)) ≈ collect(0.0:10.0:100.0)

    # Sorted
    v_sorted = collect(0.0:100.0)
    v_sorted_ra = Reactant.to_rarray(v_sorted)
    @test @jit(quantile(v_sorted_ra, p; sorted=true)) ≈ collect(0.0:10.0:100.0)

    # Float32
    v32 = collect(100.0f0:-1.0f0:0.0f0)
    p32 = collect(0.0f0:0.1f0:1.0f0)
    v32_ra = Reactant.to_rarray(v32)
    @test @jit(quantile(v32_ra, p32)) ≈ collect(0.0f0:10.0f0:100.0f0)

    # TODO: fix nan/inf handling
    # Inf
    # x_inf = [Inf, Inf]
    # x_inf_ra = Reactant.to_rarray(x_inf)
    # @test @jit(quantile(x_inf_ra, 0.5)) == Inf

    # -Inf
    # x_ninf = [-Inf, 1.0]
    # x_ninf_ra = Reactant.to_rarray(x_ninf)
    # @test @jit(quantile(x_ninf_ra, 0.5)) == -Inf

    # Small tolerance
    x_small = [0.0, 1.0]
    x_small_ra = Reactant.to_rarray(x_small)
    @test @jit(quantile(x_small_ra, 1e-18)) ≈ quantile(x_small, 1e-18)

    # Multidimensional
    x_mat = reshape(collect(1.0:100.0), (10, 10))
    x_mat_ra = Reactant.to_rarray(x_mat)
    p_vec = [0.00, 0.25, 0.50, 0.75, 1.00]
    @test @jit(quantile(x_mat_ra, p_vec)) ≈ [1.0, 25.75, 50.5, 75.25, 100.0]

    # alpha/beta parameters
    v_ab = [2.0, 3.0, 4.0, 6.0, 9.0, 2.0, 6.0, 2.0, 21.0, 17.0]
    v_ab_ra = Reactant.to_rarray(v_ab)

    @test @jit(quantile(v_ab_ra, 0.0, alpha=0.0, beta=0.0)) ≈ 2.0
    @test @jit(quantile(v_ab_ra, 0.2, alpha=1.0, beta=1.0)) ≈ 2.0
    @test @jit(quantile(v_ab_ra, 0.4, alpha=0.0, beta=1.0)) ≈ 3.0
end
