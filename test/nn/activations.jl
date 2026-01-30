using LuxLib, Reactant, Enzyme, NNlib, Test, Statistics

@testset "Activation Functions" begin
    sumabs2(f, x) = sum(abs2, f.(x))

    function ∇sumabs2(gradfn, f, x)
        Ω = f.(x)
        return 2 .* gradfn.(Ω, x) .* Ω
    end

    x_act = Reactant.TestUtils.construct_test_array(Float32, 10, 10) .- 0.5f0
    x_act_ca = Reactant.to_rarray(x_act)

    @testset "Activation: $act" for (act, gradfn) in (
        (identity, (Ω, x) -> one(Ω)),
        (relu, (Ω, x) -> (Ω > 0)),
        (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
        (tanh, (Ω, x) -> conj(1 - Ω^2)),
        (tanh_fast, (Ω, x) -> conj(1 - Ω^2)),
        (sigmoid_fast, (Ω, x) -> conj((1 - Ω) * Ω)),
        (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
        (abs2, (Ω, x) -> (2 * x)),
        (relu6, (Ω, x) -> (Ω > 0) & (Ω < 6)),
    )
        y_simple = sumabs2(act, x_act)
        y_compile = @jit sumabs2(act, x_act_ca)

        ∂x_compile = @jit(Enzyme.gradient(Reverse, sumabs2, Const(act), x_act_ca))[2]
        ∂x_compile_gt = @jit ∇sumabs2(gradfn, act, x_act_ca)

        @test y_simple ≈ y_compile
        @test ∂x_compile ≈ ∂x_compile_gt atol = 1e-3 rtol = 1e-3
    end
end

@testset "Bias Activation" begin
    biasact(act, x, b) = bias_activation(act, x, b)
    sumabs2biasact(act, x, b) = sum(abs2, biasact(act, x, b))
    biasact!!(act, x, b) = bias_activation!!(act, copy(x), b)
    sumabs2biasact!!(act, x, b) = sum(abs2, biasact!!(act, x, b))

    function ∇biasact(act, x, b)
        return Enzyme.gradient(Reverse, sumabs2biasact, Const(act), x, b)[2:end]
    end

    function ∇biasact!!(act, x, b)
        return Enzyme.gradient(Reverse, sumabs2biasact!!, Const(act), x, b)[2:end]
    end

    function ∇biasact(gradfn, act, x, b)
        xb = x .+ b
        Ω = act.(xb)
        ∂x = 2 .* gradfn.(Ω, xb) .* Ω
        ∂b = vec(sum(∂x; dims=2))
        return ∂x, ∂b
    end

    @testset "Activation: $act" for (act, gradfn) in (
        (identity, (Ω, x) -> one(Ω)),
        (relu, (Ω, x) -> (Ω > 0)),
        (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
        (tanh, (Ω, x) -> conj(1 - Ω^2)),
        (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
    )
        x = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
        b = Reactant.TestUtils.construct_test_array(Float32, 10)

        x_ra = Reactant.to_rarray(x)
        b_ra = Reactant.to_rarray(b)

        y_compile = @jit biasact(act, x_ra, b_ra)
        y_compile!! = @jit biasact!!(act, x_ra, b_ra)

        y_simple = biasact(act, x, b)
        y_simple!! = biasact!!(act, x, b)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_gt, ∂b_gt = @jit ∇biasact(gradfn, act, x_ra, b_ra)

        @testset "Enzyme: bias_activation" begin
            ∂x_enz, ∂b_enz = @jit ∇biasact(act, x_ra, b_ra)

            @test ∂x_enz ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
            @test ∂b_enz ≈ ∂b_gt atol = 1e-5 rtol = 1e-2
        end

        @testset "Enzyme: bias_activation!!" begin
            ∂x_enz!!, ∂b_enz!! = @jit ∇biasact!!(act, x_ra, b_ra)

            @test ∂x_enz!! ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
            @test ∂b_enz!! ≈ ∂b_gt atol = 1e-5 rtol = 1e-2
        end
    end
end

@testset "Fast Activation" begin
    # Here we are testing that fast_activation doesn't switch to the faster versions
    sumabs2(f, x) = sum(abs2, fast_activation(f, x))
    sumabs2!!(f, x) = sum(abs2, fast_activation!!(f, copy(x)))

    ∇sumabs2(f, x) = Enzyme.gradient(Reverse, sumabs2, Const(f), x)[2]
    ∇sumabs2!!(f, x) = Enzyme.gradient(Reverse, sumabs2!!, Const(f), x)[2]

    function ∇sumabs2(gradfn, f, x)
        Ω = f.(x)
        return 2 .* gradfn.(Ω, x) .* Ω
    end

    x_act = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
    x_act_ca = Reactant.to_rarray(x_act)

    @testset "Activation: $act" for (act, gradfn) in (
        (identity, (Ω, x) -> one(Ω)),
        (relu, (Ω, x) -> (Ω > 0)),
        (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
        (tanh, (Ω, x) -> conj(1 - Ω^2)),
        (tanh_fast, (Ω, x) -> conj(1 - Ω^2)),
        (sigmoid_fast, (Ω, x) -> conj((1 - Ω) * Ω)),
        (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
        (abs2, (Ω, x) -> (2 * x)),
        (relu6, (Ω, x) -> (Ω > 0) & (Ω < 6)),
    )
        y_simple = sumabs2(act, x_act)
        y_simple!! = sumabs2!!(act, x_act)
        y_compile = @jit sumabs2(act, x_act_ca)
        y_compile!! = @jit sumabs2!!(act, x_act_ca)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_enz = @jit ∇sumabs2(act, x_act_ca)
        ∂x_enz!! = @jit ∇sumabs2!!(act, x_act_ca)
        ∂x_gt = @jit ∇sumabs2(gradfn, act, x_act_ca)

        @test ∂x_enz ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
        @test ∂x_enz!! ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
    end
end
