using Reactant, Test
using Reactant: ProbProg, ReactantRNG, ConcreteRArray, ConcreteRNumber

const _ld_normal_logpdf = ProbProg._normal_logpdf

function _ld_prior_model(rng, μ, σ)
    _, x = ProbProg.sample(rng, ProbProg.Normal(μ, σ, (2,)); symbol=:x)
    return x
end

function _ld_linear_model(rng, xs)
    _, a = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:a)
    _, b = ProbProg.sample(rng, ProbProg.Normal(0.0, 3.0, (1,)); symbol=:b)
    _, y1 = ProbProg.sample(rng, ProbProg.Normal(a .+ xs[1:3], 0.5, (3,)); symbol=:y1)
    _, y2 = ProbProg.sample(rng, ProbProg.Normal(b .+ xs[4:6], 0.25, (3,)); symbol=:y2)
    return vcat(y1, y2)
end

function _ld_submodel(rng, mean_vec, σ)
    _, z = ProbProg.sample(rng, ProbProg.Normal(mean_vec, σ, (2,)); symbol=:z)
    return z
end

function _ld_hier_model(rng, y_std)
    _, root = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (2,)); symbol=:root)
    _, sub = ProbProg.sample(rng, _ld_submodel, root, 1.0; symbol=:sub)
    _, obs = ProbProg.sample(rng, ProbProg.Normal(sub, y_std, (2,)); symbol=:obs)
    return obs
end

function _traced_caller(rng, θ, obs_tensor, xs, sel, obs_addrs)
    lp = ProbProg.logp(
        rng,
        θ,
        _ld_linear_model,
        xs;
        selection=sel,
        observations_tensor=obs_tensor,
        observation_addresses=obs_addrs,
    )
    lp2, g = ProbProg.logp_and_grad(
        rng,
        θ,
        _ld_linear_model,
        xs;
        selection=sel,
        observations_tensor=obs_tensor,
        observation_addresses=obs_addrs,
    )
    return lp, lp2, g
end

function _exact_linear_logp_and_grad(
    a::Float64, b::Float64, xs::Vector{Float64}, y1::Vector{Float64}, y2::Vector{Float64}
)
    lp_a = _ld_normal_logpdf([a], 0.0, 2.0, (1,))
    lp_b = _ld_normal_logpdf([b], 0.0, 3.0, (1,))
    lp_y1 = _ld_normal_logpdf(y1, a .+ xs[1:3], 0.5, (3,))
    lp_y2 = _ld_normal_logpdf(y2, b .+ xs[4:6], 0.25, (3,))
    total_lp = lp_a + lp_b + lp_y1 + lp_y2
    grad_a = -a / 4.0 + sum((y1 .- (a .+ xs[1:3])) ./ 0.25)
    grad_b = -b / 9.0 + sum((y2 .- (b .+ xs[4:6])) ./ 0.0625)
    return total_lp, [grad_a, grad_b]
end

@testset "ProbProg LogDensity (#2920)" begin
    @testset "Prior-only logp and logp_and_grad" begin
        μ = 1.25
        σ = 1.5
        f_lp = ProbProg.compile_logp(_ld_prior_model, μ, σ)
        f_lpg = ProbProg.compile_logp_and_grad(_ld_prior_model, μ, σ)
        @test f_lp.dimension == 2
        @test f_lpg.dimension == 2

        θ1 = [0.5, -0.75]
        expected_lp1 = _ld_normal_logpdf(θ1, μ, σ, (2,))
        expected_g1 = .-(θ1 .- μ) ./ (σ^2)

        @test f_lp(θ1) ≈ expected_lp1 atol = 1e-10
        lp1, g1 = f_lpg(θ1)
        @test lp1 ≈ expected_lp1 atol = 1e-10
        @test g1 ≈ expected_g1 atol = 1e-10

        # Reuse compiled callables at a second parameter point without recompiling
        θ2 = [-1.1, 2.3]
        expected_lp2 = _ld_normal_logpdf(θ2, μ, σ, (2,))
        expected_g2 = .-(θ2 .- μ) ./ (σ^2)
        lp2, g2 = f_lpg(θ2)
        @test f_lp(θ2) ≈ expected_lp2 atol = 1e-10
        @test lp2 ≈ expected_lp2 atol = 1e-10
        @test g2 ≈ expected_g2 atol = 1e-10

        # ConcreteRArray inputs (1D and 2D)
        θ_dev = Reactant.to_rarray(θ1)
        lp_dev, g_dev = f_lpg(θ_dev)
        @test Float64(Reactant.to_number(lp_dev)) ≈ expected_lp1 atol = 1e-10
        @test Array(g_dev) ≈ expected_g1 atol = 1e-10

        θ_mat_dev = Reactant.to_rarray(reshape(θ1, 1, 2))
        _, g_mat_dev = f_lpg(θ_mat_dev)
        @test size(g_mat_dev) == (1, 2)
        @test vec(Array(g_mat_dev)) ≈ expected_g1 atol = 1e-10

        # Direct one-shot helpers
        @test ProbProg.logp(_ld_prior_model, θ1, μ, σ) ≈ expected_lp1 atol = 1e-10
        lp_direct, g_direct = ProbProg.logp_and_grad(_ld_prior_model, θ1, μ, σ)
        @test lp_direct ≈ expected_lp1 atol = 1e-10
        @test g_direct ≈ expected_g1 atol = 1e-10
    end

    @testset "Conditioned posterior, positional observations, selection order, and only_fg!" begin
        xs = [-1.0, 0.0, 1.0, 0.5, 1.5, 2.5]
        y1 = [0.2, 1.1, 2.4]
        y2 = [-0.3, 0.8, 1.9]
        obs = ProbProg.Constraint(:y1 => y1, :y2 => y2)

        # Positional observations signature: compile_logp_and_grad(model, obs, args...)
        fg_pos = ProbProg.compile_logp_and_grad(_ld_linear_model, obs, xs)
        lp_pos = ProbProg.compile_logp(_ld_linear_model, obs, xs)
        @test fg_pos.dimension == 2
        @test length(fg_pos.entries) == 2

        θ = [0.8, -0.4]
        exp_lp, exp_g = _exact_linear_logp_and_grad(θ[1], θ[2], xs, y1, y2)
        val, grad = fg_pos(θ)
        @test val ≈ exp_lp atol = 1e-10
        @test lp_pos(θ) ≈ exp_lp atol = 1e-10
        @test grad ≈ exp_g atol = 1e-10

        # Keyword `constraints` alias
        val_c, grad_c = ProbProg.logp_and_grad(_ld_linear_model, θ, xs; constraints=obs)
        @test val_c ≈ exp_lp atol = 1e-10
        @test grad_c ≈ exp_g atol = 1e-10

        # Custom selection order (:b first, :a second)
        rev_sel = ProbProg.select(ProbProg.Address(:b), ProbProg.Address(:a))
        fg_rev = ProbProg.compile_logp_and_grad(
            _ld_linear_model, xs; observations=obs, selection=rev_sel
        )
        θ_rev = [-0.4, 0.8]
        val_rev, grad_rev = fg_rev(θ_rev)
        @test val_rev ≈ exp_lp atol = 1e-10
        @test grad_rev ≈ [exp_g[2], exp_g[1]] atol = 1e-10

        # Optim.only_fg! 3-argument protocol
        g_buf = zeros(2)
        ret_both = fg_pos(1.0, g_buf, θ)
        @test ret_both ≈ exp_lp atol = 1e-10
        @test g_buf ≈ exp_g atol = 1e-10

        fill!(g_buf, 0.0)
        ret_grad_only = fg_pos(nothing, g_buf, θ)
        @test ret_grad_only === nothing
        @test g_buf ≈ exp_g atol = 1e-10

        ret_val_only = fg_pos(1.0, nothing, θ)
        @test ret_val_only ≈ exp_lp atol = 1e-10
    end

    @testset "Composite submodel addresses and traced in-module calls" begin
        obs_hier = ProbProg.Constraint(:obs => [0.5, -0.5])
        fg_hier = ProbProg.compile_logp_and_grad(_ld_hier_model, obs_hier, 0.5)
        @test fg_hier.dimension == 4

        θ_h = [0.2, -0.3, 0.4, -0.1]
        root_v = θ_h[1:2]
        sub_v = θ_h[3:4]
        exp_h_lp =
            _ld_normal_logpdf(root_v, 0.0, 2.0, (2,)) +
            _ld_normal_logpdf(sub_v, root_v, 1.0, (2,)) +
            _ld_normal_logpdf([0.5, -0.5], sub_v, 0.5, (2,))
        exp_g_root = .-root_v ./ 4.0 .+ (sub_v .- root_v)
        exp_g_sub = .-(sub_v .- root_v) .+ ([0.5, -0.5] .- sub_v) ./ 0.25

        val_h, grad_h = fg_hier(θ_h)
        @test val_h ≈ exp_h_lp atol = 1e-10
        @test grad_h ≈ vcat(exp_g_root, exp_g_sub) atol = 1e-10

        # Traced mode inside `@compile optimize = :probprog`
        xs = [-1.0, 0.0, 1.0, 0.5, 1.5, 2.5]
        y1 = [0.2, 1.1, 2.4]
        y2 = [-0.3, 0.8, 1.9]
        obs_tensor = Reactant.to_rarray(reshape(vcat(y1, y2), 1, :))
        obs_addrs = [ProbProg.Address(:y1), ProbProg.Address(:y2)]
        sel = [ProbProg.Address(:a), ProbProg.Address(:b)]
        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 7]))
        θ_dev = Reactant.to_rarray([0.8, -0.4])

        traced_fn = @compile optimize = :probprog _traced_caller(
            rng, θ_dev, obs_tensor, xs, sel, obs_addrs
        )
        lp_t1, lp_t2, g_t = traced_fn(rng, θ_dev, obs_tensor, xs, sel, obs_addrs)
        exp_lp, exp_g = _exact_linear_logp_and_grad(0.8, -0.4, xs, y1, y2)
        @test Float64(Reactant.to_number(lp_t1)) ≈ exp_lp atol = 1e-10
        @test Float64(Reactant.to_number(lp_t2)) ≈ exp_lp atol = 1e-10
        @test Array(g_t) ≈ exp_g atol = 1e-10
    end

    @testset "Validation errors" begin
        fg = ProbProg.compile_logp_and_grad(_ld_prior_model, 0.0, 1.0)
        @test_throws DimensionMismatch fg([1.0, 2.0, 3.0])
        @test_throws ArgumentError ProbProg.compile_logp(
            _ld_prior_model, 0.0, 1.0; observations=ProbProg.Constraint(:unknown => [1.0])
        )
        @test_throws DimensionMismatch ProbProg.compile_logp(
            _ld_prior_model, 0.0, 1.0; observations=ProbProg.Constraint(:x => [1.0])
        )
        @test_throws ArgumentError ProbProg.compile_logp(
            _ld_prior_model,
            0.0,
            1.0;
            observations=ProbProg.Constraint(:x => [1.0, 2.0]),
            selection=ProbProg.select(ProbProg.Address(:x)),
        )
        @test_throws ArgumentError ProbProg.compile_logp(
            _ld_prior_model,
            0.0,
            1.0;
            observations=ProbProg.Constraint(:x => [1.0, 2.0]),
            constraints=ProbProg.Constraint(:x => [1.0, 2.0]),
        )
    end
end
