using NNlib, Reactant, Enzyme, Test

gs_zeros(x, g) = NNlib.grid_sample(x, g; padding_mode=:zeros)
gs_border(x, g) = NNlib.grid_sample(x, g; padding_mode=:border)
gs_default(x, g) = NNlib.grid_sample(x, g)

function gs_inplace_default!(out, x, g)
    NNlib.grid_sample!(out, x, g)
    return out
end

function gs_inplace_zeros!(out, x, g)
    NNlib.grid_sample!(out, x, g, :zeros)
    return out
end

function gs_inplace_border!(out, x, g)
    NNlib.grid_sample!(out, x, g, :border)
    return out
end

gs_grad_zeros(Δ, x, g) = NNlib.∇grid_sample(Δ, x, g; padding_mode=:zeros)
gs_grad_border(Δ, x, g) = NNlib.∇grid_sample(Δ, x, g; padding_mode=:border)
gs_grad_default(Δ, x, g) = NNlib.∇grid_sample(Δ, x, g)

function gs_grad_inplace!(dx, dgrid, Δ, x, g, mode::Symbol)
    NNlib.∇grid_sample!(dx, dgrid, Δ, x, g, mode)
    return dx, dgrid
end

gs_weighted_loss_zeros(x, g, w) = sum(NNlib.grid_sample(x, g; padding_mode=:zeros) .* w)
gs_weighted_loss_border(x, g, w) = sum(NNlib.grid_sample(x, g; padding_mode=:border) .* w)

function gs_rev_grads_zeros(x, g, w)
    dx, dg, _ = Enzyme.gradient(Reverse, gs_weighted_loss_zeros, x, g, Const(w))
    return dx, dg
end

function gs_rev_grads_border(x, g, w)
    dx, dg, _ = Enzyme.gradient(Reverse, gs_weighted_loss_border, x, g, Const(w))
    return dx, dg
end

function gs_fwd_deriv_zeros(x, dx, g, dg, w)
    return Enzyme.autodiff(
        Forward,
        gs_weighted_loss_zeros,
        DuplicatedNoNeed,
        Duplicated(x, dx),
        Duplicated(g, dg),
        Const(w),
    )[1]
end

function gs_fwd_deriv_border(x, dx, g, dg, w)
    return Enzyme.autodiff(
        Forward,
        gs_weighted_loss_border,
        DuplicatedNoNeed,
        Duplicated(x, dx),
        Duplicated(g, dg),
        Const(w),
    )[1]
end

function gs_native_only_check(x, g)
    h_x = collect(Float64, reshape(1:4, 2, 2, 1, 1))
    h_g = zeros(Float64, 2, 1, 1, 1)
    h_out = NNlib.grid_sample(h_x, h_g; padding_mode=:zeros)
    return NNlib.grid_sample(x, g; padding_mode=:zeros) .+ eltype(x)(h_out[1, 1, 1, 1])
end

@testset "NNlib.grid_sample Doctest & 4D Primal" begin
    x_doc = reshape(collect(1.0:4.0), (2, 2, 1, 1))
    grid_doc = Array{Float64}(undef, 2, 3, 2, 1)
    grid_doc[:, 1, 1, 1] .= (-3.0, -1.0)
    grid_doc[:, 2, 1, 1] .= (0.0, -1.0)
    grid_doc[:, 3, 1, 1] .= (1.0, -1.0)
    grid_doc[:, 1, 2, 1] .= (-1.0, 1.0)
    grid_doc[:, 2, 2, 1] .= (0.0, 1.0)
    grid_doc[:, 3, 2, 1] .= (3.0, 1.0)

    x_doc_ra = Reactant.to_rarray(x_doc)
    grid_doc_ra = Reactant.to_rarray(grid_doc)

    @test Array(@jit(gs_default(x_doc_ra, grid_doc_ra))) ≈
        NNlib.grid_sample(x_doc, grid_doc; padding_mode=:zeros)
    @test Array(@jit(gs_zeros(x_doc_ra, grid_doc_ra))) ≈
        NNlib.grid_sample(x_doc, grid_doc; padding_mode=:zeros)
    @test Array(@jit(gs_border(x_doc_ra, grid_doc_ra))) ≈
        NNlib.grid_sample(x_doc, grid_doc; padding_mode=:border)

    # Mixed host Array + traced ConcreteRArray and native fallback inside @jit
    @test Array(@jit(gs_zeros(x_doc, grid_doc_ra))) ≈
        NNlib.grid_sample(x_doc, grid_doc; padding_mode=:zeros)
    @test Array(@jit(gs_border(x_doc_ra, grid_doc))) ≈
        NNlib.grid_sample(x_doc, grid_doc; padding_mode=:border)
    @test Array(@jit(gs_native_only_check(x_doc_ra, grid_doc_ra))) ≈
        NNlib.grid_sample(x_doc, grid_doc; padding_mode=:zeros) .+ 2.5

    for T in (Float32, Float64)
        x = Reactant.TestUtils.construct_test_array(T, 5, 4, 3, 2)
        # Include interior, exact corners (-1, 1), near-boundary, and out-of-bounds coordinates
        raw_grid = Reactant.TestUtils.construct_test_array(T, 2, 6, 5, 2)
        grid = T(1.6) .* (raw_grid .- T(0.5))
        grid[1, 1, 1, 1] = T(-1.0)
        grid[2, 1, 1, 1] = T(-1.0)
        grid[1, 6, 5, 2] = T(1.0)
        grid[2, 6, 5, 2] = T(1.0)
        grid[1, 2, 1, 1] = T(-2.25)
        grid[2, 3, 5, 2] = T(2.5)

        x_ra = Reactant.to_rarray(x)
        grid_ra = Reactant.to_rarray(grid)

        @test Array(@jit(gs_zeros(x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:zeros)
        @test Array(@jit(gs_border(x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:border)

        out_ra = Reactant.to_rarray(zeros(T, 6, 5, 3, 2))
        @test Array(@jit(gs_inplace_default!(out_ra, x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:zeros)
        @test Array(@jit(gs_inplace_border!(out_ra, x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:border)
        @test Array(@jit(gs_inplace_zeros!(out_ra, x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:zeros)
    end
end

@testset "NNlib.grid_sample 5D Primal" begin
    for T in (Float32, Float64)
        x = Reactant.TestUtils.construct_test_array(T, 4, 3, 5, 2, 2)
        raw_grid = Reactant.TestUtils.construct_test_array(T, 3, 3, 4, 2, 2)
        grid = T(1.7) .* (raw_grid .- T(0.5))
        grid[:, 1, 1, 1, 1] .= (T(-1.0), T(1.0), T(-1.0))
        grid[:, 3, 4, 2, 2] .= (T(-2.2), T(0.25), T(2.4))

        x_ra = Reactant.to_rarray(x)
        grid_ra = Reactant.to_rarray(grid)

        @test Array(@jit(gs_zeros(x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:zeros)
        @test Array(@jit(gs_border(x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:border)

        out_ra = Reactant.to_rarray(zeros(T, 3, 4, 2, 2, 2))
        @test Array(@jit(gs_inplace_zeros!(out_ra, x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:zeros)
        @test Array(@jit(gs_inplace_border!(out_ra, x_ra, grid_ra))) ≈
            NNlib.grid_sample(x, grid; padding_mode=:border)
    end
end

@testset "NNlib.grid_sample Enzyme AD & ∇grid_sample (4D and 5D)" begin
    # 4D Reverse + Forward AD + explicit ∇grid_sample / ∇grid_sample!
    for T in (Float32, Float64)
        x = Reactant.TestUtils.construct_test_array(T, 4, 5, 2, 2)
        # Avoid exact integer unnormalized interior points where floor() has a jump,
        # while explicitly testing exact corners (-1, 1) and out-of-bounds points!
        grid = fill(T(0.13), 2, 4, 3, 2)
        grid[1, 1, 1, 1] = T(-1.0)
        grid[2, 1, 1, 1] = T(1.0)
        grid[1, 2, 1, 1] = T(-1.8)
        grid[2, 2, 1, 1] = T(0.37)
        grid[1, 3, 2, 1] = T(-0.41)
        grid[2, 3, 2, 1] = T(0.62)
        grid[1, 4, 3, 2] = T(2.15)
        grid[2, 4, 3, 2] = T(-0.29)
        grid[1, 1, 3, 2] = T(0.73)
        grid[2, 1, 3, 2] = T(-1.0)

        w = Reactant.TestUtils.construct_test_array(T, 4, 3, 2, 2)
        dx_tangent = Reactant.TestUtils.construct_test_array(T, 4, 5, 2, 2)
        dg_tangent = Reactant.TestUtils.construct_test_array(T, 2, 4, 3, 2)

        x_ra = Reactant.to_rarray(x)
        grid_ra = Reactant.to_rarray(grid)
        w_ra = Reactant.to_rarray(w)
        dx_tan_ra = Reactant.to_rarray(dx_tangent)
        dg_tan_ra = Reactant.to_rarray(dg_tangent)

        ref_dx_z, ref_dg_z = NNlib.∇grid_sample(w, x, grid; padding_mode=:zeros)
        ref_dx_b, ref_dg_b = NNlib.∇grid_sample(w, x, grid; padding_mode=:border)

        # Enzyme Reverse AD
        ad_dx_z, ad_dg_z = @jit gs_rev_grads_zeros(x_ra, grid_ra, w_ra)
        @test Array(ad_dx_z) ≈ ref_dx_z
        @test Array(ad_dg_z) ≈ ref_dg_z

        ad_dx_b, ad_dg_b = @jit gs_rev_grads_border(x_ra, grid_ra, w_ra)
        @test Array(ad_dx_b) ≈ ref_dx_b
        @test Array(ad_dg_b) ≈ ref_dg_b

        # Enzyme Forward AD
        fwd_z = @jit gs_fwd_deriv_zeros(x_ra, dx_tan_ra, grid_ra, dg_tan_ra, w_ra)
        @test Number(fwd_z) ≈ sum(ref_dx_z .* dx_tangent) + sum(ref_dg_z .* dg_tangent)

        fwd_b = @jit gs_fwd_deriv_border(x_ra, dx_tan_ra, grid_ra, dg_tan_ra, w_ra)
        @test Number(fwd_b) ≈ sum(ref_dx_b .* dx_tangent) + sum(ref_dg_b .* dg_tangent)

        # Explicit ∇grid_sample and ∇grid_sample! (including accumulation into non-zero dx)
        exp_dx_d, exp_dg_d = @jit gs_grad_default(w_ra, x_ra, grid_ra)
        @test Array(exp_dx_d) ≈ ref_dx_z
        @test Array(exp_dg_d) ≈ ref_dg_z

        exp_dx_z, exp_dg_z = @jit gs_grad_zeros(w_ra, x_ra, grid_ra)
        @test Array(exp_dx_z) ≈ ref_dx_z
        @test Array(exp_dg_z) ≈ ref_dg_z

        exp_dx_b, exp_dg_b = @jit gs_grad_border(w_ra, x_ra, grid_ra)
        @test Array(exp_dx_b) ≈ ref_dx_b
        @test Array(exp_dg_b) ≈ ref_dg_b

        init_dx = ones(T, size(x))
        init_dg = fill(T(99), size(grid))
        cpu_dx = copy(init_dx)
        cpu_dg = copy(init_dg)
        NNlib.∇grid_sample!(cpu_dx, cpu_dg, w, x, grid, :border)
        ra_dx = Reactant.to_rarray(copy(init_dx))
        ra_dg = Reactant.to_rarray(copy(init_dg))
        out_dx, out_dg = @jit gs_grad_inplace!(ra_dx, ra_dg, w_ra, x_ra, grid_ra, :border)
        @test Array(out_dx) ≈ cpu_dx
        @test Array(out_dg) ≈ cpu_dg
    end

    # 5D Reverse + Forward AD + explicit ∇grid_sample / ∇grid_sample!
    x5 = Reactant.TestUtils.construct_test_array(Float64, 3, 4, 3, 2, 2)
    grid5 = fill(0.17, 3, 2, 3, 2, 2)
    grid5[:, 1, 1, 1, 1] .= (-1.0, 1.0, -1.0)
    grid5[:, 2, 1, 1, 1] .= (-1.75, 0.23, 0.41)
    grid5[:, 1, 3, 2, 2] .= (0.31, -0.57, 2.2)
    grid5[:, 2, 3, 2, 2] .= (-0.19, 0.64, -0.38)
    w5 = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 2, 2, 2)
    dx5_tan = Reactant.TestUtils.construct_test_array(Float64, 3, 4, 3, 2, 2)
    dg5_tan = Reactant.TestUtils.construct_test_array(Float64, 3, 2, 3, 2, 2)

    x5_ra = Reactant.to_rarray(x5)
    grid5_ra = Reactant.to_rarray(grid5)
    w5_ra = Reactant.to_rarray(w5)
    dx5_tan_ra = Reactant.to_rarray(dx5_tan)
    dg5_tan_ra = Reactant.to_rarray(dg5_tan)

    ref5_dx_z, ref5_dg_z = NNlib.∇grid_sample(w5, x5, grid5; padding_mode=:zeros)
    ref5_dx_b, ref5_dg_b = NNlib.∇grid_sample(w5, x5, grid5; padding_mode=:border)

    ad5_dx_z, ad5_dg_z = @jit gs_rev_grads_zeros(x5_ra, grid5_ra, w5_ra)
    @test Array(ad5_dx_z) ≈ ref5_dx_z
    @test Array(ad5_dg_z) ≈ ref5_dg_z

    ad5_dx_b, ad5_dg_b = @jit gs_rev_grads_border(x5_ra, grid5_ra, w5_ra)
    @test Array(ad5_dx_b) ≈ ref5_dx_b
    @test Array(ad5_dg_b) ≈ ref5_dg_b

    fwd5_z = @jit gs_fwd_deriv_zeros(x5_ra, dx5_tan_ra, grid5_ra, dg5_tan_ra, w5_ra)
    @test Number(fwd5_z) ≈ sum(ref5_dx_z .* dx5_tan) + sum(ref5_dg_z .* dg5_tan)

    fwd5_b = @jit gs_fwd_deriv_border(x5_ra, dx5_tan_ra, grid5_ra, dg5_tan_ra, w5_ra)
    @test Number(fwd5_b) ≈ sum(ref5_dx_b .* dx5_tan) + sum(ref5_dg_b .* dg5_tan)

    exp5_dx_z, exp5_dg_z = @jit gs_grad_zeros(w5_ra, x5_ra, grid5_ra)
    @test Array(exp5_dx_z) ≈ ref5_dx_z
    @test Array(exp5_dg_z) ≈ ref5_dg_z

    exp5_dx_b, exp5_dg_b = @jit gs_grad_border(w5_ra, x5_ra, grid5_ra)
    @test Array(exp5_dx_b) ≈ ref5_dx_b
    @test Array(exp5_dg_b) ≈ ref5_dg_b

    ra5_dx = Reactant.to_rarray(zeros(Float64, size(x5)))
    ra5_dg = Reactant.to_rarray(zeros(Float64, size(grid5)))
    out5_dx, out5_dg = @jit gs_grad_inplace!(ra5_dx, ra5_dg, w5_ra, x5_ra, grid5_ra, :zeros)
    @test Array(out5_dx) ≈ ref5_dx_z
    @test Array(out5_dg) ≈ ref5_dg_z
end
