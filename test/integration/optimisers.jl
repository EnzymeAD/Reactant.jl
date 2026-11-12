using Test, Reactant, Optimisers

is_empty_buffer(x::ConcreteIFRTNumber) = x.data === C_NULL
is_empty_buffer(x::ConcretePJRTNumber) = any(x === C_NULL for x in x.data)

@testset "No Empty Buffers #861" begin
    opt = Descent(Reactant.to_rarray(0.1; track_numbers=Number))
    x = Reactant.to_rarray((ones(10), ones(3), ones(5)))
    opt_state = @jit Optimisers.setup(opt, x)

    @test !is_empty_buffer(opt_state[1].rule.eta)
    @test !is_empty_buffer(opt_state[2].rule.eta)
    @test !is_empty_buffer(opt_state[3].rule.eta)
end

@testset "Correct Aliasing" begin
    ps = Reactant.to_rarray((a=ones(4), b=ones(2), c=ones(4)))
    opt = Reactant.to_rarray(Descent(0.001f0); track_numbers=true)

    st_opt = @jit Optimisers.setup(opt, ps)

    @test st_opt.a.rule.eta.data === st_opt.b.rule.eta.data

    gs = Reactant.to_rarray((a=ones(4), b=ones(2), c=ones(4)))

    hlo = @code_hlo Optimisers.update(st_opt, ps, gs)
    @test length(findall("stablehlo.broadcast_in_dim", repr(hlo))) == 2
end

@testset "Eager & Compiled Adam and RAdam (#3339)" begin
    for Opt in (Optimisers.Adam, Optimisers.RAdam)
        ps_jl = (a=Float64[1.0, 2.0, 3.0], b=Float64[4.0, 5.0])
        gs_jl = (a=Float64[0.1, -0.2, 0.3], b=Float64[-0.4, 0.5])

        st_jl = Optimisers.setup(Opt(), deepcopy(ps_jl))
        st_jl, ps_jl_new = Optimisers.update(st_jl, deepcopy(ps_jl), gs_jl)

        # Eager update on ConcreteRArray (hits lazy nested broadcast in `subtract!`)
        ps_ra = Reactant.to_rarray(deepcopy(ps_jl))
        gs_ra = Reactant.to_rarray(deepcopy(gs_jl))
        st_ra = Optimisers.setup(Opt(), ps_ra)
        st_ra, ps_ra_new = Optimisers.update!(st_ra, ps_ra, gs_ra)
        @test Array(ps_ra_new.a) ≈ ps_jl_new.a
        @test Array(ps_ra_new.b) ≈ ps_jl_new.b

        # Compiled update
        ps_ra2 = Reactant.to_rarray(deepcopy(ps_jl))
        gs_ra2 = Reactant.to_rarray(deepcopy(gs_jl))
        st_ra2 = @jit Optimisers.setup(Opt(), ps_ra2)
        st_ra2, ps_ra2_new = @jit Optimisers.update(st_ra2, ps_ra2, gs_ra2)
        @test Array(ps_ra2_new.a) ≈ ps_jl_new.a
        @test Array(ps_ra2_new.b) ≈ ps_jl_new.b
    end
end
