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
