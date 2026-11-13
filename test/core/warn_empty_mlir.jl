using Reactant, Test

host_add_one(x) = x .+ 1.0
identity_fn(x) = x
zero_arg_iota() = Reactant.Ops.iota(Int64, [4]; iota_dimension=1)

function side_effect_barrier_only()
    Reactant.Ops.optimization_barrier()
    return nothing
end

@testset "Warn when no MLIR code is emitted (#590)" begin
    x_host = [1.0, 2.0, 3.0]
    thunk = @test_logs (:warn, r"no MLIR code was emitted") match_mode = :any Reactant.compile(
        host_add_one, (x_host,)
    )
    @test thunk(x_host) == [2.0, 3.0, 4.0]

    res_jit = @test_logs (:warn, r"no MLIR code was emitted") match_mode = :any @jit(
        host_add_one(x_host)
    )
    @test res_jit == [2.0, 3.0, 4.0]

    # Traced ConcreteRArray input (even identity) should NOT warn
    x_ra = Reactant.to_rarray(x_host)
    id_thunk = @test_nowarn Reactant.compile(identity_fn, (x_ra,))
    @test Array(id_thunk(x_ra)) == x_host

    add_thunk = @test_nowarn Reactant.compile(host_add_one, (x_ra,))
    @test Array(add_thunk(x_ra)) ≈ [2.0, 3.0, 4.0]

    # Zero-argument function that emits MLIR ops and returns a TracedRArray should NOT warn
    iota_thunk = @test_nowarn Reactant.compile(zero_arg_iota, ())
    @test Array(iota_thunk()) == Int64[0, 1, 2, 3]
end
