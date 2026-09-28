#!/usr/bin/env julia
"""
    test_example.jl — End-to-end test of the run_mlir.jl pipeline with output verification.

Generates an execute script from the example MLIR files, then runs it with
non-zero inputs and checks the outputs match expected values.

The example MLIR computes: (dt, grid + state)
  - first_time_step: inputs (grid=[1,1,1,1], dt=3.0, state=[0,0,0,0])
                     → outputs (3.0, [1,1,1,1])
  - loop:            inputs (grid=[1,1,1,1], dt=3.0, state=[1,1,1,1], ninner=10)
                     → outputs (3.0, [2,2,2,2])

Usage:
    julia --project test_example.jl [--cpu]
"""

const USE_CPU = "--cpu" in ARGS
if USE_CPU
    ENV["CUDA_VISIBLE_DEVICES"] = ""
    ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") *
        " --xla_force_host_platform_device_count=1"
end

using Reactant
using Reactant.MLIR

if USE_CPU
    Reactant.set_default_backend("cpu")
end

const SCRIPTS_DIR = @__DIR__

# ── Load & compile ──

function compile_module(client, mlir_path; num_parameters, num_outputs, device)
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)
    mod = parse(MLIR.IR.Module, read(mlir_path, String))
    compile_opts = Reactant.XLA.make_compile_options(;
        device_id=Int64(Reactant.XLA.device_ordinal(device)),
        num_replicas=Int64(1),
        num_partitions=Int64(1),
    )
    exec = Reactant.XLA.compile(client, mod;
        compile_options=compile_opts,
        num_parameters=Int64(num_parameters),
        num_outputs=Int64(num_outputs),
        is_sharded=false,
        num_replicas=Int64(1),
        num_partitions=Int64(1),
    )
    MLIR.IR.deactivate(ctx)
    return exec
end

function get_buf_ptr(x)
    if Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
        return Reactant.XLA.synced_buffer(x.data).buffer
    else
        return Reactant.XLA.synced_buffer(only(x.data)).buffer
    end
end

function execute_and_sync(exec, inputs, n_outs, device)
    n = length(inputs)
    bufs = ntuple(i -> get_buf_ptr(inputs[i]), n)
    donated = ntuple(Returns(UInt8(0)), n)
    GC.@preserve inputs begin
        results = Reactant.XLA.execute_sharded(exec, device, bufs, donated, Val(n_outs))
    end
    # Sync all results
    for r in results
        Reactant.XLA.synced_buffer(r isa Tuple ? r[1] : r)
    end
    return results
end

function result_to_array(r)
    buf = r isa Tuple ? r[1] : r
    synced = Reactant.XLA.synced_buffer(buf)
    # Read back to host
    sz = Reactant.XLA.size(synced)
    T = Reactant.XLA.eltype(synced)
    if isempty(sz)
        data = Array{T}(undef)
        Reactant.XLA.to_host(synced, data, Reactant.Sharding.NoSharding())
        return data[]
    else
        data = Array{T}(undef, sz...)
        Reactant.XLA.to_host(synced, data, Reactant.Sharding.NoSharding())
        return data
    end
end

# ── Main ──

function main()
    println("=== test_example.jl — verifying example MLIR outputs ===")

    client = Reactant.XLA.default_backend()
    device = Reactant.XLA.default_device(client)
    println("Backend: $(Reactant.XLA.platform_name(client))")

    first_mlir = joinpath(SCRIPTS_DIR, "example_first_pre_xla_compile.mlir")
    loop_mlir  = joinpath(SCRIPTS_DIR, "example_loop_pre_xla_compile.mlir")

    # Compile
    print("Compiling first... ")
    exec_first = compile_module(client, first_mlir;
        num_parameters=3, num_outputs=2, device)
    println("OK")

    print("Compiling loop...  ")
    exec_loop = compile_module(client, loop_mlir;
        num_parameters=4, num_outputs=2, device)
    println("OK")

    # Create inputs with known values
    # MLIR: @main(grid: tensor<4xf32>, dt: tensor<f32>, state: tensor<4xf32>)
    #   → (dt, grid + state)
    grid  = Reactant.ConcreteRArray(ones(Float32, 4))
    dt    = Reactant.ConcreteRNumber{Float32}(Float32(3))
    state = Reactant.ConcreteRArray(zeros(Float32, 4))

    # Execute first_time_step
    print("Executing first_time_step... ")
    first_results = execute_and_sync(exec_first, [grid, dt, state], 2, device)

    out_dt    = result_to_array(first_results[1])
    out_state = result_to_array(first_results[2])

    @assert out_dt == Float32(3) "Expected dt=3.0, got $out_dt"
    @assert out_state ≈ ones(Float32, 4) "Expected [1,1,1,1], got $out_state"
    println("OK  (dt=$out_dt, state=$out_state)")

    # Execute loop: grid + out_state = [1,1,1,1] + [1,1,1,1] = [2,2,2,2]
    ninner = Reactant.ConcreteRNumber{Int64}(Int64(10))
    out_state_arr = Reactant.ConcreteRArray(out_state)

    print("Executing loop...           ")
    loop_results = execute_and_sync(exec_loop,
        [grid, dt, out_state_arr, ninner], 2, device)

    loop_dt    = result_to_array(loop_results[1])
    loop_state = result_to_array(loop_results[2])

    @assert loop_dt == Float32(3) "Expected dt=3.0, got $loop_dt"
    @assert loop_state ≈ fill(Float32(2), 4) "Expected [2,2,2,2], got $loop_state"
    println("OK  (dt=$loop_dt, state=$loop_state)")

    println("\n=== ALL CHECKS PASSED ===")
end

main()
