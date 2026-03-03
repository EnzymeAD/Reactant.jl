#!/usr/bin/env julia
"""
    run_mlir_v2.jl — Generator (Reactant-dependent): MLIR IR introspection → emit execute script.

Usage:
    julia --project=Reactant.jl run_mlir_v2.jl [first.mlir] [loop.mlir] [output.jl]

Same two-phase approach as run_mlir.jl, but uses Reactant's MLIR IR APIs
instead of regex to parse signatures. Compare with run_mlir.jl (regex-only).
"""

using Reactant
using Reactant.MLIR
using Reactant.MLIR.IR

# ──────────────────────────────────────────────────────────────
# MLIR IR introspection (replaces ~100 lines of regex in v1)
# ──────────────────────────────────────────────────────────────

struct TensorSig
    eltype::String
    mlir_shape::Vector{Int}
    mlir_sharding::Vector{Symbol}  # per Julia dim: :_none = replicated, :x/:y = axis
end

# TODO: upstream haskey/get for dictionary Attributes into Reactant's MLIR IR bindings
# (Attribute.jl). getattr on Operation already returns nothing for missing keys,
# but dict["key"] wraps a null pointer instead. See also: getindex fix for isdict branch.
"""Get a named element from a dictionary attribute, returning nothing if absent."""
function dict_get(dict::IR.Attribute, name::String)
    raw = MLIR.API.mlirDictionaryAttrGetElementByName(dict, name)
    return raw.ptr == C_NULL ? nothing : IR.Attribute(raw)
end

function get_main_func(mod::IR.Module)
    for op in IR.body(mod)
        IR.name(op) == "func.func" || continue
        attr = IR.getattr(op, "sym_name")
        attr !== nothing && String(attr) == "main" && return op
    end
    error("No func.func @main found in module")
end

"""Extract mesh axes/sizes from an sdy.mesh operation using the sdy C API."""
function extract_mesh_spec(mod::IR.Module)
    mesh_axes = Symbol[]
    mesh_sizes = Int[]
    for op in IR.body(mod)
        IR.name(op) == "sdy.mesh" || continue
        mesh_attr = IR.getattr(op, "mesh")
        mesh_attr === nothing && continue
        naxes = MLIR.API.sdyMeshAttrGetAxesSize(mesh_attr)
        for i in 1:naxes
            axis_attr = IR.Attribute(MLIR.API.sdyMeshAttrGetAxesElem(mesh_attr, i - 1))
            push!(mesh_axes, Symbol(String(MLIR.API.sdyMeshAxisAttrGetName(axis_attr))))
            push!(mesh_sizes, Int(MLIR.API.sdyMeshAxisAttrGetSize(axis_attr)))
        end
        break
    end
    return mesh_axes, mesh_sizes
end

"""Extract sharding partition spec from an sdy.sharding attribute using the sdy C API.
Returns per-MLIR-dim symbols (before Julia reversal), e.g. [:_none, :y, :x]."""
function extract_sharding_spec(sdy_attr)
    MLIR.API.sdyAttributeIsATensorShardingAttr(sdy_attr) || return Symbol[]
    ndims = MLIR.API.sdyTensorShardingAttrGetDimShardingsSize(sdy_attr)
    spec = Vector{Symbol}(undef, ndims)
    for i in 1:ndims
        dim_attr = IR.Attribute(
            MLIR.API.sdyTensorShardingAttrGetDimShardingsElem(sdy_attr, i - 1))
        naxes = MLIR.API.sdyDimensionShardingAttrGetAxesSize(dim_attr)
        if naxes == 0
            spec[i] = :_none
        else
            axis_attr = IR.Attribute(
                MLIR.API.sdyDimensionShardingAttrGetAxesElem(dim_attr, 0))
            spec[i] = Symbol(String(MLIR.API.sdyAxisRefAttrGetName(axis_attr)))
        end
    end
    return spec
end

function analyze_module(mlir_string::String)
    mod = parse(IR.Module, mlir_string)
    main_op = get_main_func(mod)
    ftype = IR.FunctionType(main_op)

    # Input types & shardings
    n_in = IR.ninputs(ftype)
    inputs = TensorSig[]
    arg_attrs = IR.getattr(main_op, "arg_attrs")

    n_grid_const = 0
    found_alias = false
    for i in 1:n_in
        mlir_type = IR.input(ftype, i)
        T = string(IR.julia_type(IR.eltype(mlir_type)))
        shape = IR.istensor(mlir_type) && IR.ndims(mlir_type) > 0 ?
            collect(Int, IR.size(mlir_type)) : Int[]

        sharding = Symbol[]
        has_alias = false
        if arg_attrs !== nothing && IR.isarray(arg_attrs)
            dict = arg_attrs[i - 1]  # 0-based C API
            if IR.isdict(dict)
                sdy_attr = dict_get(dict, "sdy.sharding")
                if sdy_attr !== nothing
                    sharding = extract_sharding_spec(sdy_attr)
                end
                has_alias = dict_get(dict, "tf.aliasing_output") !== nothing
            end
        end
        if !found_alias && !has_alias
            n_grid_const += 1
        else
            found_alias = true
        end
        push!(inputs, TensorSig(T, shape, sharding))
    end

    # Output types
    n_out = IR.nresults(ftype)
    outputs = [begin
        t = IR.result(ftype, i)
        T = string(IR.julia_type(IR.eltype(t)))
        shape = IR.istensor(t) && IR.ndims(t) > 0 ? collect(Int, IR.size(t)) : Int[]
        TensorSig(T, shape, Symbol[])
    end for i in 1:n_out]

    # Module attributes
    mod_op = IR.Operation(mod)
    np = IR.getattr(mod_op, "mhlo.num_partitions")
    num_partitions = np !== nothing ? Int(np) : 1
    nr = IR.getattr(mod_op, "mhlo.num_replicas")
    num_replicas = nr !== nothing ? Int(nr) : 1

    # Mesh spec via sdy C API
    mesh_axes, mesh_sizes = extract_mesh_spec(mod)

    return (; inputs, outputs, n_grid_const, num_partitions, num_replicas,
              mesh_axes, mesh_sizes)
end

# ──────────────────────────────────────────────────────────────
# Code generation helpers (same as v1)
# ──────────────────────────────────────────────────────────────

function julia_shape_str(sig::TensorSig)
    isempty(sig.mlir_shape) && return "()"
    return string(tuple(reverse(sig.mlir_shape)...))
end

function julia_sharding_expr(sig::TensorSig)
    isempty(sig.mlir_sharding) && return "Sharding.Replicated(mesh)"
    spec = reverse(sig.mlir_sharding)
    parts = [s == :_none ? "nothing" : ":$s" for s in spec]
    all(p -> p == "nothing", parts) && return "Sharding.Replicated(mesh)"
    return "Sharding.NamedSharding(mesh, ($(join(parts, ", ")),))"
end

function codegen_create_input(i::Int, sig::TensorSig; is_sharded::Bool)
    T = sig.eltype
    shard_kw = is_sharded ? "; sharding=$(julia_sharding_expr(sig))" : ""
    if isempty(sig.mlir_shape)
        default_val = T in ("Float32", "Float64", "Float16") ? "$T(60)" : "$T(0)"
        return "    ConcreteRNumber{$T}($default_val$shard_kw)"
    else
        shape = julia_shape_str(sig)
        return "    ConcreteRArray(randn($T, $shape...)$shard_kw)"
    end
end

# ──────────────────────────────────────────────────────────────
# Script generation (identical to v1 from here down)
# ──────────────────────────────────────────────────────────────

function generate_script(;
    first_path, loop_path,
    first_in, first_out, loop_in, loop_out,
    mesh_axes, mesh_sizes, num_partitions, num_replicas,
    n_grid_const, output_dir,
)
    n_first_in  = length(first_in)
    n_first_out = length(first_out)
    n_loop_in   = length(loop_in)
    n_loop_out  = length(loop_out)
    ndevices    = prod(mesh_sizes)

    first_rel = relpath(first_path, output_dir)
    loop_rel  = relpath(loop_path, output_dir)
    mesh_names_str = "($(join([":" * string(a) for a in mesh_axes], ", ")),)"
    mesh_shape_str = join(mesh_sizes, ", ")
    is_sharded = num_partitions > 1

    input_lines = [codegen_create_input(i, sig; is_sharded) * ",  # arg$(i-1)"
                   for (i, sig) in enumerate(first_in)]
    summary_lines = ["#   arg$(i-1): $(sig.eltype) $(julia_shape_str(sig))"
                     for (i, sig) in enumerate(first_in)]

    io = IOBuffer()
    W(s) = println(io, s)

    W("#!/usr/bin/env julia")
    W("#")
    W("# Generated by run_mlir_v2.jl — do not edit by hand.")
    W("#")
    W("# Standalone XLA compile + execute for pre_xla MLIR modules.")
    W("# Requires $ndevices device(s) (num_partitions=$num_partitions, num_replicas=$num_replicas).")
    W("#")
    W("# Pass --cpu to run on $ndevices virtual CPU devices instead of GPUs:")
    W("#   julia --project=Reactant.jl <script>.jl --cpu")
    W("#")
    W("# Signature (first_time_step): $n_first_in inputs → $n_first_out outputs")
    W("# Signature (loop):            $n_loop_in inputs → $n_loop_out outputs")
    W("#")
    for l in summary_lines; W(l); end
    W("#")
    W("")
    W("# ── CPU mode (must be set before `using Reactant`) ──")
    W("const USE_CPU = \"--cpu\" in ARGS")
    W("if USE_CPU")
    W("    ENV[\"CUDA_VISIBLE_DEVICES\"] = \"\"")
    W("    ENV[\"XLA_FLAGS\"] = get(ENV, \"XLA_FLAGS\", \"\") *")
    W("        \" --xla_force_host_platform_device_count=$ndevices\"")
    W("end")
    W("")
    W("using Reactant")
    W("using Reactant.MLIR")
    W("")
    W("if USE_CPU")
    W("    Reactant.set_default_backend(\"cpu\")")
    W("end")
    W("")
    W("const IS_IFRT = Reactant.XLA.REACTANT_XLA_RUNTIME == \"IFRT\"")
    W("")
    W("# ── MLIR file paths (relative to this script) ──")
    W("")
    W("const FIRST_MLIR = joinpath(@__DIR__, $(repr(first_rel)))")
    W("const LOOP_MLIR  = joinpath(@__DIR__, $(repr(loop_rel)))")
    W("")
    W("# ── Module constants ──")
    W("")
    W("const NUM_PARTITIONS = $num_partitions")
    W("const NUM_REPLICAS   = $num_replicas")
    W("const NDEVICES       = $ndevices")
    W("const N_GRID_CONST   = $n_grid_const")
    W("const N_FIRST_IN     = $n_first_in")
    W("const N_FIRST_OUT    = $n_first_out")
    W("const N_LOOP_IN      = $n_loop_in")
    W("const N_LOOP_OUT     = $n_loop_out")
    W("")
    W("const IS_SHARDED = $(is_sharded)")
    W("")
    W("# ── Mesh & mock data ──")
    W("")
    if is_sharded
        W("function create_mesh(devs)")
        W("    Sharding = Reactant.Sharding")
        W("    return Sharding.Mesh(reshape(devs[1:NDEVICES], $mesh_shape_str), $mesh_names_str)")
        W("end")
        W("")
        W("function create_first_inputs(mesh)")
        W("    Sharding = Reactant.Sharding")
    else
        W("function create_first_inputs()")
    end
    W("    ConcreteRArray  = Reactant.ConcreteRArray")
    W("    ConcreteRNumber = Reactant.ConcreteRNumber")
    W("    return [")
    for l in input_lines; W(l); end
    W("    ]")
    W("end")
    W("")
    if is_sharded
        W("function create_ninner(mesh)")
        W("    Sharding = Reactant.Sharding")
        W("    return Reactant.ConcreteRNumber{Int64}(Int64(10);")
        W("        sharding=Sharding.Replicated(mesh))")
        W("end")
    else
        W("function create_ninner()")
        W("    return Reactant.ConcreteRNumber{Int64}(Int64(10))")
        W("end")
    end
    W("")

    # Static runtime code
    print(io, raw"""
# ── Buffer helpers ──

function get_buf_ptr(x)
    if IS_IFRT
        return Reactant.XLA.synced_buffer(x.data).buffer
    else
        return Reactant.XLA.synced_buffer(only(x.data)).buffer
    end
end

function build_inputs_unsharded(inputs)
    n = length(inputs)
    bufs = ntuple(i -> get_buf_ptr(inputs[i]), n)
    donated = ntuple(Returns(UInt8(0)), n)
    return bufs, donated
end

function build_ifrt_inputs(inputs)
    n = length(inputs)
    bufs = ntuple(i -> Reactant.XLA.synced_buffer(inputs[i].data).buffer, n)
    donated = ntuple(Returns(UInt8(0)), n)
    return bufs, donated
end

function build_pjrt_inputs(inputs, ndevices)
    ptrs = Ptr{Cvoid}[]
    for j in 1:ndevices
        for arg in inputs
            push!(ptrs, Reactant.XLA.synced_buffer(arg.data[j]).buffer)
        end
    end
    n = length(ptrs)
    return ntuple(i -> ptrs[i], n), ntuple(Returns(UInt8(0)), n)
end

function xla_execute(exec, inputs, n_outs::Int; device=nothing)
    if !IS_SHARDED
        bufs, donated = build_inputs_unsharded(inputs)
        GC.@preserve inputs begin
            return Reactant.XLA.execute_sharded(exec, device, bufs, donated, Val(n_outs))
        end
    elseif IS_IFRT
        bufs, donated = build_ifrt_inputs(inputs)
        GC.@preserve inputs begin
            return Reactant.XLA.execute(exec, bufs, donated, Val(n_outs), Val(NDEVICES))
        end
    else
        bufs, donated = build_pjrt_inputs(inputs, NDEVICES)
        GC.@preserve inputs begin
            return Reactant.XLA.execute(exec, bufs, donated, Val(n_outs), Val(NDEVICES))
        end
    end
end

function sync_results(results)
    for r in results
        Reactant.XLA.synced_buffer(r isa Tuple ? r[1] : r)
    end
end

# ── Compile ──

function compile_module(client, mlir_path; num_parameters, num_outputs, device=nothing)
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)
    mod = parse(MLIR.IR.Module, read(mlir_path, String))

    device_id = IS_SHARDED ? Int64(-1) : Int64(Reactant.XLA.device_ordinal(device))
    compile_opts = Reactant.XLA.make_compile_options(;
        device_id,
        num_replicas=Int64(NUM_REPLICAS),
        num_partitions=Int64(NUM_PARTITIONS),
        mesh_ids=IS_SHARDED ? collect(Int64, 0:(NDEVICES - 1)) : nothing,
        xla_executable_build_options=(;
            use_shardy_partitioner=IS_SHARDED,
            use_spmd_partitioning=IS_SHARDED,
        ),
    )
    exec = Reactant.XLA.compile(client, mod;
        compile_options=compile_opts,
        num_parameters=Int64(num_parameters),
        num_outputs=Int64(num_outputs),
        is_sharded=IS_SHARDED,
        num_replicas=Int64(NUM_REPLICAS),
        num_partitions=Int64(NUM_PARTITIONS),
    )
    MLIR.IR.deactivate(ctx)
    return exec
end

# ── Marshal first → loop ──

function marshal_loop_inputs(mock_inputs, first_results, ninner)
    if !IS_SHARDED || IS_IFRT
        lp = Vector{Ptr{Cvoid}}(undef, N_LOOP_IN)
        for i in 1:N_GRID_CONST
            lp[i] = get_buf_ptr(mock_inputs[i])
        end
        for i in 1:N_FIRST_OUT
            buf = first_results[i]
            lp[N_GRID_CONST + i] = IS_SHARDED ?
                Reactant.XLA.synced_buffer(buf).buffer :
                (buf isa Tuple ? Reactant.XLA.synced_buffer(buf[1]).buffer :
                                Reactant.XLA.synced_buffer(buf).buffer)
        end
        lp[N_LOOP_IN] = get_buf_ptr(ninner)
        return ntuple(i -> lp[i], N_LOOP_IN), ntuple(Returns(UInt8(0)), N_LOOP_IN)
    else
        lp = Ptr{Cvoid}[]
        for j in 1:NDEVICES
            for i in 1:N_GRID_CONST
                push!(lp, Reactant.XLA.synced_buffer(mock_inputs[i].data[j]).buffer)
            end
            for i in 1:N_FIRST_OUT
                push!(lp, Reactant.XLA.synced_buffer(first_results[i][j]).buffer)
            end
            push!(lp, Reactant.XLA.synced_buffer(ninner.data[j]).buffer)
        end
        n = length(lp)
        return ntuple(i -> lp[i], n), ntuple(Returns(UInt8(0)), n)
    end
end

# ── Main ──

function main()
    println("=== MLIR → XLA compile → execute ===")
    println("Runtime:  $(Reactant.XLA.REACTANT_XLA_RUNTIME)")

    client = Reactant.XLA.default_backend()
    devs   = Reactant.devices()
    println("Backend:  $(Reactant.XLA.platform_name(client))")
    println("Devices:  $(length(devs))")
    IS_SHARDED && length(devs) < NDEVICES && error(
        "Need $(NDEVICES) devices but only $(length(devs)) available.")

    device = IS_SHARDED ? nothing : Reactant.XLA.default_device(client)

    # Smoke test
    print("Smoke test... ")
    x = Reactant.ConcreteRArray(ones(Float32, 4))
    y = @jit identity(x)
    @assert Array(y) ≈ ones(Float32, 4)
    println("OK")

    # Mesh & mock data
    println("\nCreating mock data...")
    if IS_SHARDED
        mesh = create_mesh(devs)
        mock_inputs = create_first_inputs(mesh)
        ninner = create_ninner(mesh)
    else
        mock_inputs = create_first_inputs()
        ninner = create_ninner()
    end
    println("  $(length(mock_inputs)) inputs for first_time_step")

    # Compile
    println("\nCompiling first_time_step...")
    t0 = time()
    exec_first = compile_module(client, FIRST_MLIR;
        num_parameters=N_FIRST_IN, num_outputs=N_FIRST_OUT, device)
    println("  done ($(round(time() - t0; digits=1))s)")

    println("Compiling loop...")
    t0 = time()
    exec_loop = compile_module(client, LOOP_MLIR;
        num_parameters=N_LOOP_IN, num_outputs=N_LOOP_OUT, device)
    println("  done ($(round(time() - t0; digits=1))s)")

    # Execute first_time_step
    println("\nExecuting first_time_step...")
    t0 = time()
    first_results = xla_execute(exec_first, mock_inputs, N_FIRST_OUT; device)
    sync_results(first_results)
    println("  $(N_FIRST_OUT) outputs ($(round(time() - t0; digits=1))s)")

    # Marshal & execute loop
    println("\nMarshaling → loop...")
    loop_bufs, loop_donated = marshal_loop_inputs(mock_inputs, first_results, ninner)
    println("  $(length(loop_bufs)) buffer pointers")

    println("Executing loop...")
    t0 = time()
    GC.@preserve mock_inputs first_results ninner begin
        if IS_SHARDED
            loop_results = Reactant.XLA.execute(
                exec_loop, loop_bufs, loop_donated,
                Val(N_LOOP_OUT), Val(NDEVICES),
            )
        else
            loop_results = Reactant.XLA.execute_sharded(
                exec_loop, device, loop_bufs, loop_donated, Val(N_LOOP_OUT),
            )
        end
    end
    sync_results(loop_results)
    println("  $(N_LOOP_OUT) outputs ($(round(time() - t0; digits=1))s)")

    println("\n=== SUCCESS ===")
    return first_results, loop_results
end

main()
""")

    return String(take!(io))
end

# ──────────────────────────────────────────────────────────────
# Main: analyze MLIR via IR APIs, emit script
# ──────────────────────────────────────────────────────────────

function main()
    default_first = joinpath(@__DIR__,
        "mlir_artifacts/tmp/reactant_I06p6L",
        "module_430_reactant_first_t..._pre_xla_compile.mlir")
    default_loop = joinpath(@__DIR__,
        "mlir_artifacts/tmp/reactant_I06p6L",
        "module_446_reactant_loop!_pre_xla_compile.mlir")
    default_out = joinpath(@__DIR__, "gb25_execute.jl")

    first_path = length(ARGS) >= 1 ? ARGS[1] : default_first
    loop_path  = length(ARGS) >= 2 ? ARGS[2] : default_loop
    out_path   = length(ARGS) >= 3 ? ARGS[3] : default_out

    println("=== run_mlir_v2.jl — MLIR IR → script generator ===")
    println("First time step: $first_path")
    println("Loop:            $loop_path")
    println("Output script:   $out_path")

    # Use Reactant's MLIR IR to analyze modules
    ctx = Reactant.ReactantContext()
    IR.activate(ctx)

    println("\nAnalyzing MLIR modules...")
    first = analyze_module(read(first_path, String))
    loop  = analyze_module(read(loop_path, String))

    IR.deactivate(ctx)

    println("  first: $(length(first.inputs)) in → $(length(first.outputs)) out " *
            "($(first.n_grid_const) grid constants)")
    println("  loop:  $(length(loop.inputs)) in → $(length(loop.outputs)) out")
    println("  mesh:  $(collect(zip(first.mesh_axes, first.mesh_sizes)))")
    println("  partitions=$(first.num_partitions) replicas=$(first.num_replicas)")

    for (i, sig) in enumerate(first.inputs)
        jshape = julia_shape_str(sig)
        shard = julia_sharding_expr(sig)
        println("    arg$(i-1): $(sig.eltype) $jshape  $shard")
    end

    # Generate
    output_dir = dirname(abspath(out_path))
    script = generate_script(;
        first_path=abspath(first_path),
        loop_path=abspath(loop_path),
        first_in=first.inputs, first_out=first.outputs,
        loop_in=loop.inputs, loop_out=loop.outputs,
        mesh_axes=first.mesh_axes, mesh_sizes=first.mesh_sizes,
        first.num_partitions, first.num_replicas,
        first.n_grid_const,
        output_dir,
    )

    write(out_path, script)
    println("\nWrote $(out_path) ($(count(==('\n'), script) + 1) lines)")
    println("Run with:  julia --project=Reactant.jl $(basename(out_path))")
end

main()
