"""
    MLIRRunner

Generate standalone Julia scripts that compile and execute pre-XLA MLIR modules
via Reactant's XLA backend.

The main entry point is [`generate_mlir_runner`](@ref), which accepts N MLIR files
and emits a self-contained Julia script. Marshaling between sequential modules
is driven automatically by `tf.aliasing_output` attributes in the MLIR IR.
"""
module MLIRRunner

using ..Reactant: Reactant, MLIR
using ..Reactant.MLIR: IR

# ──────────────────────────────────────────────────────────────
# MLIR IR introspection
# ──────────────────────────────────────────────────────────────

struct TensorSig
    eltype::String
    mlir_shape::Vector{Int}
    mlir_sharding::Vector{Symbol}  # per Julia dim: :_none = replicated, :x/:y = axis
end

"""Get a named element from a dictionary attribute, returning `nothing` if absent."""
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

"""Extract mesh axes/sizes from an `sdy.mesh` operation."""
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

"""Extract per-dimension sharding spec from an `sdy.sharding` attribute."""
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

"""
    analyze_module(mlir_string::String)

Parse an MLIR module string and extract:
- `inputs`/`outputs`: `Vector{TensorSig}` with type, shape, sharding info
- `alias_map`: `Dict{Int,Int}` mapping output index → input index (1-based),
  derived from `tf.aliasing_output` attributes
- `num_partitions`, `num_replicas`: module-level attributes
- `mesh_axes`, `mesh_sizes`: from `sdy.mesh`
"""
function analyze_module(mlir_string::String)
    mod = parse(IR.Module, mlir_string)
    main_op = get_main_func(mod)
    ftype = IR.FunctionType(main_op)

    # Input types, shardings, and alias map
    n_in = IR.ninputs(ftype)
    inputs = TensorSig[]
    alias_map = Dict{Int,Int}()  # output_idx (1-based) → input_idx (1-based)
    arg_attrs = IR.getattr(main_op, "arg_attrs")

    for i in 1:n_in
        mlir_type = IR.input(ftype, i)
        T = string(IR.julia_type(IR.eltype(mlir_type)))
        shape = IR.istensor(mlir_type) && IR.ndims(mlir_type) > 0 ?
            collect(Int, IR.size(mlir_type)) : Int[]

        sharding = Symbol[]
        if arg_attrs !== nothing && IR.isarray(arg_attrs)
            dict = arg_attrs[i - 1]  # 0-based C API
            if IR.isdict(dict)
                sdy_attr = dict_get(dict, "sdy.sharding")
                if sdy_attr !== nothing
                    sharding = extract_sharding_spec(sdy_attr)
                end
                alias_attr = dict_get(dict, "tf.aliasing_output")
                if alias_attr !== nothing
                    out_idx = Int(alias_attr) + 1  # 0-based → 1-based
                    alias_map[out_idx] = i
                end
            end
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

    # Mesh spec
    mesh_axes, mesh_sizes = extract_mesh_spec(mod)

    return (; inputs, outputs, alias_map, num_partitions, num_replicas,
              mesh_axes, mesh_sizes)
end

# ──────────────────────────────────────────────────────────────
# Code generation helpers
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
# Script generation — supports N modules with auto-marshaling
# ──────────────────────────────────────────────────────────────

"""
    generate_mlir_runner(mlir_files::Vector{String}; output_path::String)

Analyze N pre-XLA MLIR modules and generate a standalone Julia script that
compiles and executes them sequentially, automatically marshaling outputs→inputs
between modules using `tf.aliasing_output` attributes.

Each module's `tf.aliasing_output` annotations specify which output replaces
which input in the next call. Inputs without an alias are treated as constants
and carried forward unchanged.
"""
function generate_mlir_runner(
    mlir_files::Vector{String};
    output_path::String,
)
    isempty(mlir_files) && error("At least one MLIR file is required")
    output_dir = dirname(abspath(output_path))

    # Analyze all modules
    ctx = Reactant.ReactantContext()
    IR.activate(ctx)

    modules = []
    for path in mlir_files
        info = analyze_module(read(path, String))
        push!(modules, (; path=abspath(path), info...))
    end

    IR.deactivate(ctx)

    # Use first module for mesh/partition info
    first_mod = modules[1]
    mesh_axes = first_mod.mesh_axes
    mesh_sizes = first_mod.mesh_sizes
    num_partitions = first_mod.num_partitions
    num_replicas = first_mod.num_replicas
    ndevices = max(1, prod(mesh_sizes))
    is_sharded = num_partitions > 1

    script = _generate_script(
        modules, output_dir;
        mesh_axes, mesh_sizes, num_partitions, num_replicas,
        ndevices, is_sharded,
    )

    write(output_path, script)
    return output_path
end

function _generate_script(
    modules, output_dir;
    mesh_axes, mesh_sizes, num_partitions, num_replicas,
    ndevices, is_sharded,
)
    io = IOBuffer()
    W(s) = println(io, s)

    first_mod = modules[1]
    first_in = first_mod.inputs

    mesh_names_str = "($(join([":" * string(a) for a in mesh_axes], ", ")),)"
    mesh_shape_str = join(mesh_sizes, ", ")

    input_lines = [codegen_create_input(i, sig; is_sharded) * ",  # arg$(i-1)"
                   for (i, sig) in enumerate(first_in)]

    # Header
    W("#!/usr/bin/env julia")
    W("#")
    W("# Generated by Reactant.Serialization.generate_mlir_runner — do not edit by hand.")
    W("#")
    W("# Standalone XLA compile + execute for $(length(modules)) pre-XLA MLIR module(s).")
    W("# Requires $ndevices device(s) (num_partitions=$num_partitions, num_replicas=$num_replicas).")
    W("#")
    W("# Pass --cpu to run on $ndevices virtual CPU devices instead of GPUs:")
    W("#   julia --project=Reactant.jl <script>.jl --cpu")
    W("#")
    for (k, m) in enumerate(modules)
        W("# Module $k: $(length(m.inputs)) inputs → $(length(m.outputs)) outputs")
    end
    W("#")
    W("")

    # CPU mode
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

    # MLIR file paths
    W("# MLIR file paths (relative to this script)")
    for (k, m) in enumerate(modules)
        rel = relpath(m.path, output_dir)
        W("const MLIR_PATH_$k = joinpath(@__DIR__, $(repr(rel)))")
    end
    W("")

    # Module constants
    W("const NUM_PARTITIONS = $num_partitions")
    W("const NUM_REPLICAS   = $num_replicas")
    W("const NDEVICES       = $ndevices")
    W("const IS_SHARDED = $(is_sharded)")
    W("")

    # Per-module info
    for (k, m) in enumerate(modules)
        W("const N_IN_$k  = $(length(m.inputs))")
        W("const N_OUT_$k = $(length(m.outputs))")
    end
    W("")

    # Alias maps
    for (k, m) in enumerate(modules)
        if !isempty(m.alias_map)
            pairs = sort(collect(m.alias_map))
            W("const ALIAS_MAP_$k = Dict{Int,Int}($(join(["$o => $i" for (o,i) in pairs], ", ")))")
        else
            W("const ALIAS_MAP_$k = Dict{Int,Int}()")
        end
    end
    W("")

    # Constant indices: inputs of first module that are NOT aliased targets
    aliased_inputs_1 = Set(values(first_mod.alias_map))
    const_indices = [i for i in 1:length(first_in) if i ∉ aliased_inputs_1]
    W("const CONST_INDICES = $(const_indices)")
    W("")

    # Mesh & mock data
    if is_sharded
        W("function create_mesh(devs)")
        W("    Sharding = Reactant.Sharding")
        W("    return Sharding.Mesh(reshape(devs[1:NDEVICES], $mesh_shape_str), $mesh_names_str)")
        W("end")
        W("")
        W("function create_inputs(mesh)")
        W("    Sharding = Reactant.Sharding")
    else
        W("function create_inputs()")
    end
    W("    ConcreteRArray  = Reactant.ConcreteRArray")
    W("    ConcreteRNumber = Reactant.ConcreteRNumber")
    W("    return [")
    for l in input_lines; W(l); end
    W("    ]")
    W("end")
    W("")

    # Extra inputs for later modules (args beyond what the first module has)
    for (k, m) in enumerate(modules)
        k == 1 && continue
        n_extra = length(m.inputs) - length(first_in)
        if n_extra > 0
            extra_sigs = m.inputs[length(first_in)+1:end]
            if is_sharded
                W("function create_extra_inputs_$k(mesh)")
                W("    Sharding = Reactant.Sharding")
            else
                W("function create_extra_inputs_$k()")
            end
            W("    ConcreteRArray  = Reactant.ConcreteRArray")
            W("    ConcreteRNumber = Reactant.ConcreteRNumber")
            W("    return [")
            for (j, sig) in enumerate(extra_sigs)
                idx = length(first_in) + j
                W(codegen_create_input(idx, sig; is_sharded) * ",  # extra arg$(idx-1)")
            end
            W("    ]")
            W("end")
            W("")
        end
    end

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

# ── Marshal between modules using alias maps ──

function marshal_next_inputs(
    mock_inputs, prev_results, alias_map, const_indices, extra_inputs;
    n_in,
)
    next = Vector{Any}(undef, n_in)
    n_base = length(mock_inputs)

    # Constants from first module carry forward
    for idx in const_indices
        if idx <= n_base
            next[idx] = mock_inputs[idx]
        end
    end

    # Aliased: output K → input J
    for (out_idx, in_idx) in alias_map
        if in_idx <= n_in
            buf = prev_results[out_idx]
            next[in_idx] = buf
        end
    end

    # Extra inputs beyond first module's arg count
    for (j, extra) in enumerate(extra_inputs)
        next[n_base + j] = extra
    end

    return next
end

function marshal_bufs(next_inputs, mock_inputs; is_raw_result=false)
    if !IS_SHARDED || IS_IFRT
        n = length(next_inputs)
        lp = Vector{Ptr{Cvoid}}(undef, n)
        for i in 1:n
            v = next_inputs[i]
            if is_raw_result && !(v isa Reactant.ConcreteRArray || v isa Reactant.ConcreteRNumber)
                buf = v isa Tuple ? v[1] : v
                lp[i] = IS_SHARDED ?
                    Reactant.XLA.synced_buffer(buf).buffer :
                    Reactant.XLA.synced_buffer(buf).buffer
            else
                lp[i] = get_buf_ptr(v)
            end
        end
        return ntuple(i -> lp[i], n), ntuple(Returns(UInt8(0)), n)
    else
        ptrs = Ptr{Cvoid}[]
        for j in 1:NDEVICES
            for v in next_inputs
                if !(v isa Reactant.ConcreteRArray || v isa Reactant.ConcreteRNumber)
                    push!(ptrs, Reactant.XLA.synced_buffer(v[j]).buffer)
                else
                    push!(ptrs, Reactant.XLA.synced_buffer(v.data[j]).buffer)
                end
            end
        end
        n = length(ptrs)
        return ntuple(i -> ptrs[i], n), ntuple(Returns(UInt8(0)), n)
    end
end

""")

    # Main function
    W("")
    W("# ── Main ──")
    W("")
    W("function main()")
    W("    println(\"=== MLIR → XLA compile → execute ===\")")
    W("    println(\"Runtime:  \$(Reactant.XLA.REACTANT_XLA_RUNTIME)\")")
    W("")
    W("    client = Reactant.XLA.default_backend()")
    W("    devs   = Reactant.devices()")
    W("    println(\"Backend:  \$(Reactant.XLA.platform_name(client))\")")
    W("    println(\"Devices:  \$(length(devs))\")")
    W("    IS_SHARDED && length(devs) < NDEVICES && error(")
    W("        \"Need \$(NDEVICES) devices but only \$(length(devs)) available.\")")
    W("")
    W("    device = IS_SHARDED ? nothing : Reactant.XLA.default_device(client)")
    W("")
    W("    # Smoke test")
    W("    print(\"Smoke test... \")")
    W("    x = Reactant.ConcreteRArray(ones(Float32, 4))")
    W("    y = @jit identity(x)")
    W("    @assert Array(y) ≈ ones(Float32, 4)")
    W("    println(\"OK\")")
    W("")

    # Create mock data
    W("    println(\"\\nCreating mock data...\")")
    if is_sharded
        W("    mesh = create_mesh(devs)")
        W("    mock_inputs = create_inputs(mesh)")
    else
        W("    mock_inputs = create_inputs()")
    end
    W("    println(\"  \$(length(mock_inputs)) inputs for module 1\")")
    W("")

    # Create extra inputs for later modules
    for (k, m) in enumerate(modules)
        k == 1 && continue
        n_extra = length(m.inputs) - length(first_in)
        if n_extra > 0
            if is_sharded
                W("    extra_$k = create_extra_inputs_$k(mesh)")
            else
                W("    extra_$k = create_extra_inputs_$k()")
            end
        else
            W("    extra_$k = []")
        end
    end
    W("")

    # Compile all modules
    for (k, m) in enumerate(modules)
        W("    println(\"Compiling module $k...\")")
        W("    t0 = time()")
        W("    exec_$k = compile_module(client, MLIR_PATH_$k;")
        W("        num_parameters=N_IN_$k, num_outputs=N_OUT_$k, device)")
        W("    println(\"  done (\$(round(time() - t0; digits=1))s)\")")
        W("")
    end

    # Execute module 1
    W("    println(\"\\nExecuting module 1...\")")
    W("    t0 = time()")
    W("    results_1 = xla_execute(exec_1, mock_inputs, N_OUT_1; device)")
    W("    sync_results(results_1)")
    W("    println(\"  \$(N_OUT_1) outputs (\$(round(time() - t0; digits=1))s)\")")
    W("")

    # Execute subsequent modules with marshaling
    for k in 2:length(modules)
        prev_k = k - 1
        W("    println(\"\\nMarshaling → module $k...\")")
        W("    next_inputs_$k = marshal_next_inputs(")
        W("        mock_inputs, results_$prev_k, ALIAS_MAP_$k, CONST_INDICES, extra_$k;")
        W("        n_in=N_IN_$k)")
        W("    bufs_$k, donated_$k = marshal_bufs(next_inputs_$k, mock_inputs; is_raw_result=true)")
        W("    println(\"  \$(length(bufs_$k)) buffer pointers\")")
        W("")
        W("    println(\"Executing module $k...\")")
        W("    t0 = time()")
        W("    GC.@preserve mock_inputs results_$prev_k extra_$k begin")
        if is_sharded
            W("        if IS_IFRT")
            W("            results_$k = Reactant.XLA.execute(")
            W("                exec_$k, bufs_$k, donated_$k,")
            W("                Val(N_OUT_$k), Val(NDEVICES))")
            W("        else")
            W("            results_$k = Reactant.XLA.execute(")
            W("                exec_$k, bufs_$k, donated_$k,")
            W("                Val(N_OUT_$k), Val(NDEVICES))")
            W("        end")
        else
            W("        results_$k = Reactant.XLA.execute_sharded(")
            W("            exec_$k, device, bufs_$k, donated_$k, Val(N_OUT_$k))")
        end
        W("    end")
        W("    sync_results(results_$k)")
        W("    println(\"  \$(N_OUT_$k) outputs (\$(round(time() - t0; digits=1))s)\")")
        W("")
    end

    last_k = length(modules)
    W("    println(\"\\n=== SUCCESS ===\")")
    W("    return results_$last_k")
    W("end")
    W("")
    W("main()")

    return String(take!(io))
end

end # module MLIRRunner
