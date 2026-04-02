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
    eltype::Type
    mlir_shape::Vector{Int}
    mlir_sharding::Vector{Symbol}  # per Julia dim: :_none = replicated, :x/:y = axis
end

"""Result of [`analyze_module`](@ref)."""
struct ModuleInfo
    inputs::Vector{TensorSig}
    outputs::Vector{TensorSig}
    alias_map::Dict{Int,Int}       # output_idx (1-based) → input_idx (1-based)
    num_partitions::Int
    num_replicas::Int
    mesh_axes::Vector{Symbol}
    mesh_sizes::Vector{Int}
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
        T = IR.julia_type(IR.eltype(mlir_type))
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
        T = IR.julia_type(IR.eltype(t))
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

    return ModuleInfo(inputs, outputs, alias_map, num_partitions, num_replicas,
                      mesh_axes, mesh_sizes)
end

# ──────────────────────────────────────────────────────────────
# Runtime helpers — called by generated scripts via
#   Reactant.Serialization.MLIRRunner.<func>(...)
# ──────────────────────────────────────────────────────────────

function get_buf_ptr(x, is_ifrt::Bool)
    if is_ifrt
        return Reactant.XLA.synced_buffer(x.data).buffer
    else
        return Reactant.XLA.synced_buffer(only(x.data)).buffer
    end
end

function xla_execute(exec, inputs, n_outs::Int;
                     device=nothing, is_sharded::Bool, is_ifrt::Bool, ndevices::Int)
    n = length(inputs)
    if !is_sharded
        bufs = ntuple(i -> get_buf_ptr(inputs[i], is_ifrt), n)
        donated = ntuple(Returns(UInt8(0)), n)
        GC.@preserve inputs begin
            return Reactant.XLA.execute_sharded(exec, device, bufs, donated, Val(n_outs))
        end
    else
        if is_ifrt
            bufs = ntuple(i -> get_buf_ptr(inputs[i], is_ifrt), n)
            donated = ntuple(Returns(UInt8(0)), n)
        else
            ptrs = Ptr{Cvoid}[]
            for j in 1:ndevices
                for arg in inputs
                    push!(ptrs, Reactant.XLA.synced_buffer(arg.data[j]).buffer)
                end
            end
            np = length(ptrs)
            bufs = ntuple(i -> ptrs[i], np)
            donated = ntuple(Returns(UInt8(0)), np)
        end
        GC.@preserve inputs begin
            return Reactant.XLA.execute(exec, bufs, donated, Val(n_outs), Val(ndevices))
        end
    end
end

function sync_results(results)
    for r in results
        Reactant.XLA.synced_buffer(r isa Tuple ? r[1] : r)
    end
end

function compile_module(client, mlir_path;
                        num_parameters, num_outputs, device=nothing,
                        is_sharded::Bool, num_replicas::Int, num_partitions::Int,
                        ndevices::Int)
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)
    try
        mod = parse(MLIR.IR.Module, read(mlir_path, String))

        device_id = is_sharded ? Int64(-1) : Int64(Reactant.XLA.device_ordinal(device))
        compile_opts = Reactant.XLA.make_compile_options(;
            device_id,
            num_replicas=Int64(num_replicas),
            num_partitions=Int64(num_partitions),
            mesh_ids=is_sharded ? collect(Int64, 0:(ndevices - 1)) : nothing,
            xla_executable_build_options=(;
                use_shardy_partitioner=is_sharded,
                use_spmd_partitioning=is_sharded,
            ),
        )
        return Reactant.XLA.compile(client, mod;
            compile_options=compile_opts,
            num_parameters=Int64(num_parameters),
            num_outputs=Int64(num_outputs),
            is_sharded,
            num_replicas=Int64(num_replicas),
            num_partitions=Int64(num_partitions),
        )
    finally
        MLIR.IR.deactivate(ctx)
    end
end

"""
    marshal_next_inputs(mock_inputs, prev_results, alias_map, const_indices, extra_inputs; n_in)

Build the input vector for the next module by:
1. Copying constants from `mock_inputs` at `const_indices`
2. Wiring `prev_results[out_idx]` → slot `in_idx` per `alias_map`
3. Appending `extra_inputs` beyond the first module's arg count
"""
function marshal_next_inputs(
    mock_inputs, prev_results, alias_map, const_indices, extra_inputs;
    n_in,
)
    next = Vector{Any}(undef, n_in)
    n_base = length(mock_inputs)

    for idx in const_indices
        if idx <= n_base
            next[idx] = mock_inputs[idx]
        end
    end

    for (out_idx, in_idx) in alias_map
        if in_idx <= n_in
            next[in_idx] = prev_results[out_idx]
        end
    end

    for (j, extra) in enumerate(extra_inputs)
        next[n_base + j] = extra
    end

    # Verify all slots were filled
    for i in 1:n_in
        isassigned(next, i) || error(
            "marshal_next_inputs: slot $i was not filled. " *
            "Check alias_map, const_indices, and extra_inputs coverage.")
    end

    return next
end

const _ConcreteRData = Union{Reactant.ConcreteRArray, Reactant.ConcreteRNumber}

function marshal_bufs(next_inputs, is_ifrt::Bool, is_sharded::Bool, ndevices::Int)
    if !is_sharded || is_ifrt
        n = length(next_inputs)
        lp = Vector{Ptr{Cvoid}}(undef, n)
        for i in 1:n
            v = next_inputs[i]
            if v isa _ConcreteRData
                lp[i] = get_buf_ptr(v, is_ifrt)
            else
                buf = v isa Tuple ? v[1] : v
                lp[i] = Reactant.XLA.synced_buffer(buf).buffer
            end
        end
        return ntuple(i -> lp[i], n), ntuple(Returns(UInt8(0)), n)
    else
        ptrs = Ptr{Cvoid}[]
        for j in 1:ndevices
            for v in next_inputs
                if v isa _ConcreteRData
                    push!(ptrs, Reactant.XLA.synced_buffer(v.data[j]).buffer)
                else
                    push!(ptrs, Reactant.XLA.synced_buffer(v[j]).buffer)
                end
            end
        end
        n = length(ptrs)
        return ntuple(i -> ptrs[i], n), ntuple(Returns(UInt8(0)), n)
    end
end

function xla_execute_raw(exec, bufs, donated, n_outs::Int;
                         device=nothing, is_sharded::Bool, ndevices::Int)
    if !is_sharded
        return Reactant.XLA.execute_sharded(exec, device, bufs, donated, Val(n_outs))
    else
        return Reactant.XLA.execute(exec, bufs, donated, Val(n_outs), Val(ndevices))
    end
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
        default_val = T <: AbstractFloat ? "$T(60)" : "$T(0)"
        return "    ConcreteRNumber{$T}($default_val$shard_kw)"
    else
        shape = julia_shape_str(sig)
        return "    ConcreteRArray(randn($T, $shape...)$shard_kw)"
    end
end

"""
    codegen_input_constructor(name, sigs, is_sharded; mesh_arg=is_sharded)

Generate a `function <name>(...) ... end` block that returns a vector of mock inputs.
"""
function codegen_input_constructor(name::String, sigs::Vector{TensorSig},
                                   is_sharded::Bool; start_idx::Int=1)
    lines = String[]
    if is_sharded
        push!(lines, "function $name(mesh)")
        push!(lines, "    Sharding = Reactant.Sharding")
    else
        push!(lines, "function $name()")
    end
    push!(lines, "    ConcreteRArray  = Reactant.ConcreteRArray")
    push!(lines, "    ConcreteRNumber = Reactant.ConcreteRNumber")
    push!(lines, "    return [")
    for (j, sig) in enumerate(sigs)
        idx = start_idx + j - 1
        push!(lines, codegen_create_input(idx, sig; is_sharded) * ",  # arg$(idx-1)")
    end
    push!(lines, "    ]")
    push!(lines, "end")
    return lines
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

    modules = MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        IR.activate(ctx)
        try
            [(; path=abspath(p), info=analyze_module(read(p, String))) for p in mlir_files]
        finally
            IR.deactivate(ctx)
        end
    end

    first_info = modules[1].info
    ndevices = prod(first_info.mesh_sizes; init=1)
    is_sharded = first_info.num_partitions > 1

    script = _generate_script(
        modules, output_dir;
        first_info.mesh_axes, first_info.mesh_sizes,
        first_info.num_partitions, first_info.num_replicas,
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

    first_info = modules[1].info
    first_in = first_info.inputs
    n_modules = length(modules)

    mesh_names_str = "($(join([":" * string(a) for a in mesh_axes], ", ")),)"
    mesh_shape_str = join(mesh_sizes, ", ")

    # Precompute extra-input info for modules 2..N
    extra_info = Dict{Int,Vector{TensorSig}}()
    for k in 2:n_modules
        m_in = modules[k].info.inputs
        n_extra = length(m_in) - length(first_in)
        if n_extra > 0
            extra_info[k] = m_in[length(first_in)+1:end]
        end
    end

    # ── Header ──
    W("#!/usr/bin/env julia")
    W("#")
    W("# Generated by Reactant.Serialization.generate_mlir_runner — do not edit by hand.")
    W("#")
    W("# Standalone XLA compile + execute for $n_modules pre-XLA MLIR module(s).")
    W("# Requires $ndevices device(s) (num_partitions=$num_partitions, num_replicas=$num_replicas).")
    W("#")
    W("# Pass --cpu to run on $ndevices virtual CPU devices instead of GPUs:")
    W("#   julia --project=Reactant.jl <script>.jl --cpu")
    W("#")
    for (k, m) in enumerate(modules)
        W("# Module $k: $(length(m.info.inputs)) inputs → $(length(m.info.outputs)) outputs")
    end
    W("#")
    W("")

    # ── CPU mode ──
    W("const USE_CPU = \"--cpu\" in ARGS")
    W("if USE_CPU")
    W("    ENV[\"CUDA_VISIBLE_DEVICES\"] = \"\"")
    W("    ENV[\"XLA_FLAGS\"] = get(ENV, \"XLA_FLAGS\", \"\") *")
    W("        \" --xla_force_host_platform_device_count=$ndevices\"")
    W("end")
    W("")
    W("using Reactant")
    W("const RT = Reactant.Serialization.MLIRRunner")
    W("")
    W("if USE_CPU")
    W("    Reactant.set_default_backend(\"cpu\")")
    W("end")
    W("")

    # ── Constants ──
    W("const IS_IFRT    = Reactant.XLA.REACTANT_XLA_RUNTIME == \"IFRT\"")
    W("const IS_SHARDED = $(is_sharded)")
    W("const NDEVICES   = $ndevices")
    W("const NUM_PARTITIONS = $num_partitions")
    W("const NUM_REPLICAS   = $num_replicas")
    W("")

    for (k, m) in enumerate(modules)
        rel = relpath(m.path, output_dir)
        W("const MLIR_PATH_$k = joinpath(@__DIR__, $(repr(rel)))")
    end
    W("")

    for (k, m) in enumerate(modules)
        W("const N_IN_$k  = $(length(m.info.inputs))")
        W("const N_OUT_$k = $(length(m.info.outputs))")
    end
    W("")

    for (k, m) in enumerate(modules)
        if !isempty(m.info.alias_map)
            pairs = sort(collect(m.info.alias_map))
            W("const ALIAS_MAP_$k = Dict{Int,Int}($(join(["$o => $i" for (o,i) in pairs], ", ")))")
        else
            W("const ALIAS_MAP_$k = Dict{Int,Int}()")
        end
    end
    W("")

    aliased_inputs_1 = Set(values(first_info.alias_map))
    const_indices = [i for i in 1:length(first_in) if i ∉ aliased_inputs_1]
    W("const CONST_INDICES = $(const_indices)")
    W("")

    # ── Input constructors ──
    if is_sharded
        W("function create_mesh(devs)")
        W("    Sharding = Reactant.Sharding")
        W("    return Sharding.Mesh(reshape(devs[1:NDEVICES], $mesh_shape_str), $mesh_names_str)")
        W("end")
        W("")
    end

    for line in codegen_input_constructor("create_inputs", first_in, is_sharded)
        W(line)
    end
    W("")

    for (k, sigs) in sort(collect(extra_info))
        for line in codegen_input_constructor(
                "create_extra_inputs_$k", sigs, is_sharded;
                start_idx=length(first_in) + 1)
            W(line)
        end
        W("")
    end

    # ── Main ──
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

    W("    println(\"\\nCreating mock data...\")")
    if is_sharded
        W("    mesh = create_mesh(devs)")
        W("    mock_inputs = create_inputs(mesh)")
    else
        W("    mock_inputs = create_inputs()")
    end
    W("    println(\"  \$(length(mock_inputs)) inputs for module 1\")")
    W("")

    for k in 2:n_modules
        if haskey(extra_info, k)
            if is_sharded
                W("    extra_$k = create_extra_inputs_$k(mesh)")
            else
                W("    extra_$k = create_extra_inputs_$k()")
            end
        else
            W("    extra_$k = []")
        end
    end
    if n_modules > 1
        W("")
    end

    for k in 1:n_modules
        W("    println(\"Compiling module $k...\")")
        W("    t0 = time()")
        W("    exec_$k = RT.compile_module(client, MLIR_PATH_$k;")
        W("        num_parameters=N_IN_$k, num_outputs=N_OUT_$k, device,")
        W("        is_sharded=IS_SHARDED, num_replicas=NUM_REPLICAS,")
        W("        num_partitions=NUM_PARTITIONS, ndevices=NDEVICES)")
        W("    println(\"  done (\$(round(time() - t0; digits=1))s)\")")
        W("")
    end

    W("    println(\"\\nExecuting module 1...\")")
    W("    t0 = time()")
    W("    results_1 = RT.xla_execute(exec_1, mock_inputs, N_OUT_1;")
    W("        device, is_sharded=IS_SHARDED, is_ifrt=IS_IFRT, ndevices=NDEVICES)")
    W("    RT.sync_results(results_1)")
    W("    println(\"  \$(N_OUT_1) outputs (\$(round(time() - t0; digits=1))s)\")")
    W("")

    for k in 2:n_modules
        prev_k = k - 1
        W("    println(\"\\nMarshaling → module $k...\")")
        W("    next_$k = RT.marshal_next_inputs(")
        W("        mock_inputs, results_$prev_k, ALIAS_MAP_$k, CONST_INDICES, extra_$k;")
        W("        n_in=N_IN_$k)")
        W("    bufs_$k, donated_$k = RT.marshal_bufs(next_$k, IS_IFRT, IS_SHARDED, NDEVICES)")
        W("    println(\"  \$(length(bufs_$k)) buffer pointers\")")
        W("")
        W("    println(\"Executing module $k...\")")
        W("    t0 = time()")
        W("    GC.@preserve mock_inputs results_$prev_k extra_$k begin")
        W("        results_$k = RT.xla_execute_raw(exec_$k, bufs_$k, donated_$k, N_OUT_$k;")
        W("            device, is_sharded=IS_SHARDED, ndevices=NDEVICES)")
        W("    end")
        W("    RT.sync_results(results_$k)")
        W("    println(\"  \$(N_OUT_$k) outputs (\$(round(time() - t0; digits=1))s)\")")
        W("")
    end

    last_k = n_modules
    W("    println(\"\\n=== SUCCESS ===\")")
    W("    return results_$last_k")
    W("end")
    W("")
    W("main()")

    return String(take!(io))
end

end # module MLIRRunner
