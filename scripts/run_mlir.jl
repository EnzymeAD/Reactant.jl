#!/usr/bin/env julia
"""
    run_mlir.jl — Generator: parse pre_xla MLIR files → emit standalone execute script.

Usage:
    julia run_mlir.jl [first_time_step.mlir] [loop.mlir] [output_script.jl]

Reads two GB-25 pre_xla_compile MLIR files, parses their func.func @main
signatures, and writes a self-contained Julia script that:
  1. Creates mock data with the correct shapes and shardings
  2. Loads the MLIR modules
  3. XLA-compiles both
  4. Runs first_time_step → marshals outputs → runs loop

The generated script has no parsing logic — all shapes, types, and shardings
are hardcoded from the analysis done here.
"""

# ──────────────────────────────────────────────────────────────
# MLIR Signature Parsing (no Reactant dependency needed)
# ──────────────────────────────────────────────────────────────

struct TensorSig
    eltype::String                # Julia type name: "Float32", "Int64", ...
    mlir_shape::Vector{Int}       # MLIR dimension order (row-major)
    mlir_sharding::Vector{Symbol} # per MLIR dim: :_none = replicated, :x/:y = axis name
end

const MLIR_ELTYPES = Dict{String,String}(
    "f16" => "Float16", "bf16" => "Float16", "f32" => "Float32", "f64" => "Float64",
    "i1" => "Bool", "i8" => "Int8", "i16" => "Int16", "i32" => "Int32", "i64" => "Int64",
)

function parse_tensor_type(s::AbstractString)
    m = match(r"tensor<((?:\d+x)*)(\w+)>", s)
    m === nothing && error("Cannot parse tensor type: $s")
    dims_part, elem_part = m.captures
    T = get(MLIR_ELTYPES, elem_part, nothing)
    T === nothing && error("Unknown MLIR element type: $elem_part")
    shape = isempty(dims_part) ? Int[] : parse.(Int, split(rstrip(dims_part, 'x'), 'x'))
    return T, shape
end

function parse_sharding_spec(s::AbstractString)
    dims = Symbol[]
    for m in eachmatch(r"\{([^}]*)\}", s)
        content = strip(m.captures[1])
        if isempty(content)
            push!(dims, :_none)
        else
            ax = match(r"\"(\w+)\"", content)
            ax === nothing && error("Cannot parse axis name in sharding: $content")
            push!(dims, Symbol(ax.captures[1]))
        end
    end
    return dims
end

function parse_typed_section(s::AbstractString)
    types = [m.match for m in eachmatch(r"tensor<[^>]+>", s)]
    shards = [m.captures[1] for m in eachmatch(
        r"sdy\.sharding\s*=\s*#sdy\.sharding<@mesh,\s*(\[[^\]]*\])>", s,
    )]
    length(types) == length(shards) || error(
        "Mismatch: $(length(types)) tensor types vs $(length(shards)) sharding specs",
    )
    return [TensorSig(parse_tensor_type(t)..., parse_sharding_spec(sh))
            for (t, sh) in zip(types, shards)]
end

function find_close_paren(s::AbstractString, pos::Int)
    depth = 1
    while depth > 0 && pos <= lastindex(s)
        c = s[pos]
        c == '(' && (depth += 1)
        c == ')' && (depth -= 1)
        depth > 0 && (pos = nextind(s, pos))
    end
    return pos
end

function parse_main_signature(mlir::AbstractString)
    idx = findfirst("func.func @main(", mlir)
    idx === nothing && error("Cannot find func.func @main")
    args_start = last(idx) + 1

    args_close = find_close_paren(mlir, args_start)
    args_str = SubString(mlir, args_start, prevind(mlir, args_close))

    rest = SubString(mlir, args_close)
    rets_str = ""
    m = match(r"^\)\s*->\s*\(", rest)
    if m !== nothing
        rets_start = args_close + length(m.match)
        rets_close = find_close_paren(mlir, rets_start)
        rets_str = SubString(mlir, rets_start, prevind(mlir, rets_close))
    end

    return parse_typed_section(args_str), parse_typed_section(rets_str)
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
    if all(p -> p == "nothing", parts)
        return "Sharding.Replicated(mesh)"
    end
    return "Sharding.NamedSharding(mesh, ($(join(parts, ", ")),))"
end

function codegen_create_input(i::Int, sig::TensorSig; is_sharded::Bool)
    T = sig.eltype
    if isempty(sig.mlir_shape)
        # Scalar
        default_val = T in ("Float32", "Float64", "Float16") ? "$T(60)" : "$T(0)"
        if is_sharded
            sharding = julia_sharding_expr(sig)
            return "    ConcreteRNumber{$T}($default_val; sharding=$sharding)"
        else
            return "    ConcreteRNumber{$T}($default_val)"
        end
    else
        shape = julia_shape_str(sig)
        if is_sharded
            sharding = julia_sharding_expr(sig)
            return "    ConcreteRArray(zeros($T, $shape...); sharding=$sharding)"
        else
            return "    ConcreteRArray(zeros($T, $shape...))"
        end
    end
end

function parse_mesh_spec(mlir::AbstractString)
    # Extract mesh shape from: sdy.mesh @mesh = <["x"=2, "y"=2]>
    m = match(r"sdy\.mesh\s+@mesh\s*=\s*<\[([^\]]+)\]>", mlir)
    m === nothing && return ((:x, :y), (2, 2))  # fallback
    axes = Symbol[]
    sizes = Int[]
    for am in eachmatch(r"\"(\w+)\"\s*=\s*(\d+)", m.captures[1])
        push!(axes, Symbol(am.captures[1]))
        push!(sizes, parse(Int, am.captures[2]))
    end
    return tuple(axes...), tuple(sizes...)
end

function count_grid_constants(mlir::AbstractString)
    # Count args that don't have tf.aliasing_output — those are grid constants.
    # Parse the args section and count %argN entries without aliasing.
    idx = findfirst("func.func @main(", mlir)
    idx === nothing && return 0
    args_start = last(idx) + 1
    args_close = find_close_paren(mlir, args_start)
    args_str = SubString(mlir, args_start, prevind(mlir, args_close))

    # Split by %arg to get per-argument blocks
    n = 0
    for m in eachmatch(r"%arg(\d+):[^%]*", args_str)
        if !contains(m.match, "tf.aliasing_output")
            n += 1
        else
            break  # first aliased arg → stop counting
        end
    end
    return n
end

function parse_module_attrs(mlir::AbstractString)
    num_partitions = 1
    num_replicas = 1
    m = match(r"mhlo\.num_partitions\s*=\s*(\d+)", mlir)
    m !== nothing && (num_partitions = parse(Int, m.captures[1]))
    m = match(r"mhlo\.num_replicas\s*=\s*(\d+)", mlir)
    m !== nothing && (num_replicas = parse(Int, m.captures[1]))
    return num_partitions, num_replicas
end

# ──────────────────────────────────────────────────────────────
# Script generation
# ──────────────────────────────────────────────────────────────

function generate_script(;
    first_path, loop_path,
    first_in, first_out, loop_in, loop_out,
    mesh_axes, mesh_sizes, num_partitions, num_replicas,
    n_grid_const,
    output_dir,
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

    # Build create_mock_inputs body
    input_lines = String[]
    for (i, sig) in enumerate(first_in)
        push!(input_lines, codegen_create_input(i, sig; is_sharded) * ",  # arg$(i-1)")
    end

    # Build summary comments
    summary_lines = String[]
    for (i, sig) in enumerate(first_in)
        jshape = julia_shape_str(sig)
        push!(summary_lines, "#   arg$(i-1): $(sig.eltype) $jshape")
    end

    # Use IOBuffer to avoid string interpolation escaping issues
    io = IOBuffer()
    W(s) = println(io, s)

    W("#!/usr/bin/env julia")
    W("#")
    W("# Generated by run_mlir.jl — do not edit by hand.")
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

    # The rest is static code — write it verbatim
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
# Main: parse MLIR, emit script
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

    println("=== run_mlir.jl — MLIR → script generator ===")
    println("First time step: $first_path")
    println("Loop:            $loop_path")
    println("Output script:   $out_path")

    # Read & parse
    first_mlir = read(first_path, String)
    loop_mlir  = read(loop_path, String)

    first_in, first_out = parse_main_signature(first_mlir)
    loop_in,  loop_out  = parse_main_signature(loop_mlir)

    println("\nParsed signatures:")
    println("  first_time_step: $(length(first_in)) inputs → $(length(first_out)) outputs")
    println("  loop:            $(length(loop_in)) inputs → $(length(loop_out)) outputs")

    for (i, sig) in enumerate(first_in)
        jshape = julia_shape_str(sig)
        shard  = julia_sharding_expr(sig)
        println("    arg$(i-1): $(sig.eltype) $jshape  $shard")
    end

    # Extract mesh / module attributes from the first MLIR
    mesh_axes, mesh_sizes = parse_mesh_spec(first_mlir)
    num_partitions, num_replicas = parse_module_attrs(first_mlir)
    ndevices = prod(mesh_sizes)

    println("\nModule attributes:")
    println("  mesh:           $(collect(zip(mesh_axes, mesh_sizes)))")
    println("  num_partitions: $num_partitions")
    println("  num_replicas:   $num_replicas")
    println("  ndevices:       $ndevices")

    # Detect n_grid_const: count args before the first tf.aliasing_output
    n_grid_const = count_grid_constants(first_mlir)
    println("  n_grid_const:   $n_grid_const (args 0-$(n_grid_const-1))")

    # Generate
    output_dir = dirname(abspath(out_path))
    script = generate_script(;
        first_path=abspath(first_path),
        loop_path=abspath(loop_path),
        first_in, first_out, loop_in, loop_out,
        mesh_axes, mesh_sizes, num_partitions, num_replicas,
        n_grid_const,
        output_dir,
    )

    write(out_path, script)
    println("\nWrote $(out_path) ($(count(==('\n'), script) + 1) lines)")
    println("Run with:  julia --project=Reactant.jl $(basename(out_path))")
end

main()
