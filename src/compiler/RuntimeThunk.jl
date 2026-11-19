# An alternative to generating the whole thunk body as one straight-line function.
#
# The generated body reads every input buffer, calls the executable and writes every result
# back through `traced_getfield`/`traced_setfield!` chains, all inlined into one function
# that Julia compiles on the first call of the thunk. Its size grows with the number of
# buffers in the arguments, and native code generation for it is superlinear in that size.
#
# With `RUNTIME_THUNK[] = true`, this work is instead described by a `ThunkPlan` (the paths
# to read and write) and carried out by the functions below, which take their arguments
# without specializing on them and so are compiled once for all thunks. Only the
# construction of the returned value, whose size is that of the result, is still generated.
# This currently covers the unsharded PJRT case; other cases use the generated body.

const RUNTIME_THUNK = Ref(false)

# `REACTANT_RUNTIME_THUNK=1` also enables it, e.g. to run the test suite in this mode
runtime_thunk_enabled() = RUNTIME_THUNK[] || get(ENV, "REACTANT_RUNTIME_THUNK", "") == "1"

# Write the `result`-th result of the executable back into the argument at `path`
# (`path[1]` indexes `args`, the rest are fields).
struct ResultWrite
    result::Int
    path::Tuple
end

# Read the buffer of the argument at `path`, copying it if `copy`, either before the
# result write-back (`pre_execution`) or after it.
struct AliasRead
    path::Tuple
    pre_execution::Bool
    copy::Bool
end

# Write the value read by the `read`-th `AliasRead` into the argument at `path`.
struct AliasWrite
    read::Int
    path::Tuple
end

struct ThunkPlan
    input_paths::Vector{Tuple}
    nresults::Int
    result_writes::Vector{ResultWrite}
    alias_reads::Vector{AliasRead}
    alias_writes::Vector{AliasWrite}
end

function ThunkPlan(input_paths, nresults)
    return ThunkPlan(input_paths, nresults, ResultWrite[], AliasRead[], AliasWrite[])
end

# Following a path from `args` again for every buffer is costly: each step through an
# immutable container (a tuple or `NamedTuple` of many fields, say) boxes a copy of it, so
# the work grows with the number of buffers times the size of the containers they sit in.
# A `ThunkProgram` is a `ThunkPlan` whose paths are merged into a tree of `PathNode`s, so
# that each object on the way to a buffer is fetched once per call of the thunk.

# The object at `field` of the `parent`-th node, or `args[field]` if `parent == 0`.
struct PathNode
    parent::Int
    field::Any
end

# An object reached from the `node`-th node through `fields`.
struct NodePath
    node::Int
    fields::Tuple
end

struct ProgramResultWrite
    result::Int
    target::NodePath # of the argument's parent container, see `runtime_thunk_write_back!`
    path::Tuple
end

struct ProgramAliasRead
    source::NodePath
    pre_execution::Bool
    copy::Bool
end

struct ProgramAliasWrite
    read::Int
    target::NodePath
    path::Tuple
end

struct ThunkProgram
    nodes::Vector{PathNode} # a node comes after its parent
    inputs::Vector{Int} # the node of each input
    input_paths::Vector{Tuple} # from `:args`, to report an input
    nresults::Int
    result_writes::Vector{ProgramResultWrite}
    alias_reads::Vector{ProgramAliasRead}
    alias_writes::Vector{ProgramAliasWrite}
end

function path_node!(nodes, node_of_path, path)
    return get!(node_of_path, path) do
        parent = length(path) == 1 ? 0 : path_node!(nodes, node_of_path, path[1:(end - 1)])
        push!(nodes, PathNode(parent, path[end]))
        length(nodes)
    end
end

# The nodes are resolved before the executable is called and used after the write-back
# too, so a node must not be an object that the write-back replaces in its container. The
# write-back replaces traced values at the end of a path and traced enums one step before
# it, so after the execution the last two steps of a path are followed again.
function stable_node_path(nodes, node_of_path, path)
    n = max(1, length(path) - 2)
    return NodePath(path_node!(nodes, node_of_path, path[1:n]), path[(n + 1):end])
end

function ThunkProgram(plan::ThunkPlan)
    nodes = PathNode[]
    node_of_path = Dict{Tuple,Int}()
    inputs = Int[path_node!(nodes, node_of_path, path) for path in plan.input_paths]
    result_writes = map(plan.result_writes) do write
        # the last two steps are taken by `traced_setfield_buffer_at_parent!`
        n = max(1, length(write.path) - 2)
        target = NodePath(path_node!(nodes, node_of_path, write.path[1:n]), ())
        ProgramResultWrite(write.result, target, Base.tail(write.path))
    end
    alias_reads = map(plan.alias_reads) do read
        source = stable_node_path(nodes, node_of_path, read.path)
        ProgramAliasRead(source, read.pre_execution, read.copy)
    end
    alias_writes = map(plan.alias_writes) do write
        target = stable_node_path(nodes, node_of_path, write.path)
        ProgramAliasWrite(write.read, target, Base.tail(write.path))
    end
    return ThunkProgram(
        nodes,
        inputs,
        Tuple[(:args, path...) for path in plan.input_paths],
        plan.nresults,
        result_writes,
        alias_reads,
        alias_writes,
    )
end

# Equal plans are interned to the same program, so that equal plans give equal thunk bodies
# (`Expr` compares embedded objects by identity) and compiling the same function again
# reuses the thunk (`__thunk_rev_body_cache` is keyed by the body).
const THUNK_PROGRAMS = Dict{ThunkPlan,ThunkProgram}()

intern_plan(plan) = get!(() -> ThunkProgram(plan), THUNK_PROGRAMS, plan)

for T in (:ResultWrite, :AliasRead, :AliasWrite, :ThunkPlan)
    @eval begin
        function Base.:(==)(a::$T, b::$T)
            return all(f -> getfield(a, f) == getfield(b, f), fieldnames($T))
        end
        function Base.hash(a::$T, h::UInt)
            return foldl(
                (h, f) -> hash(getfield(a, f), h), fieldnames($T); init=hash($T, h)
            )
        end
    end
end

function can_use_runtime_thunk(is_sharded, client, resharded_inputs)
    return runtime_thunk_enabled() &&
           !is_sharded &&
           XLA.runtime(client) isa Val{:PJRT} &&
           isempty(resharded_inputs) &&
           !DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[]
end

# The symbols that appear in `code`, to bind only the result buffers it uses.
function referenced_symbols(code)
    symbols = Set{Symbol}()
    collect_symbols!(symbols, code)
    return symbols
end

collect_symbols!(symbols, x::Symbol) = push!(symbols, x)
collect_symbols!(symbols, x::Expr) = foreach(a -> collect_symbols!(symbols, a), x.args)
collect_symbols!(symbols, x::AbstractVector) = foreach(a -> collect_symbols!(symbols, a), x)
collect_symbols!(symbols, _) = symbols

# The objects of the program's nodes.
function resolve_nodes(@nospecialize(args), program::ThunkProgram)
    nodes = program.nodes
    objects = Vector{Any}(undef, length(nodes))
    for (i, node) in enumerate(nodes)
        objects[i] = if node.parent == 0
            args[node.field]
        else
            traced_getfield(objects[node.parent], node.field)
        end
    end
    return objects
end

function resolve_node_path(objects::Vector{Any}, path::NodePath)
    obj = objects[path.node]
    for p in path.fields
        obj = traced_getfield(obj, p)
    end
    return obj
end

# The buffer to pass to the executable, without waiting for the computation producing it
# (see `XLA.unsynced_buffer`).
function input_buffer(@nospecialize(carg))
    data = carg.data
    data isa Tuple{XLA.PJRT.AsyncBuffer} && return data[1].buffer
    return only(XLA.unsynced_buffer(data))::XLA.PJRT.Buffer
end

# Read the input buffers and mark donated ones, like the generated flatten code.
@noinline function runtime_thunk_inputs(
    @nospecialize(args), program::ThunkProgram, donated_args_mask, donated_buffers
)
    objects = resolve_nodes(args, program)
    inputs = program.inputs
    n = length(inputs)
    pointers = Vector{Ptr{Cvoid}}(undef, n)
    buffers = Vector{XLA.PJRT.Buffer}(undef, n)
    for i in 1:n
        carg = objects[inputs[i]]
        buffer = input_buffer(carg)
        buffers[i] = buffer
        pointers[i] = buffer.buffer
        # Important to mark donated after we have extracted the data
        if donated_args_mask[i]::Bool
            donate_argument!(
                donated_args_mask, carg, i, donated_buffers, program.input_paths[i]
            )
        end
    end
    return pointers, buffers, objects
end

# `XLA.execute_sharded` with a number of inputs and outputs known only at runtime.
@noinline function runtime_thunk_execute(
    exec::XLA.PJRT.LoadedExecutable,
    device::XLA.PJRT.Device,
    inputs::Vector{Ptr{Cvoid}},
    input_buffers::Vector{XLA.PJRT.Buffer},
    donated_args_mask,
    nresults::Int,
)
    donated = if donated_args_mask isa Vector{Bool}
        donated_args_mask
    else
        Bool[d for d in donated_args_mask]
    end
    outputs = Vector{Ptr{Cvoid}}(undef, nresults)
    future_results = Vector{Ptr{Cvoid}}(undef, nresults)
    futures = Ref{UInt8}(0)
    GC.@preserve exec device inputs input_buffers donated outputs future_results futures begin
        MLIR.API.XLAExecuteSharded(
            exec.exec,
            length(inputs),
            pointer(inputs),
            device.device,
            Ptr{UInt8}(pointer(donated)),
            nresults,
            pointer(outputs),
            futures,
            pointer(future_results),
        )
    end
    future = futures[] != 0
    return Tuple{XLA.PJRT.AsyncBuffer}[
        (
            XLA.PJRT.AsyncBuffer(
                XLA.PJRT.Buffer(outputs[i]),
                future ? XLA.PJRT.Future(future_results[i]) : nothing,
            ),
        ) for i in 1:nresults
    ]
end

@noinline function runtime_thunk_sync!(results)
    foreach(XLA.synced_buffer, results)
    return nothing
end

function runtime_thunk_read(objects::Vector{Any}, read::ProgramAliasRead)
    data = resolve_node_path(objects, read.source).data
    return read.copy ? map(copy, data) : data
end

# The reads that must happen before the results are written back.
@noinline function runtime_thunk_pre_reads(program::ThunkProgram, objects::Vector{Any})
    values = Vector{Any}(undef, length(program.alias_reads))
    for (i, read) in enumerate(program.alias_reads)
        read.pre_execution && (values[i] = runtime_thunk_read(objects, read))
    end
    return values
end

# Write the results back into the arguments, then propagate aliased arguments, like the
# generated unflatten code.
@noinline function runtime_thunk_write_back!(
    program::ThunkProgram, objects::Vector{Any}, results, values::Vector{Any}
)
    runtime = Val(:PJRT)
    cache_dict = IdDict{
        Union{TracedRArray,TracedRNumber,Reactant.TracedEnum},
        Union{ConcretePJRTArray,ConcretePJRTNumber,Reactant.ConcreteEnum},
    }()
    for write in program.result_writes
        result = results[write.result]
        target = resolve_node_path(objects, write.target)
        path = write.path
        if length(path) > 1
            traced_setfield_buffer_at_parent!(
                runtime, cache_dict, result, target, path[end - 1], path[end], path
            )
        elseif length(path) > 0
            traced_setfield_buffer!(runtime, cache_dict, result, target, path[end], path)
        else
            traced_setfield!(target, :data, result, path)
        end
    end
    for (i, read) in enumerate(program.alias_reads)
        read.pre_execution || (values[i] = runtime_thunk_read(objects, read))
    end
    for write in program.alias_writes
        target = resolve_node_path(objects, write.target)
        traced_setfield!(target, :data, values[write.read], write.path)
    end
    return nothing
end
