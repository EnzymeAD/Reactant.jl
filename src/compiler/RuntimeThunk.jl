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

# Equal plans are interned to the same object, so that equal plans give equal thunk bodies
# (`Expr` compares embedded objects by identity) and compiling the same function again
# reuses the thunk (`__thunk_rev_body_cache` is keyed by the body).
const THUNK_PLANS = Dict{Any,Any}()

intern_plan(plan) = get!(THUNK_PLANS, plan, plan)

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

function resolve_argument_path(@nospecialize(args), path)
    obj = args[path[1]]
    for p in Base.tail(path)
        obj = traced_getfield(obj, p)
    end
    return obj
end

# Read the input buffers and mark donated ones, like the generated flatten code.
@noinline function runtime_thunk_inputs(
    @nospecialize(args), plan::ThunkPlan, donated_args_mask, donated_buffers
)
    n = length(plan.input_paths)
    pointers = Vector{Ptr{Cvoid}}(undef, n)
    buffers = Vector{XLA.PJRT.Buffer}(undef, n)
    for (i, path) in enumerate(plan.input_paths)
        carg = resolve_argument_path(args, path)
        buffer = only(XLA.synced_buffer(carg.data))
        buffers[i] = buffer
        pointers[i] = buffer.buffer
        # Important to mark donated after we have extracted the data
        donate_argument!(donated_args_mask, carg, i, donated_buffers, (:args, path...))
    end
    return pointers, buffers
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
    donated = UInt8[UInt8(d) for d in donated_args_mask]
    outputs = Vector{Ptr{Cvoid}}(undef, nresults)
    future_results = Vector{Ptr{Cvoid}}(undef, nresults)
    futures = Ref{UInt8}(0)
    GC.@preserve exec device inputs input_buffers donated outputs future_results futures begin
        MLIR.API.XLAExecuteSharded(
            exec.exec,
            length(inputs),
            pointer(inputs),
            device.device,
            pointer(donated),
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

function runtime_thunk_read(@nospecialize(args), read::AliasRead)
    data = resolve_argument_path(args, read.path).data
    return read.copy ? map(copy, data) : data
end

# The reads that must happen before the results are written back.
@noinline function runtime_thunk_pre_reads(@nospecialize(args), plan::ThunkPlan)
    values = Vector{Any}(undef, length(plan.alias_reads))
    for (i, read) in enumerate(plan.alias_reads)
        read.pre_execution && (values[i] = runtime_thunk_read(args, read))
    end
    return values
end

# Write the results back into the arguments, then propagate aliased arguments, like the
# generated unflatten code.
@noinline function runtime_thunk_write_back!(
    @nospecialize(args), plan::ThunkPlan, results, values::Vector{Any}
)
    runtime = Val(:PJRT)
    cache_dict = IdDict{
        Union{TracedRArray,TracedRNumber,Reactant.TracedEnum},
        Union{ConcretePJRTArray,ConcretePJRTNumber,Reactant.ConcreteEnum},
    }()
    for write in plan.result_writes
        result = results[write.result]
        argidx = write.path[1]
        path = Base.tail(write.path)
        if length(path) > 1
            parent = resolve_argument_path(args, (argidx, path[1:(end - 2)]...))
            traced_setfield_buffer_at_parent!(
                runtime, cache_dict, result, parent, path[end - 1], path[end], path
            )
        elseif length(path) > 0
            traced_setfield_buffer!(
                runtime, cache_dict, result, args[argidx], path[end], path
            )
        else
            traced_setfield!(args[argidx], :data, result, path)
        end
    end
    for (i, read) in enumerate(plan.alias_reads)
        read.pre_execution || (values[i] = runtime_thunk_read(args, read))
    end
    for write in plan.alias_writes
        target = resolve_argument_path(args, write.path)
        traced_setfield!(target, :data, values[write.read], Base.tail(write.path))
    end
    return nothing
end
