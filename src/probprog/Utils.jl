using ..Reactant:
    MLIR,
    TracedUtils,
    Ops,
    TracedRArray,
    TracedRNumber,
    Compiler,
    OrderedIdDict,
    TracedToTypes,
    TracedType,
    TracedTrack,
    TracedSetPath,
    ConcreteToTraced,
    AbstractConcreteArray,
    XLA,
    Sharding,
    to_number
import ..Reactant: promote_to, make_tracer
import ..Compiler: donate_argument!

"""
    process_probprog_function(f, args, op_name)

Note: by convention `args` must have the RNG state as the first argument.

This function handles the probprog argument convention where:
- **Index 1**: RNG state
- **Index 2**: Function `f` (when wrapped)
- **Index 3+**: Remaining arguments

This wrapper ensures the RNG state is threaded through as the first result,
followed by the actual function results.
"""
function process_probprog_function(f, args, op_name, with_rng=true)
    seen = OrderedIdDict()
    cache_key = []
    make_tracer(seen, (f, args...), cache_key, TracedToTypes)
    cache = Compiler.callcache()

    if haskey(cache, cache_key)
        (; f_name, mlir_result_types, traced_result, mutated_args, linear_results, fnwrapped, argprefix, resprefix, resargprefix) = cache[cache_key]
    else
        f_name = String(gensym(Symbol(f)))
        argprefix::Symbol = gensym(op_name * "arg")
        resprefix::Symbol = gensym(op_name * "result")
        resargprefix::Symbol = gensym(op_name * "resarg")

        wrapper_fn = if !with_rng
            f
        else
            (all_args...) -> begin
                res = f(all_args...)
                (all_args[1], (res isa Tuple ? res : (res,))...)
            end
        end

        temp = TracedUtils.make_mlir_fn(
            wrapper_fn,
            args,
            (),
            f_name,
            false;
            do_transpose=false,
            args_in_result=:result,
            argprefix,
            resprefix,
            resargprefix,
        )

        (; traced_result, ret, mutated_args, linear_results, fnwrapped) = temp
        mlir_result_types = [
            MLIR.IR.type(MLIR.IR.operand(ret, i)) for i in 1:MLIR.IR.noperands(ret)
        ]
        cache[cache_key] = (;
            f_name,
            mlir_result_types,
            traced_result,
            mutated_args,
            linear_results,
            fnwrapped,
            argprefix,
            resprefix,
            resargprefix,
        )
    end

    seen_cache = OrderedIdDict()
    make_tracer(seen_cache, fnwrapped ? (f, args) : args, (), TracedTrack; toscalar=false)
    linear_args = []
    mlir_caller_args = MLIR.IR.Value[]
    for (_, v) in seen_cache
        v isa TracedType || continue
        push!(linear_args, v)
        push!(mlir_caller_args, v.mlir_data)
        v.paths = v.paths[1:(end - 1)]
    end

    return (;
        f_name,
        linear_args,
        mlir_caller_args,
        mlir_result_types,
        traced_result,
        linear_results,
        fnwrapped,
        argprefix,
        resprefix,
        resargprefix,
    )
end

"""
    process_probprog_outputs(op, linear_results, traced_result, f, args, fnwrapped, resprefix, argprefix, offset=0, rng_only=false)

This function handles the probprog argument convention where:
- **Index 1**: RNG state
- **Index 2**: Function `f` (when `fnwrap` is true)
- **Index 3+**: Other arguments

When setting results, the function checks:
1. If result path matches `resprefix`, store in `result`
2. If result path matches `argprefix`, store in `args` (adjust indices for wrapped function)

`offset` varies depending on the ProbProg operation:
- `sample` and `untraced_call` return only function outputs:
  Use `offset=0`: `linear_results[i]` corresponds to `op.result[i]`
- `simulate` and `generate` return trace, weight, then outputs:
  Use `offset=2`: `linear_results[i]` corresponds to `op.result[i+2]`
- `mh` and `regenerate` return trace, accepted/weight, rng_state (no model outputs):
  Use `offset=2, rng_only=true`: only process first result (rng_state)

`rng_only`: When true, only process the first result (RNG state), skipping model outputs
"""
function process_probprog_outputs(
    op,
    linear_results,
    traced_result,
    f,
    args,
    fnwrapped,
    resprefix,
    argprefix,
    offset=0,
    rng_only=false,
)
    seen_results = OrderedIdDict()
    traced_result = make_tracer(
        seen_results, traced_result, (), TracedSetPath; toscalar=false
    )

    num_to_process = rng_only ? 1 : length(linear_results)

    for i in 1:num_to_process
        res = linear_results[i]
        resv = MLIR.IR.result(op, i + offset)

        for path in res.paths
            if length(path) == 0
                continue
            end
            if path[1] == resprefix
                TracedUtils.set!(traced_result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if fnwrapped && idx == 2
                    TracedUtils.set!(f, path[3:end], resv)
                else
                    if fnwrapped && idx > 2
                        idx -= 1
                    end
                    TracedUtils.set!(args[idx], path[3:end], resv)
                end
            end
        end
    end

    return traced_result
end

function promote_to(::Type{TracedRArray{UInt64,0}}, t::Union{ProbProgTrace,Constraint})
    return Ops.fill(reinterpret(UInt64, pointer_from_objref(t)), Int64[])
end

function Base.convert(
    ::Type{T}, x::AbstractConcreteArray
) where {T<:Union{ProbProgTrace,Constraint}}
    while !isready(x)
        yield()
    end
    return unsafe_pointer_to_objref(Ptr{Any}(collect(x)[1]))::T
end

function Base.convert(
    ::Type{T}, x::AbstractConcreteNumber
) where {T<:Union{ProbProgTrace,Constraint}}
    while !isready(x)
        yield()
    end
    return unsafe_pointer_to_objref(Ptr{Any}(to_number(x)))::T
end

function Base.getproperty(t::Union{ProbProgTrace,Constraint}, s::Symbol)
    if s === :data
        return ConcreteRNumber(reinterpret(UInt64, pointer_from_objref(t))).data
    else
        return getfield(t, s)
    end
end

function donate_argument!(
    ::Any, ::Union{ProbProgTrace,Constraint}, ::Int, ::Any, ::Any
)
    return nothing
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Union{ProbProgTrace,Constraint}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    if mode == ConcreteToTraced
        haskey(seen, prev) && return seen[prev]::TracedRNumber{UInt64}
        result = TracedRNumber{UInt64}((path,), nothing)
        seen[prev] = result
        return result
    elseif mode == TracedToTypes
        push!(path, typeof(prev))
        return nothing
    else
        error("Unsupported mode for $(typeof(prev)): $mode")
    end
end
