module Reactant

using ReactantCore: ReactantCore, @trace, MissingTracedValue

using Adapt: Adapt, WrappedArray

# auxiliary types and functions
include("OrderedIdDict.jl")

using Enzyme

@static if isdefined(Core, :BFloat16)
    const ReactantPrimitive = Union{
        Bool,
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Int64,
        UInt64,
        Float16,
        Core.BFloat16,
        Float32,
        Float64,
        Complex{Float32},
        Complex{Float64},
    }
else
    const ReactantPrimitive = Union{
        Bool,
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Int64,
        UInt64,
        Float16,
        Float32,
        Float64,
        Complex{Float32},
        Complex{Float64},
    }
end

abstract type RArray{T<:ReactantPrimitive,N} <: AbstractArray{T,N} end
abstract type RNumber{T<:ReactantPrimitive} <: Number end

function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    return reshape(A, Base._reshape_uncolon(A, dims))
end

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:RArray}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    if RT <: ConcreteRArray
        res = RT(zeros(eltype(RT), size(prev)))
        seen[prev] = res
        return res
    end

    if RT <: TracedRArray
        res = broadcast_to_size(eltype(RT)(0), size(prev))
        seen[prev] = res
        return res
    end

    attr = fill(MLIR.IR.Attribute(eltype(RT)(0)), mlir_type(prev))
    cst = MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
    res = RT((), cst)
    seen[prev] = res
    return res
end

include("mlir/MLIR.jl")
include("XLA.jl")
include("Interpreter.jl")

include("utils.jl")

include("ConcreteRArray.jl")
include("TracedRNumber.jl")
include("TracedRArray.jl")

# XXX: Generalize using @reactant_override
# XXX: move to a different place
# @reactant_override
# function Base.mapfoldl(f, op, itr::Vector{<:TracedRArray}; init = Base._InitialValue())
#     itr0, itr_rem = Iterators.peel(itr)

#     # @show itr0

#     fnwrap1, f_compiled, traced_result1, result1, seen_args1, ret1, linear_args1, in_tys1, linear_results1 = make_mlir_fn(
#         f, (itr0,), (), string(f) * "_map", false
#     )
#     # @show f_compiled

#     # @show result1

#     # printfn(f_compiled)

#     # fnwrap2, op_compiled, traced_result2, result2, seen_args2, ret2, linear_args2, in_tys2, linear_results2 = make_mlir_fn(
#     #     op, (result1), (), string(op) * "_map", false
#     # )

#     # printfn(op_compiled)

#     # If we can't successfully compile we will unroll the reduce operation
#     return Base.mapfoldl_impl(f, op, init, itr)
# end

# XXX: Multiple arguments
function Base.map(f, itr::Vector{<:TracedRArray})
    # return f.(itr)
    # itr0, itr_rem = Iterators.peel(itr)

    f_wrapped = let f = f
        (i, N, xs) -> f(xs[i])
    end

    @show f_wrapped(1, length(itr), itr)

    (fnwrap, f_compiled, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results) = make_mlir_fn(
        f_wrapped, (1, length(itr), itr), (), string(f) * "_map", false;
        return_dialect=:stablehlo,
        # no_args_in_result=true,,
        track_numbers=(Number,),
    )

    cond(i, N, x) = i ≤ N

    (_, cond_compiled, cond_results, _, _, _, _, _, cond_linear_results) = make_mlir_fn(
        cond, (1, length(itr), itr), (), string(cond) * "_condition", false;
        return_dialect=:stablehlo,
        track_numbers=(Number,),
        no_args_in_result=true,
    )

    @show cond_compiled


    # region = MLIR.IR.Region()
    # MLIR.API.mlirRegionTakeBody(region, MLIR.API.mlirOperationGetRegion(compiled_fn, 0))

    @show f_compiled
    @show traced_result
    @show linear_results

    return error(1)
end

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

include("ControlFlow.jl")
include("Tracing.jl")
include("Compiler.jl")

using .Compiler: @compile, @code_hlo, @jit, traced_getfield, create_result, compile
export ConcreteRArray, ConcreteRNumber, @compile, @code_hlo, @jit, @trace

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

function set_default_backend(backend::XLA.Client)
    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    backend = XLA.backends[backend]
    return XLA.default_backend[] = backend
end

end # module
