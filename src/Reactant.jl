module Reactant

using Enzyme

abstract type RArray{T,N} <: AbstractArray{T,N} end

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
include("TracedRArray.jl")
include("Tracing.jl")

struct MakeConcreteRArray{T,N}
    shape::NTuple{N,Int}
end
struct MakeArray{AT,Vals} end
struct MakeString{AT,Val} end
struct MakeStruct{AT,Val} end
struct MakeVal{AT} end
struct MakeSymbol{AT} end

function make_valable(tocopy)
    if tocopy isa ConcreteRArray
        return MakeConcreteRArray{eltype(tocopy),ndims(tocopy)}(size(tocopy))
    end
    if tocopy isa Array
        return MakeArray{Core.Typeof(tocopy),Tuple{map(make_valable, tocopy)...}}()
    end
    if tocopy isa Symbol
        return tocopy
    end
    if tocopy isa Int || tocopy isa AbstractFloat || tocopy isa Nothing || tocopy isa Type
        return MakeVal{Val{tocopy}}()
    end
    if tocopy isa AbstractString
        return MakeString{Core.Typeof(tocopy),Symbol(string)}() || T <: Nothing
    end
    T = Core.Typeof(tocopy)
    if tocopy isa Tuple || tocopy isa NamedTuple || isstructtype(T)
        elems = []
        nf = fieldcount(T)
        for i in 1:nf
            push!(elems, make_valable(getfield(tocopy, i)))
        end
        return MakeStruct{Core.Typeof(tocopy),Tuple{elems...}}()
    end

    return error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
end

function create_result(tocopy::MakeConcreteRArray{T,N}, path, result_stores) where {T,N}
    return :(ConcreteRArray{$T,$N}($(result_stores[path]), $(tocopy.shape)))
end

function create_result(tocopy::Tuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, (path..., k), result_stores))
    end
    return quote
        ($(elems...),)
    end
end

function create_result(tocopy::NamedTuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, (path..., k), result_Stores))
    end
    return quote
        NamedTuple{$(keys(tocopy))}($elems)
    end
end

function create_result(::MakeArray{AT,tocopy}, path, result_stores) where {AT,tocopy}
    elems = Expr[]
    for (i, v) in enumerate(tocopy.parameters)
        push!(elems, create_result(v, (path..., i), result_stores))
    end
    return quote
        $(eltype(AT))[$(elems...)]
    end
end

function create_result(::MakeVal{Val{nothing}}, path, result_stores)
    return :(nothing)
end

function create_result(::MakeVal{Val{elem}}, path, result_stores) where {elem}
    return :($elem)
end

function create_result(tocopy::Symbol, path, result_stores)
    return Meta.quot(tocopy)
end

function create_result(::MakeString{AT,Val}, path, result_stores) where {AT,Val}
    return :($(AT(Val)))
end

function create_result(::MakeStruct{AT,tocopy}, path, result_stores) where {AT,tocopy}
    # @info "create_result" AT tocopy path tocopy.parameters result_stores
    elems = Union{Symbol,Expr}[]
    for (i, v) in enumerate(tocopy.parameters)
        ev = create_result(v, (path..., i), result_stores)
        push!(elems, ev)
    end
    return Expr(:new, AT, elems...)
end

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

include("Compiler.jl")

function set_default_backend(backend::XLA.Client)
    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    backend = XLA.backends[backend]
    return XLA.default_backend[] = backend
end

end # module
