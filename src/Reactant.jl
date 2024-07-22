module Reactant

using PackageExtensionCompat
using Enzyme

include("mlir/MLIR.jl")
include("XLA.jl")
include("Interpreter.jl")
include("utils.jl")

abstract type RArray{T,N} <: AbstractArray{T,N} end

function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    return reshape(A, Base._reshape_uncolon(A, dims))
end

function mlir_type(x::RArray{T,N}) where {T,N}
    return MLIR.IR.TensorType(size(x), MLIR.IR.Type(T))
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

struct XLAArray{T,N} <: RArray{T,N}
    # size::NTuple{N,Int}
end

mutable struct ConcreteRArray{T,N} <: RArray{T,N}
    data::XLA.AsyncBuffer
    #	data::XLAArray{T, N}
    shape::NTuple{N,Int}
end

ConcreteRArray(data::T) where {T<:Number} = ConcreteRArray{T,0}(data, ())

function ConcreteRArray(
    data::Array{T,N}; client=XLA.default_backend[], idx=XLA.default_device_idx[]
) where {T,N}
    device = XLA.ClientGetDevice(client, idx)
    return ConcreteRArray{T,N}(
        XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, data, device), nothing), size(data)
    )
    # ConcreteRArray{T, size(data), N}(XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, XLA.to_row_major(data), device), nothing))
end

Base.size(x::ConcreteRArray) = x.shape

function Base.reshape(A::ConcreteRArray{T,N}, dims::NTuple{NT,Int}) where {T,N,NT}
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    host = convert(Array{T,N}, A)
    # HLO reshape semantics collapse the opposite so enforce on Julia Side
    # until we later make the transpose/reshape/transpose
    host = reshape(host, dims)
    client = XLA.client(A.data)
    device = XLA.device(A.data)
    buffer = XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, host, device), nothing)
    return ConcreteRArray{T,NT}(buffer, dims)
    # ConcreteRArray{T, dims, NT}(XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, XLA.to_row_major(host), device), nothing))
end

function Base.convert(::Type{T}, X::ConcreteRArray{ElType,N}) where {T<:Array,ElType,N}
    data = Array{ElType,N}(undef, size(X)...) # TODO replace for `similar`?
    XLA.await(X.data)
    buf = X.data.buffer
    GC.@preserve data buf begin
        XLA.BufferToHost(buf, pointer(data))
    end
    return data
    # XLA.from_row_major(data)
end

# function Base.similar(x::ConcreteRArray{T,N}, ::Type{T2}) where {T,N,T2}
#     return ConcreteRArray{T,N}(x.data)
# end
# function Base.convert(::Type{ConcreteRArray{T2,N}}, x::ConcreteRArray{T,N}) where {T,N,T2}
#     return ConcreteRArray{T,N}(x.data)
# end

function to_float(X::ConcreteRArray{T,0}) where {T}
    data = Ref{T}()
    XLA.await(X.data)
    buf = X.data.buffer
    GC.@preserve data buf begin
        XLA.BufferToHost(buf, data)
    end
    return data[]
end

function Base.isapprox(x::ConcreteRArray{T,0}, y; kwargs...) where {T}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.isapprox(x, y::ConcreteRArray{T,0}; kwargs...) where {T}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.isapprox(
    x::ConcreteRArray{T,0}, y::ConcreteRArray{T2,0}; kwargs...
) where {T,T2}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.print_array(io::IO, X::ConcreteRArray)
    if X.data == XLA.AsyncEmptyBuffer
        println(io, "<Empty buffer>")
        return nothing
    end
    return Base.print_array(io, convert(Array, X))
end

function Base.show(io::IO, X::ConcreteRArray)
    if X.data == XLA.AsyncEmptyBuffer
        println(io, "<Empty buffer>")
        return nothing
    end
    return Base.show(io, convert(Array, X))
end

function Base.getindex(a::ConcreteRArray{T}, args::Vararg{Int,N}) where {T,N}
    if a.data == XLA.AsyncEmptyBuffer
        throw("Cannot getindex from empty buffer")
    end
    # error("""Scalar indexing is disallowed.""")
    XLA.await(a.data)
    if XLA.BufferOnCPU(a.data.buffer)
        buf = a.data.buffer
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.UnsafeBufferPointer(buf))
            start = 0
            for i in 1:N
                start *= size(a, N - i + 1)
                start += (args[N - i + 1] - 1)
                # start *= size(a, i)
                # start += (args[i]-1)
            end
            start += 1
            return unsafe_load(ptr, start)
        end
    end
    return convert(Array, a)[args...]
end

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
    # PackageExtensionCompat: required for weakdeps to work in Julia <1.9
    @require_extensions

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
