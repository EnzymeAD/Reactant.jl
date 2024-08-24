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

function Base.convert(::Type{T}, x::ConcreteRArray{T,0}) where {T}
    return to_float(x)
end

Base.promote_rule(::Type{<:RArray{T1, 0}}, ::Type{T2}) where {T1, T2} = Base.promote_rule(T1, T2)

for jlop in (
    :(Base.isless),
    :(Base.:+),
    :(Base.:-),
    :(Base.:*),
    :(Base.:/),
    :(Base.:^),
)
@eval begin
function $jlop(x::ConcreteRArray{T,0}, y::ConcreteRArray{U,0}) where {T, U}
    return $jlop(to_float(x), to_float(y))
end
function $jlop(x::ConcreteRArray{T,0}, y) where {T}
    return $jlop(to_float(x), y)
end
function $jlop(x, y::ConcreteRArray{U,0}) where {U}
    return $jlop(x, to_float(y))
end
end
end

function Base.isapprox(x::ConcreteRArray{T,0}, y; kwargs...) where {T}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.isapprox(x, y::ConcreteRArray{T,0}; kwargs...) where {T}
    return Base.isapprox(x, to_float(y); kwargs...)
end

function Base.isapprox(
    x::ConcreteRArray{T,0}, y::ConcreteRArray{T2,0}; kwargs...
) where {T,T2}
    return Base.isapprox(to_float(x), to_float(y); kwargs...)
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
