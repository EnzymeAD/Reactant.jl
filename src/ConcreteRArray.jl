struct XLAArray{T,N} <: RArray{T,N}
    # size::NTuple{N,Int}
end

mutable struct ConcreteRArray{T,N} <: RArray{T,N}
    data::XLA.AsyncBuffer
    #   data::XLAArray{T, N}
    shape::NTuple{N,Int}
end

mutable struct ConcreteRNumber{T} <: RNumber{T}
    data::XLA.AsyncBuffer
end

function ConcreteRNumber(
    data::T; client=XLA.default_backend[], idx=XLA.default_device_idx[], device=nothing
) where {T<:Number}
    crarray = ConcreteRArray(fill(data); client, idx, device)
    return ConcreteRNumber{T}(crarray.data)
end

Base.size(::ConcreteRNumber) = ()

# Ensure the device and client are the same as the input
function Base.float(x::ConcreteRNumber{T}) where {T}
    client = XLA.client(x.data)
    device = XLA.device(x.data)
    return ConcreteRNumber(float(T)(to_number(x)); client, device)
end

# written like this to avoid ambiguity errors
for T in Base.uniontypes(ReactantPrimitive)
    @eval (::Type{$(T)})(x::ConcreteRNumber) = convert($T, x)
end

Base.convert(::Type{T}, x::ConcreteRNumber) where {T<:Number} = convert(T, to_number(x))

function ConcreteRArray(
    data::T; client=XLA.default_backend[], idx=XLA.default_device_idx[], device=nothing
) where {T<:Number}
    Base.depwarn(
        "ConcreteRArray(data::Number) is deprecated, use ConcreteRNumber(data) instead",
        :ConcreteRArray,
    )
    return ConcreteRArray(fill(data); client, idx, device)
end

const ConcreteRScalar{T} = Union{ConcreteRArray{T,0},ConcreteRNumber{T}}

Adapt.adapt_storage(::Type{T}, x::AbstractArray) where {T<:ConcreteRArray} = T(x)

function ConcreteRArray(
    data::Array{T,N};
    client=XLA.default_backend[],
    idx=XLA.default_device_idx[],
    device=nothing,
) where {T,N}
    device = device === nothing ? XLA.ClientGetDevice(client, idx) : device
    return ConcreteRArray{T,N}(
        XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, data, device), nothing), size(data)
    )
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
Base.Array(x::ConcreteRArray) = convert(Array, x)

function synchronize(x::Union{ConcreteRArray,ConcreteRNumber})
    XLA.synced_buffer(x.data)
    return nothing
end

# function Base.similar(x::ConcreteRArray{T,N}, ::Type{T2}) where {T,N,T2}
#     return ConcreteRArray{T,N}(x.data)
# end
# function Base.convert(::Type{ConcreteRArray{T2,N}}, x::ConcreteRArray{T,N}) where {T,N,T2}
#     return ConcreteRArray{T,N}(x.data)
# end

function to_number(X::ConcreteRScalar{T}) where {T}
    data = Ref{T}()
    XLA.await(X.data)
    buf = X.data.buffer
    GC.@preserve data buf begin
        XLA.BufferToHost(buf, data)
    end
    return data[]
end

Base.convert(::Type{T}, x::ConcreteRScalar{T}) where {T} = to_number(x)

for jlop in (
        :(Base.isless),
        :(Base.:+),
        :(Base.:-),
        :(Base.:*),
        :(Base.:/),
        :(Base.:^),
        :(Base.:(==)),
    ),
    T in (ConcreteRNumber, ConcreteRArray{<:Any,0})

    @eval begin
        $(jlop)(x::$(T), y::$(T)) = $(jlop)(to_number(x), to_number(y))
        $(jlop)(x::$(T), y::Number) = $(jlop)(to_number(x), y)
        $(jlop)(x::Number, y::$(T)) = $(jlop)(x, to_number(y))
    end
end

for T in (ConcreteRNumber, ConcreteRArray{<:Any,0})
    @eval begin
        function Base.isapprox(x::$(T), y::Number; kwargs...)
            return Base.isapprox(to_number(x), y; kwargs...)
        end

        function Base.isapprox(x::Number, y::$(T); kwargs...)
            return Base.isapprox(x, to_number(y); kwargs...)
        end

        function Base.isapprox(x::$(T), y::$(T); kwargs...)
            return Base.isapprox(to_number(x), to_number(y); kwargs...)
        end
    end
end

function Base.isapprox(x::ConcreteRArray, y::AbstractArray; kwargs...)
    return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
end
function Base.isapprox(x::AbstractArray, y::ConcreteRArray; kwargs...)
    return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
end
function Base.isapprox(x::ConcreteRArray, y::ConcreteRArray; kwargs...)
    return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
end

Base.:(==)(x::ConcreteRArray, y::AbstractArray) = convert(Array, x) == convert(Array, y)
Base.:(==)(x::AbstractArray, y::ConcreteRArray) = convert(Array, x) == convert(Array, y)
Base.:(==)(x::ConcreteRArray, y::ConcreteRArray) = convert(Array, x) == convert(Array, y)

function Base.show(io::IO, X::ConcreteRScalar{T}) where {T}
    if X.data == XLA.AsyncEmptyBuffer
        println(io, "<Empty buffer>")
        return nothing
    end
    str = sprint(show, to_number(X))
    return print(io, "$(typeof(X))($(str))")
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
    str = sprint(show, convert(Array, X))
    return print(io, "$(typeof(X))($(str))")
end

function Base.getindex(a::ConcreteRArray{T}, args::Vararg{Int,N}) where {T,N}
    if a.data == XLA.AsyncEmptyBuffer
        throw("Cannot getindex from empty buffer")
    end

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

    GPUArraysCore.assertscalar("getindex(::ConcreteRArray, ::Vararg{Int, N})")
    return convert(Array, a)[args...]
end

function mysetindex!(a, v, args::Vararg{Int,N}) where {N}
    setindex!(a, v, args...)
    return nothing
end

function Base.setindex!(a::ConcreteRArray{T}, v, args::Vararg{Int,N}) where {T,N}
    if a.data == XLA.AsyncEmptyBuffer
        throw("Cannot setindex! to empty buffer")
    end

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
            unsafe_store!(ptr, v, start)
        end
        return a
    end

    GPUArraysCore.assertscalar("setindex!(::ConcreteRArray, ::Any, ::Vararg{Int, N})")
    fn = Reactant.compile(mysetindex!, (a, v, args...))
    fn(a, v, args...)
    return a
end

# TODO is there any way to allocate an uninitialized buffer in XLA?
function Base.similar(a::ConcreteRArray{T}, ::Type{S}=T, dims::Dims=size(a)) where {T,S}
    return ConcreteRArray(Array{S}(undef, dims))
end
Base.similar(a::ConcreteRArray, dims::Dims) = similar(a, eltype(a), dims)

function Base.similar(::Type{ConcreteRArray{T}}, dims) where {T}
    return ConcreteRArray(similar(Array{T}, dims))
end

# Broadcasting interface
Base.BroadcastStyle(::Type{<:ConcreteRArray}) = Broadcast.ArrayStyle{ConcreteRArray}()
function Base.similar(
    bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcreteRArray}}, ::Type{T}
) where {T}
    return ConcreteRArray(similar(Array{T}, axes(bc)))
end

# TODO replace this copy for `setindex!` maybe? how to copy data to already existing buffer? (i.e. `copyto!`)
function Base.copy(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcreteRArray}})
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    if !Base.isconcretetype(ElType)
        throw(
            ErrorException(
                "`copy` on `ConcreteRArray` for non-concrete eltype is not implemented"
            ),
        )
    end

    aux = copyto!(similar(Array{ElType}, axes(bc)), bc)
    return ConcreteRArray(aux)
end
