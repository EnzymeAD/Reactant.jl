get_buffer(x::ConcreteRNumber) = x.data.buffer
get_buffer(x::ConcreteRArray{T,0}) where {T} = only(x.data).buffer
function get_buffer(x::ConcreteRArray{T,N}) where {T,N}
    if Sharding.is_sharded(x.sharding)
        error("`x` is sharded, so `get_buffer` is not defined")
    end
    return only(x.data).buffer
end

function Base.collect(x::ConcreteRNumber{T}) where {T}
    return collect(ConcreteRArray{T,0}(copy(x).data, ()))
end

Base.size(::ConcreteRNumber) = ()
Base.real(x::ConcreteRNumber{<:Real}) = x
function Base.rtoldefault(::Type{ConcreteRNumber{T}}) where {T}
    return ConcreteRNumber(Base.rtoldefault(T))
end

Base.strides(x::ConcreteRArray) = Base.size_to_strides(1, size(x)...)

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

Adapt.adapt_storage(::Type{T}, x::AbstractArray) where {T<:ConcreteRArray} = T(x)

Base.size(x::ConcreteRArray) = x.shape

Base.isempty(x::ConcreteRNumber) = x.data == XLA.AsyncEmptyBuffer
Base.isempty(x::ConcreteRArray) = any(==(XLA.AsyncEmptyBuffer), x.data)
Base.isempty(x::WrappedConcreteRArray) = isempty(ancestor(x))

function Base.convert(::Type{<:Array}, X::ConcreteRArray{T,N}) where {T,N}
    data = Array{T,N}(undef, size(X)...)
    XLA.await(X)

    if Sharding.is_sharded(X)
        # TODO: We can we much more efficient here and only move data from the minimal
        #       slices that populates the array.
        for idx in 1:length(X.data)
            buffer = X.data[idx].buffer
            # We can't use a pointer to a subarray since BufferToHost expects the data to
            # be contiguous.
            slice = X.sharding.device_to_array_slices[idx]
            data_slice = data[slice...]
            GC.@preserve data_slice buffer begin
                XLA.BufferToHost(buffer, pointer(data_slice))
            end
            data[slice...] = data_slice
        end
    else
        buf = only(X.data).buffer
        GC.@preserve data buf begin
            XLA.BufferToHost(buf, pointer(data))
        end
    end

    return data
end
function Base.convert(::Type{<:Array}, X::WrappedConcreteRArray)
    fn = compile(TracedUtils.materialize_traced_array, (X,))
    return convert(Array, fn(X))
end
Base.Array(x::AnyConcreteRArray) = convert(Array, x)

function synchronize(x::Union{ConcreteRArray,ConcreteRNumber})
    foreach(XLA.synced_buffer, x.data)
    return nothing
end

function to_number(X::ConcreteRScalar{T}) where {T}
    data = Ref{T}()
    XLA.await(X)
    buf = get_buffer(X)
    GC.@preserve data buf begin
        XLA.BufferToHost(buf, data)
    end
    return data[]
end

Base.convert(::Type{T}, x::ConcreteRScalar{T}) where {T} = to_number(x)

for jlop in (:(Base.abs),), T in (ConcreteRNumber,)
    @eval $(jlop)(x::$(T)) = $(jlop)(to_number(x))
end

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

for jlop in (:(Base.isnan), :(Base.isfinite)),
    T in (ConcreteRNumber, ConcreteRArray{<:Any,0})

    @eval $(jlop)(x::$(T)) = $(jlop)(to_number(x))
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

function Base.isapprox(x::AnyConcreteRArray, y::AbstractArray; kwargs...)
    return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
end
function Base.isapprox(x::AbstractArray, y::AnyConcreteRArray; kwargs...)
    return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
end
function Base.isapprox(x::AnyConcreteRArray, y::AnyConcreteRArray; kwargs...)
    return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
end

Base.:(==)(x::AnyConcreteRArray, y::AbstractArray) = convert(Array, x) == convert(Array, y)
Base.:(==)(x::AbstractArray, y::AnyConcreteRArray) = convert(Array, x) == convert(Array, y)
function Base.:(==)(x::AnyConcreteRArray, y::AnyConcreteRArray)
    return convert(Array, x) == convert(Array, y)
end

function Base.show(io::IO, X::ConcreteRScalar{T}) where {T}
    if isempty(X)
        print(io, "<Empty Buffer eltype $(eltype(X)) of size $(size(X))>")
        return nothing
    end
    print(io, "$(typeof(X))(")
    show(io, to_number(X))
    print(io, ")")
    return nothing
end

function Base.print_array(io::IO, X::AnyConcreteRArray)
    if isempty(X)
        print(io, "<Empty Buffer eltype $(eltype(X)) of size $(size(X))>")
        return nothing
    end
    return Base.print_array(io, convert(Array, X))
end

function Base.show(io::IO, X::AnyConcreteRArray)
    if isempty(X)
        print(io, "<Empty Buffer eltype $(eltype(X)) of size $(size(X))>")
        return nothing
    end
    print(io, "$(typeof(X))(")
    show(io, convert(Array, X))
    print(io, ")")
    return nothing
end

function Base.getindex(a::ConcreteRArray{T}, args::Vararg{Int,N}) where {T,N}
    isempty(a) && throw("Cannot getindex from empty buffer")

    XLA.await(a)
    if buffer_on_cpu(a) && !Sharding.is_sharded(a)
        buf = get_buffer(a)
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.UnsafeBufferPointer(buf))
            start = 0
            for i in 1:N
                start *= size(a, N - i + 1)
                start += (args[N - i + 1] - 1)
            end
            start += 1
            return unsafe_load(ptr, start)
        end
    end

    GPUArraysCore.assertscalar("getindex(::ConcreteRArray, ::Vararg{Int, N})")
    return convert(Array, a)[args...]
end

function mysetindex!(a, v, args::Vararg{Any,N}) where {N}
    setindex!(a, v, args...)
    return nothing
end

function Base.setindex!(a::ConcreteRArray{T}, v, args::Vararg{Int,N}) where {T,N}
    isempty(a) && throw("Cannot setindex! to empty buffer")

    XLA.await(a)
    if buffer_on_cpu(a) && !Sharding.is_sharded(a)
        buf = get_buffer(a)
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.UnsafeBufferPointer(buf))
            start = 0
            for i in 1:N
                start *= size(a, N - i + 1)
                start += (args[N - i + 1] - 1)
            end
            start += 1
            unsafe_store!(ptr, v, start)
        end
        return a
    end

    GPUArraysCore.assertscalar("setindex!(::ConcreteRArray, ::Any, ::Vararg{Int, N})")
    fn = compile(mysetindex!, (a, v, args...))
    fn(a, v, args...)
    return a
end

# TODO is there any way to allocate an uninitialized buffer in XLA?
function Base.similar(a::ConcreteRArray{T}, ::Type{S}=T, dims::Dims=size(a)) where {T,S}
    return ConcreteRArray(
        Array{S}(undef, dims); client=XLA.client(a), device=XLA.device(a), a.sharding
    )
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
    # XXX: correct device + sharding?
    return ConcreteRArray(similar(Array{T}, axes(bc)))
end

# TODO replace this copy for `setindex!` maybe? how to copy data to already existing buffer? (i.e. `copyto!`)
function Base.copy(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcreteRArray}})
    for x in bc.args
        x isa ConcreteRArray && XLA.await(x)
    end

    if all(buffer_on_cpu, bc.args) && all(
        x -> !(x isa ConcreteRArray) || (x isa ConcreteRArray && !Sharding.is_sharded(x)),
        bc.args,
    )
        ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
        if !Base.isconcretetype(ElType)
            throw(
                ErrorException(
                    "`copy` on `ConcreteRArray` for non-concrete eltype is not implemented"
                ),
            )
        end
        aux = copyto!(similar(Array{ElType}, axes(bc)), bc)
        return ConcreteRArray(aux) # XXX: result should be on correct device?
    end

    fn = compile(Broadcast.BroadcastFunction(bc.f), (bc.args...,))
    return fn(bc.args...)
end

function Base.copyto!(dest::ConcreteRArray, src::ConcreteRArray)
    dest.data = src.data
    return dest
end

Base.collect(x::AnyConcreteRArray) = convert(Array, x)

function Base.mapreduce(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(A::ConcreteRArray{T,N});
    dims=:,
    init=nothing,
) where {T,N}
    fn = compile(CallMapReduce(f, op, dims, init), (A,))
    return fn(A)
end

struct CallMapReduce{Fn,Op,Dims,Init}
    f::Fn
    op::Op
    dims::Dims
    init::Init
end

(f::CallMapReduce)(A) = Base.mapreduce(f.f, f.op, A; f.dims, f.init)

buffer_on_cpu(::Any) = true
function buffer_on_cpu(x::ConcreteRArray)
    return all(XLA.BufferOnCPU ∘ Base.Fix2(getproperty, :buffer), x.data)
end

function Ops.constant(x::ConcreteRArray; kwargs...)
    return Ops.constant(Base.convert(Array, x); kwargs...)
end

function Ops.constant(x::ConcreteRNumber{T}; kwargs...) where {T}
    return Ops.constant(Base.convert(T, x); kwargs...)
end

function Base.zero(x::ConcreteRArray{T,N}) where {T,N}
    return ConcreteRArray(
        zeros(T, size(x)...); client=XLA.client(x), device=XLA.device(x), x.sharding
    )
end

function Base.fill!(a::ConcreteRArray{T,N}, val) where {T,N}
    isempty(a) && throw("Cannot setindex! to empty buffer")

    XLA.await(a)
    if buffer_on_cpu(a) && !Sharding.is_sharded(a)
        buf = get_buffer(a)
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.UnsafeBufferPointer(buf))
            for start in 1:length(a)
                unsafe_store!(ptr, val, start)
            end
        end
        return a
    end

    idxs = ntuple(Returns(Colon()), N)
    fn = compile(mysetindex!, (a, val, idxs...))
    fn(a, val, idxs...)
    return a
end
