function get_buffer(
    x::Union{ConcretePJRTArray,ConcretePJRTNumber}; no_error_for_scalar=false
)
    if Sharding.is_sharded(x.sharding)
        # For scalars this is mostly replicated
        no_error_for_scalar && return first(x.data).buffer
        error("`x` is sharded, so `get_buffer` is not defined")
    end
    return only(x.data).buffer
end

for runtime in (:PJRT, :IFRT)
    numType = Symbol(:Concrete, runtime, :Number)
    arrType = Symbol(:Concrete, runtime, :Array)
    @eval function Base.collect(x::$(numType){T}) where {T}
        return collect($(arrType){T,0}(copy(x).data, (), x.sharding))
    end
end

# copy
function Base.copy(x::Union{AbstractConcreteArray,AbstractConcreteNumber})
    fn = Reactant.compile(copy, (x,))
    return fn(x)
end

# deepcopy
function Base.deepcopy(x::Union{AbstractConcreteArray,AbstractConcreteNumber})
    fn = Reactant.compile(copy, (x,))
    return fn(x)
end

# One more reason why users shouldn't call `deepcopy`
function Base.deepcopy_internal(
    x::Union{AbstractConcreteArray,AbstractConcreteNumber}, stackdict::IdDict
)
    haskey(stackdict, x) && return stackdict[x]
    return deepcopy(x)
end

Base.size(::AbstractConcreteNumber) = ()
Base.real(x::AbstractConcreteNumber{<:Real}) = x
function Base.rtoldefault(T::Type{<:AbstractConcreteNumber})
    return T(Base.rtoldefault(unwrapped_eltype(T)))
end

Base.strides(x::AbstractConcreteArray) = Base.size_to_strides(1, size(x)...)

Base.OneTo(x::AbstractConcreteNumber{<:Integer}) = Base.OneTo(to_number(x))

@static if isdefined(Base, :unchecked_oneto)
    function Base.unchecked_oneto(x::AbstractConcreteNumber{<:Integer})
        return Base.unchecked_oneto(to_number(x))
    end
end

# Ensure the device and client are the same as the input
for numType in (:ConcretePJRTNumber, :ConcreteIFRTNumber)
    @eval function Base.float(x::$(numType){T}) where {T}
        return $(numType)(
            float(T)(to_number(x)); client=XLA.client(x), device=XLA.device(x), x.sharding
        )
    end
end

# written like this to avoid ambiguity errors
for T in Base.uniontypes(ReactantPrimitive)
    @eval (::Type{$(T)})(x::AbstractConcreteNumber) = convert($T, x)
end

function Base.convert(::Type{T}, x::AbstractConcreteNumber) where {T<:Number}
    return convert(T, to_number(x))
end

Adapt.adapt_storage(::Type{T}, x::AbstractArray) where {T<:AbstractConcreteArray} = T(x)

Base.size(x::AbstractConcreteArray) = x.shape

function Base.isempty(x::Union{AbstractConcreteArray,AbstractConcreteNumber})
    data = x.data
    data isa Tuple && return any(isempty, data)
    return isempty(data)
end

function Base.isempty(x::Union{WrappedConcretePJRTArray,WrappedConcreteIFRTArray})
    return isempty(ancestor(x))
end

function Base.convert(::Type{<:Array}, X::AbstractConcreteArray{T,N}) where {T,N}
    if has_padding(X)
        padding = get_padding(X)
        data = Array{T,N}(undef, (size(X) .+ padding)...)
        write_to_host_buffer!(data, X)
        return view(data, [1:size(X, i) for i in 1:ndims(X)]...)
    else
        data = Array{T,N}(undef, size(X)...)
        write_to_host_buffer!(data, X)
        return data
    end
end

function write_to_host_buffer!(data::Array, X::ConcretePJRTArray{T,N}) where {T,N}
    if Sharding.is_sharded(X)
        completed = Set{eltype(X.sharding.device_to_array_slices)}()
        for idx in 1:length(X.data)
            slice = X.sharding.device_to_array_slices[idx]
            if slice ∉ completed
                push!(completed, slice)
            else
                continue
            end
            data_slice = data[slice...]
            XLA.to_host(X.data[idx], data_slice, Reactant.Sharding.NoSharding())
            data[slice...] .= data_slice
        end
    else
        XLA.to_host(XLA.synced_buffer(only(X.data)), data, Reactant.Sharding.NoSharding())
    end
    return nothing
end

function write_to_host_buffer!(data::Array, X::ConcreteIFRTArray{T,N}) where {T,N}
    XLA.to_host(X.data, data, X.sharding)
    return nothing
end

function Base.convert(
    ::Type{<:Array}, X::Union{WrappedConcretePJRTArray,WrappedConcreteIFRTArray}
)
    fn = compile(TracedUtils.materialize_traced_array, (X,))
    return convert(Array, fn(X))
end

Base.Array(x::Union{AnyConcretePJRTArray,AnyConcreteIFRTArray}) = convert(Array, x)

function synchronize(x::Union{ConcretePJRTArray,ConcretePJRTNumber})
    foreach(wait, x.data)
    return nothing
end
function synchronize(x::Union{ConcreteIFRTArray,ConcreteIFRTNumber})
    wait(x.data)
    return nothing
end

function to_number(tp::Base.TwicePrecision)
    return Base.TwicePrecision(to_number(tp.hi), to_number(tp.lo))
end

to_number(x::Number) = x

function to_number(X::ConcretePJRTScalar{T}) where {T}
    data = Ref{T}()
    XLA.to_host(get_buffer(X; no_error_for_scalar=true), data, X.sharding)
    return data[]
end

function to_number(X::ConcreteIFRTScalar{T}) where {T}
    data = Ref{T}()
    XLA.to_host(X.data, data, X.sharding)
    return data[]
end

function Base.convert(
    ::Type{T}, x::Union{ConcretePJRTScalar{T},ConcreteIFRTScalar{T}}
) where {T<:Number}
    return to_number(x)
end

for jlop in (:(Base.abs),), T in (AbstractConcreteNumber,)
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
    T in (AbstractConcreteNumber, AbstractConcreteArray{<:Any,0})

    @eval begin
        $(jlop)(x::$(T), y::$(T)) = $(jlop)(to_number(x), to_number(y))
        $(jlop)(x::$(T), y::Number) = $(jlop)(to_number(x), y)
        $(jlop)(x::Number, y::$(T)) = $(jlop)(x, to_number(y))
    end
end

Base.:^(x::AbstractConcreteNumber, y::Integer) = ^(to_number(x), y)

for jlop in (:(Base.isnan), :(Base.isfinite)),
    T in (AbstractConcreteNumber, AbstractConcreteArray{<:Any,0})

    @eval $(jlop)(x::$(T)) = $(jlop)(to_number(x))
end

for T in (AbstractConcreteNumber, AbstractConcreteArray{<:Any,0})
    for (T1, T2) in ((T, Number), (Number, T), (T, T))
        @eval begin
            function Base.isapprox(x::$(T1), y::$(T2); kwargs...)
                return Base.isapprox(to_number(x), to_number(y); kwargs...)
            end
            function Base.isapprox(
                x::AbstractArray{<:$(T1)}, y::AbstractArray{<:$(T2)}; kwargs...
            )
                return Base.isapprox(to_number.(x), to_number.(y); kwargs...)
            end
        end
    end
end

for (T1, T2) in (
    (AnyConcretePJRTArray, AbstractArray),
    (AbstractArray, AnyConcretePJRTArray),
    (AnyConcretePJRTArray, AnyConcretePJRTArray),
    (AnyConcreteIFRTArray, AbstractArray),
    (AbstractArray, AnyConcreteIFRTArray),
    (AnyConcreteIFRTArray, AnyConcreteIFRTArray),
)
    @eval begin
        function Base.isapprox(x::$(T1), y::$(T2); kwargs...)
            return Base.isapprox(convert(Array, x), convert(Array, y); kwargs...)
        end
        Base.:(==)(x::$(T1), y::$(T2)) = convert(Array, x) == convert(Array, y)
    end
end

function Base.show(io::IO, X::Union{ConcretePJRTScalar,ConcreteIFRTScalar})
    if isempty(X)
        print(io, "<Empty Buffer eltype $(eltype(X)) of size $(size(X))>")
        return nothing
    end
    print(io, "$(typeof(X))(")
    show(io, to_number(X))
    print(io, ")")
    return nothing
end

function Base.print_array(io::IO, X::Union{AnyConcretePJRTArray,AnyConcreteIFRTArray})
    if isempty(X)
        print(io, "<Empty Buffer eltype $(eltype(X)) of size $(size(X))>")
        return nothing
    end
    return Base.print_array(io, convert(Array, X))
end

function Base.showarg(
    io::IO, a::Union{ConcretePJRTArray{T,N},ConcreteIFRTArray{T,N}}, toplevel
) where {T,N}
    toplevel || print(io, "::")
    print(io, "$(typeof(a).name.wrapper){$T,$N}")
    if Sharding.is_sharded(a)
        (; hlo_sharding) = Sharding.HloSharding(
            Sharding.unwrap_shardinfo(a.sharding), size(a)
        )
        print(io, " with \"mhlo.sharding = $(string(hlo_sharding))\"")
    end
    return nothing
end

function Base.show(io::IO, X::Union{AnyConcretePJRTArray,AnyConcreteIFRTArray})
    if isempty(X)
        print(io, "<Empty Buffer eltype $(eltype(X)) of size $(size(X))>")
        return nothing
    end
    print(io, "$(typeof(X))(")
    show(io, convert(Array, X))
    print(io, ")")
    return nothing
end

function Base.getindex(a::ConcretePJRTArray{T,N}, args::Vararg{Int,N}) where {T,N}
    isempty(a) && throw("Cannot getindex from empty buffer")

    wait(a)
    if buffer_on_cpu(a) && !Sharding.is_sharded(a)
        buf = get_buffer(a)
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.unsafe_buffer_pointer(buf))
            start = 0
            for i in 1:N
                start *= size(a, N - i + 1)
                start += (args[N - i + 1] - 1)
            end
            start += 1
            return unsafe_load(ptr, start)
        end
    end

    GPUArraysCore.assertscalar("getindex(::ConcretePJRTArray, ::Vararg{Int, N})")
    return convert(Array, a)[args...]
end

function Base.getindex(a::ConcreteIFRTArray{T,N}, args::Vararg{Int,N}) where {T,N}
    GPUArraysCore.assertscalar("getindex(::ConcreteIFRTArray, ::Vararg{Int, N})")
    return convert(Array, a)[args...]
end

function Base.getindex(
    a::Union{ConcreteIFRTArray{T,N},ConcretePJRTArray{T,N}}, args::Vararg{Any,N}
) where {T,N}
    return compile(getindex, (a, args...))(a, args...)
end

# This doesn't follow the semantics of getindex with ranges. It is mostly meant to be used
# inside Compiler.jl
@inline function _fast_slice(
    a::AbstractConcreteArray{T,N}, args::Vararg{UnitRange,N}
) where {T,N}
    # Avoid slicing all-together
    args == ntuple(Base.Fix1(UnitRange, 1) ∘ Base.Fix1(size, a), N) && return a
    # For all other cases do a compile
    fn = compile(getindex, (a, args...))
    return fn(a, args...)
end

@inline _fast_slice(a::AbstractConcreteNumber) = a

function mysetindex!(a, v, args::Vararg{Any,N}) where {N}
    setindex!(a, v, args...)
    return nothing
end

function Base.setindex!(a::ConcretePJRTArray{T}, v, args::Vararg{Int,N}) where {T,N}
    isempty(a) && throw("Cannot setindex! to empty buffer")

    wait(a)
    if buffer_on_cpu(a) && !Sharding.is_sharded(a)
        buf = get_buffer(a)
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.unsafe_buffer_pointer(buf))
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

    GPUArraysCore.assertscalar("setindex!(::ConcretePJRTArray, ::Any, ::Vararg{Int, N})")
    fn = compile(mysetindex!, (a, v, args...))
    fn(a, v, args...)
    return a
end

function Base.setindex!(a::ConcreteIFRTArray, v, args::Vararg{Int,N}) where {N}
    isempty(args) && throw("Cannot setindex! to empty buffer")

    GPUArraysCore.assertscalar("setindex!(::ConcreteIFRTArray, ::Any, ::Vararg{Int, N})")
    fn = compile(mysetindex!, (a, v, args...))
    fn(a, v, args...)
    return a
end

# TODO is there any way to allocate an uninitialized buffer in XLA?
function Base.similar(
    a::ConcretePJRTArray{T}, (::Type{S})=T, dims::Dims=size(a)
) where {T,S}
    return ConcretePJRTArray(
        Array{S}(undef, dims); client=XLA.client(a), device=XLA.device(a), a.sharding
    )
end
Base.similar(a::ConcretePJRTArray, dims::Dims) = similar(a, eltype(a), dims)
function Base.similar(::Type{ConcretePJRTArray{T}}, dims) where {T}
    return ConcretePJRTArray(similar(Array{T}, dims))
end

function Base.similar(
    a::ConcreteIFRTArray{T}, (::Type{S})=T, dims::Dims=size(a)
) where {T,S}
    return ConcreteIFRTArray(
        Array{S}(undef, dims); client=XLA.client(a), device=XLA.device(a), a.sharding
    )
end
Base.similar(a::ConcreteIFRTArray, dims::Dims) = similar(a, eltype(a), dims)
function Base.similar(::Type{ConcreteIFRTArray{T}}, dims) where {T}
    return ConcreteIFRTArray(similar(Array{T}, dims))
end

# Broadcasting interface
Base.BroadcastStyle(::Type{<:ConcretePJRTArray}) = Broadcast.ArrayStyle{ConcretePJRTArray}()
Base.BroadcastStyle(::Type{<:ConcreteIFRTArray}) = Broadcast.ArrayStyle{ConcreteIFRTArray}()

# XXX: correct device + sharding?
function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcretePJRTArray}}, ::Type{T}
) where {T}
    return ConcretePJRTArray(similar(Array{T}, axes(bc)))
end
function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcreteIFRTArray}}, ::Type{T}
) where {T}
    return ConcreteIFRTArray(similar(Array{T}, axes(bc)))
end

# TODO replace this copy for `setindex!` maybe? how to copy data to already existing buffer? (i.e. `copyto!`)
function Base.copy(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcretePJRTArray}})
    bc = Broadcast.flatten(bc)
    for x in bc.args
        x isa ConcretePJRTArray && wait(x)
    end

    if all(buffer_on_cpu, bc.args) && all(
        x ->
            !(x isa ConcretePJRTArray) ||
            (x isa ConcretePJRTArray && !Sharding.is_sharded(x)),
        bc.args,
    )
        ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
        if !Base.isconcretetype(ElType)
            throw(
                ErrorException(
                    "`copy` on `ConcretePJRTArray` for non-concrete eltype is not implemented",
                ),
            )
        end
        aux = copyto!(
            similar(Array{ElType}, axes(bc)), convert(Broadcast.Broadcasted{Nothing}, bc)
        )
        return ConcretePJRTArray(aux) # XXX: result should be on correct device?
    end

    fn = compile(Broadcast.BroadcastFunction(bc.f), (bc.args...,))
    return fn(bc.args...)
end

function Base.copy(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{ConcreteIFRTArray}})
    bc = Broadcast.flatten(bc)
    fn = compile(Broadcast.BroadcastFunction(bc.f), (bc.args...,))
    return fn(bc.args...)
end

function mycopyto!(dest, src)
    dest .= src # use broadcasting instead of copyto!
    return nothing
end

for aType in (:ConcretePJRTArray, :ConcreteIFRTArray)
    @eval function Base.copyto!(dest::$(aType), src::$(aType))
        # We can't directly set the data field. it will alias the inner buffers without
        # actually copying them.
        fn = compile(mycopyto!, (dest, src))
        fn(dest, src)
        return dest
    end
end

function Base.copyto!(
    dest::Union{AnyConcreteIFRTArray,AnyConcretePJRTArray}, src::AbstractConcreteArray
)
    fn = compile(mycopyto!, (dest, src))
    fn(dest, src)
    return dest
end

for aType in (:ConcretePJRTArray, :ConcreteIFRTArray)
    anyaType = Symbol(:Any, aType)
    @eval function Base.copyto!(dest::$(anyaType), src::Array{<:ReactantPrimitive})
        ancestor_dest = ancestor(dest)
        return copyto!(
            dest,
            $(aType)(
                src;
                sharding=ancestor_dest.sharding,
                client=XLA.client(ancestor_dest),
                device=XLA.device(ancestor_dest),
            ),
        )
    end
end

for aType in (:ConcretePJRTArray, :ConcreteIFRTArray)
    @eval begin
        function Base.copyto!(
            dest::AbstractConcreteArray,
            src::Broadcast.Broadcasted{Broadcast.ArrayStyle{$(aType)}},
        )
            dest.data = copy(src).data
            return dest
        end

        function Base.copyto!(
            dest::Array, src::Broadcast.Broadcasted{Broadcast.ArrayStyle{$(aType)}}
        )
            write_to_host_buffer!(dest, copy(src))
            return dest
        end

        function Base.copyto!(
            dest::AbstractArray, src::Broadcast.Broadcasted{Broadcast.ArrayStyle{$(aType)}}
        )
            copyto!(dest, convert(Array, copy(src)))
            return dest
        end
    end
end

Base.collect(x::AbstractConcreteArray) = convert(Array, x)

function Base.mapreduce(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(A::AbstractConcreteArray{T,N});
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
buffer_on_cpu(x::ConcretePJRTArray) = all(XLA.buffer_on_cpu, x.data)
buffer_on_cpu(x::ConcreteIFRTArray) = XLA.buffer_on_cpu(x.data)

function Base.zero(x::ConcretePJRTArray{T,N}) where {T,N}
    return ConcretePJRTArray(
        zeros(T, size(x)...); client=XLA.client(x), device=XLA.device(x), x.sharding
    )
end
function Base.zero(x::ConcreteIFRTArray{T,N}) where {T,N}
    return ConcreteIFRTArray(
        zeros(T, size(x)...); client=XLA.client(x), device=XLA.device(x), x.sharding
    )
end

function Base.fill!(a::ConcretePJRTArray{T,N}, val) where {T,N}
    isempty(a) && throw("Cannot setindex! to empty buffer")

    wait(a)
    if buffer_on_cpu(a) && !Sharding.is_sharded(a)
        buf = get_buffer(a)
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{T}, XLA.unsafe_buffer_pointer(buf))
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

function Base.fill!(a::ConcreteIFRTArray{T,N}, val) where {T,N}
    isempty(a) && throw("Cannot setindex! to empty buffer")

    idxs = ntuple(Returns(Colon()), N)
    fn = compile(mysetindex!, (a, val, idxs...))
    fn(a, val, idxs...)
    return a
end

function Base.fill!(x::Union{AnyConcreteIFRTArray,AnyConcretePJRTArray}, val)
    idxs = ntuple(Returns(Colon()), ndims(x))
    fn = compile(mysetindex!, (x, val, idxs...))
    fn(x, val, idxs...)
    return x
end

function mymapreducedim!(f, op, R, A)
    Base.mapreducedim!(f, op, R, A)
    return nothing
end

function Base.mapreducedim!(
    f,
    op,
    R::Union{AnyConcreteIFRTArray,AnyConcretePJRTArray},
    A::Union{Base.AbstractBroadcasted,AbstractArray},
)
    fn = compile(mymapreducedim!, (f, op, R, A))
    fn(f, op, R, A)
    return R
end

function mymap!(f, R, A)
    map!(f, R, A)
    return nothing
end

function Base.map!(f, R::Union{AnyConcreteIFRTArray,AnyConcretePJRTArray}, A::AbstractArray)
    fn = compile(mymap!, (f, R, A))
    fn(f, R, A)
    return R
end

# Directly initialize a Device Array
function Base.fill(
    ::Type{<:Union{ConcreteIFRTArray,ConcretePJRTArray}},
    val,
    dims::Vararg{Int};
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
)
    output_shardings = Sharding.is_sharded(sharding) ? Dict(1 => sharding) : nothing
    fn = Reactant.compile((); output_shardings) do
        return Ops.fill(val, collect(Int64, dims))
    end
    return fn()
end
