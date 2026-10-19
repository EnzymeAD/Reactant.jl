import KernelInterface as KI
using Adapt: Adapt

# ToDo: Include XLA client, device and sharding in ReactantBackend struct, to
# support more complex applications? If so, need to adapt implementation of
# `KI.get_backend` and `KI.allocate` accordingly.
struct ReactantBackend <: KI.GPU end

function Base.getproperty(x::ReactantBackend, sym::Symbol)
    if sym === :always_inline
        return true
    elseif sym === :prefer_blocks
        return false
    else
        return Base.getfield(x, sym)
    end
end

function KI.allocate(::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    ET = unwrapped_eltype(T)

    # Inside a trace there is no uninitialized device memory to hand back, and a concrete array
    # cannot take part in the traced program: filling or otherwise mutating one re-enters the
    # compiler. Return a traced array instead, mirroring what `similar` already does when it is
    # asked for a traced element type.
    within_compile() && return Ops.fill(zero(ET), dims)

    return ConcreteRArray{ET}(undef, dims)
end

function KI.zeros(b::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    A = KI.allocate(b, T, dims)
    isempty(A) || fill!(A, zero(T))
    return A
end
function KI.ones(b::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    A = KI.allocate(b, T, dims)
    isempty(A) || fill!(A, one(T))
    return A
end

KI.get_backend(::AnyTracedRArray) = ReactantBackend()
KI.get_backend(::UnionAnyConcreteRArray) = ReactantBackend()
function KI.synchronize(::ReactantBackend) end

Adapt.adapt_storage(::ReactantBackend, a::Array) = a
Adapt.adapt_storage(::ReactantBackend, a::Reactant.AnyTracedRArray) = a
Adapt.adapt_storage(::ReactantBackend, a::Reactant.AnyConcretePJRTArray) = a
Adapt.adapt_storage(::ReactantBackend, a::Reactant.AnyConcreteIFRTArray) = a

## memory operations

function KI.copyto!(::ReactantBackend, A, B)
    Base.copyto!(A, B)
    return A
end

## kernel launch

KI.argconvert(k::KI.Kernel{ReactantBackend}, arg) = arg

function tokw(ndrange, workgroupsize, obj, args...)
    @inline obj(args...; ndrange, workgroupsize)
end

function (obj::KI.Kernel{ReactantBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    if Reactant.precompiling()
        Reactant.@code_hlo optimize = false tokw(ndrange, workgroupsize, obj, args...)
    else
        Reactant.@jit tokw(ndrange, workgroupsize, obj, args...)
    end
    return nothing
end

@static if VERSION < v"1.12-"
    Reactant.@reactant_overlay Base.@nospecializeinfer @noinline function (
        obj::KI.Kernel{ReactantBackend}
    )(
        @nospecialize args...; ndrange=nothing, workgroupsize=nothing
    )
        return Reactant.call_with_reactant(
            Reactant.ka_with_reactant, ndrange, workgroupsize, obj, args...
        )
    end
else
    Reactant.@reactant_overlay function (obj::KI.Kernel{ReactantBackend})(
        args...; ndrange=nothing, workgroupsize=nothing
    )
        Base.@_noinline_meta
        Base.@_nospecializeinfer_meta
        return Reactant.call_with_reactant(
            Reactant.ka_with_reactant, ndrange, workgroupsize, obj, args...
        )
    end
end
