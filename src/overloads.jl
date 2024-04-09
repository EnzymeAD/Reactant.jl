function promote_to(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
    if !(rhs <: Number)
        if ElType != eltype(rhs)
            throw(ArgumentError("Cannot promote $(typeof(rhs)) to $(TracedRArray{ElType,Shape,N}) with different element types"))
        end
        if Shape != size(rhs)
            throw(ArgumentError("Cannot promote to TracedRArray with different shapes"))
        end
    end

    if isa(rhs, TracedRArray)
        if isa(rhs, Number)
            throw(ArgumentError("TODO broadcast"))
        end
        return rhs
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(ElType(rhs)), mlir_type(lhs))
        return TracedRArray{ElType,Shape,N}(nothing, MLIR.IR.stablehlo.constant(attr))
    end
    attr = MLIR.IR.DenseElementsAttribute(mlir_type(lhs), rhs)
    return TracedRArray{ElType,Shape,N}(nothing, MLIR.IR.stablehlo.constant(attr))
end

# for (jlop, hloop) in ((:(Base.:+), :add), (:(.+), :add), (:(Base.:-), :subtract), (:(.-), :subtract), (:(.*), :multiply), (:(./), :divide))
for (jlop, hloop) in ((:(Base.:+), :add), (:(Base.:-), :subtract))
        @eval begin
        function $jlop(lhs::TracedRArray{ElType,Shape,N}, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
        end

        function $jlop(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
        end

        function $jlop(lhs, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
        end
    end
end

function Base.:*(lhs::TracedRArray{ElType,Shape,2}, rhs::TracedRArray{ElType,Shape2,2}) where {ElType,Shape, Shape2}
    lhsty = MLIR.IR.type(lhs.mlir_data)
    rhsty = MLIR.IR.type(rhs.mlir_data)
    resty = MLIR.IR.TensorType((Base.size(lhsty)[1], Base.size(rhsty)[2]), eltype(lhsty))
    dot_dimension_numbers = MLIR.API.stablehloDotDimensionNumbersGet(MLIR.IR.context(), 0, [], 0, [], 1, [1], 1, [0])
    prec = MLIR.IR.Attribute(MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), "DEFAULT"))
    precar = MLIR.IR.Attribute([prec, prec])
    res = MLIR.IR.result(MLIR.Dialects.stablehlo.dot_general(lhs.mlir_data, rhs.mlir_data; result_0=resty, dot_dimension_numbers=dot_dimension_numbers, precision_config=precar), 1)
    return TracedRArray{ElType,(Base.size(lhsty)[1], Base.size(rhsty)[2]),2}((),  res)
end

for (jlop, hloop) in ((:(Base.sin), :sine), (:(Base.cos), :cosine), (:(Base.tanh), :tanh), (:(Base.FastMath.tanh_fast), :tanh), (:(Base.exp), :exp), (:(Base.log), :log), (:(Base.sqrt), :sqrt))
    @eval begin
        function $jlop(lhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}((), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1))
        end
    end
end


for (jlop, hloop) in ((:(Base.:+), :add), (:(Base.:-), :subtract), (:(Base.:*), :multiply), (:(Base.:/), :divide))
    @eval begin
    function elem_apply(::typeof($jlop), lhs::TracedRArray{ElType,Shape,N}, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
        return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
    end

    function elem_apply(::typeof($jlop), lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
        rhs = promote_to(lhs, rhs)
        return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
    end

    function elem_apply(::typeof($jlop), lhs, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
        lhs = promote_to(rhs, lhs)
        return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
    end
end
end

for (jlop, hloop) in ((:(Base.sin), :sine), (:(Base.cos), :cosine), (:(Base.tanh), :tanh), (:(Base.exp), :exp), (:(Base.log), :log), (:(Base.sqrt), :sqrt))
    @eval begin
        function elem_apply(::typeof($jlop), lhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}((), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1))
        end
    end
end



@inline function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    reshape(A, Base._reshape_uncolon(A, dims))
end

@inline function Base.reshape(A::ConcreteRArray{T, Shape, N}, dims::NTuple{NT, Int}) where {T, Shape, N, NT}
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    host = convert(Array{T, N}, A)
    # HLO reshape semantics collapse the opposite so enforce on Julia Side
    # until we later make the transpose/reshape/transpose
    host = reshape(host, dims)
    client = XLA.client(A.data)
    device = XLA.device(A.data)
    ConcreteRArray{T, dims, NT}(XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, data, device), nothing))
end

Base.copy(A::TracedRArray{T, Shape, N}) where {T, Shape, N} = TracedRArray((), A.mlir_data)


@inline function Base.permutedims(A::TracedRArray{T, Shape, N}, perm) where {T, Shape, N}
    TracedArray{T, tuple(Shape[i] for i in perm), N}(
        (),
        MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(A.mlir_data, DenseArrayAttribute([Int32(i-1) for i in perm])), 1)
    )
end

@inline function Base.reshape(A::TracedRArray{T, Shape, N}, dims::NTuple{NT, Int}) where {T, Shape, N, NT}
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    
    # HLO reshape semantics collapse the opposite way
    permutedims()

    res1 = MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(A.mlir_data, DenseArrayAttribute([Int32(N-1-i) for i in 0:N-1])), 1)

    res2 = MLIR.IR.result(MLIR.Dialects.stablehlo.reshape(res1, DenseArrayAttribute([Int32(i-1) for i in reverse(dims)])), 1)

    res3 = MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(res2, DenseArrayAttribute([Int32(NT-1-i) for i in 0:NT-1])), 1)

    return TracedRArray{T, dims, NT}((), res3)
end


using Base.Broadcast

using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

struct AbstractReactantArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
AbstractReactantArrayStyle(::Val{N}) where {N} = AbstractReactantArrayStyle{N}()
AbstractReactantArrayStyle{M}(::Val{N}) where {N,M} = AbstractReactantArrayStyle{N}()


# @inline function Broadcast.materialize(bc::Broadcasted) 
#    @show bc
#    inst = instantiate(bc)
#    @show inst
#    copy(inst)
# end

BroadcastStyle(::Type{T}) where {T<:TracedRArray} = AbstractReactantArrayStyle{ndims(T)}()

@inline function Base.similar(bc::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{T}, dims) where {T,N}
    @assert N isa Int
    TracedRArray{T, map(length, dims), N}((), nothing)
end

function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    dest = copyto!(similar(bc, ElType), bc)
    return dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

@inline Base.eltype(b::Broadcast.Extruded{T}) where T = eltype(T)

# we need to override the outer copy method to make sure we never fall back to scalar
# iteration (see, e.g., CUDA.jl#145)
@inline function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if  ElType == Any
        a1 = bc.args[1]
        @show a1
        b1 = a1.args[1]
        @show b1
        @show typeof(b1)
        @show eltype(b1)
        @show Broadcast._broadcast_getindex_eltype(a1.args[1])
        @show Broadcast.eltypes(a1.args)
        @show Broadcast._broadcast_getindex_eltype(a1)
        @show typeof(bc.args)
        argT = Broadcast.eltypes(bc.args)
        @show argT
        RT = Base._return_type(bc.f, argT)
        @show RT
        T = Base.promote_typejoin_union(RT)
        @show T
        @show bc.f, bc.args
    end
    @assert ElType != Any
    sim = similar(bc, ElType)
    @show ElType sim
    copyto!(sim, bc)
end

@inline function Base.materialize!(::Style, dest, bc::Broadcasted) where {Style<:AbstractReactantArrayStyle}
    return _copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end

@inline Base.copyto!(dest::TracedRArray, bc::Broadcasted{Nothing}) =
    _copyto!(dest, bc) # Keep it for ArrayConflict

@inline function broadcast_to_size(arg::AbstractArray, rsize)
    attr = MLIR.IR.DenseElementsAttribute(arg)
    len = ndims(arg)
    @assert typeof(len) == Int
    arg = TracedRArray{eltype(arg),size(arg),len}((), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(value=attr), 1))
    return arg
end

@inline function broadcast_to_size(arg::TracedRArray, rsize)
    return arg
end

@inline function broadcast_to_size(arg::Broadcast.Extruded, rsize)
    newidx = Broadcast.newindex(CartesianIndex(Base.OneTo(length(rsize))...), arg.keeps, arg.defaults)
    rsize2 = (rsize[i] for i in Tuple(newidx))
    x = broadcast_to_size(arg.x, rsize2)

    if size(x) == rsize
        return x
    end

    dims = [Int32(i) for i in Tuple(newidx)]
    mlirty = MLIR.IR.type(x.mlir_data)

    len = length(rsize)
    @assert typeof(len) == Int
    return TracedRArray{eltype(x),rsize,len}((), MLIR.IR.result(MLIR.Dialects.stablehlo.broadcast_in_dim(x.mlir_data; result_0=MLIR.IR.TensorType([t for t in rsize],eltype(mlirty)), broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims)), 1))
end

@inline function _copyto!(dest::TracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    @show bc.f
    @show bc.args
    bc = Broadcast.preprocess(dest, bc)

    @show bc.f
    @show bc.args

    args = (broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    @show args
    res = elem_apply(bc.f, args...)
    dest.mlir_data = res.mlir_data
    return dest
end