module CuTracedOverloads

using Reactant: CuTracedArray, CuTracedRNumber

# TODO: arrays as allocated by the CUDA APIs are 256-byte aligned. we should keep track of
#       this information, because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@generated function alignment(::CuTracedArray{T}) where {T}
    if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
end

@generated function alignment(::CuTracedRNumber{T}) where {T}
    if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
end

Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(a::Type{<:CuTracedRNumber{T}}),
    @nospecialize(b::Type{<:CuTracedRNumber{T2}})
) where {T,T2}
    return promote_rule(T, T2)
end
Base.@nospecializeinfer function Base.promote_rule(
    ::Type{Any}, @nospecialize(b::Type{<:CuTracedRNumber})
)
    return Any
end
Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(a::Type{<:CuTracedRNumber}), ::Type{Any}
)
    return Any
end
Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(T2::Type), @nospecialize(b::Type{<:CuTracedRNumber{T}})
) where {T}
    if T == T2
        return T
    else
        return promote_rule(T, T2)
    end
end
Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(a::Type{<:CuTracedRNumber{T}}), @nospecialize(T2::Type)
) where {T}
    if T == T2
        return T
    else
        return promote_rule(T, T2)
    end
end

@inline function Base.convert(CT::Type{CuTracedRNumber{Float64,1}}, x::Number)
    return CT(
        Base.reinterpret(
            Core.LLVMPtr{Float64,1},
            Base.llvmcall(
                (
                    """define i8 addrspace(1)* @entry(double %d) alwaysinline {
              %a = alloca double
              store atomic double %d, double* %a release, align 8
       %bc = bitcast double* %a to i8*
              %ac = addrspacecast i8* %bc to i8 addrspace(1)*
              ret i8 addrspace(1)* %ac
                        }
          """,
                    "entry",
                ),
                Core.LLVMPtr{UInt8,1},
                Tuple{Float64},
                convert(Float64, x),
            ),
        ),
    )
end

@inline function Base.convert(CT::Type{CuTracedRNumber{Float32,1}}, x::Number)
    return CT(
        Base.reinterpret(
            Core.LLVMPtr{Float32,1},
            Base.llvmcall(
                (
                    """define i8 addrspace(1)* @entry(float %d) alwaysinline {
              %a = alloca float
              store atomic float %d, float* %a release, align 4
       %bc = bitcast float* %a to i8*
              %ac = addrspacecast i8* %bc to i8 addrspace(1)*
              ret i8 addrspace(1)* %ac
                        }
          """,
                    "entry",
                ),
                Core.LLVMPtr{UInt8,1},
                Tuple{Float32},
                convert(Float32, x),
            ),
        ),
    )
end

Base.convert(::Type{<:CuTracedRNumber{T}}, x::CuTracedRNumber{T}) where {T} = x

## array interface

Base.elsize(::Type{<:CuTracedArray{T}}) where {T} = sizeof(T)
Base.size(::CuTracedArray{T,N,A,Size}) where {T,N,A,Size} = Size
Base.sizeof(x::CuTracedArray) = Base.elsize(x) * length(x)
function Base.pointer(x::CuTracedArray{T,<:Any,A}) where {T,A}
    return Base.unsafe_convert(Core.LLVMPtr{T,A}, x)
end
@inline function Base.pointer(x::CuTracedArray{T,<:Any,A}, i::Integer) where {T,A}
    return Base.unsafe_convert(Core.LLVMPtr{T,A}, x) + Base._memory_offset(x, i)
end

## conversions

function Base.unsafe_convert(
    ::Type{Core.LLVMPtr{T,A}}, x::CuTracedArray{T,<:Any,A}
) where {T,A}
    return x.ptr
end

## indexing

Base.IndexStyle(::Type{<:CuTracedArray}) = Base.IndexLinear()

function Base.getindex(RN::CuTracedRNumber{T,A}) where {T,A}
    align = alignment(RN)
    return @inbounds unsafe_load(RN.ptr, 1, Val(align))
end

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::CuTracedArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
Base.@propagate_inbounds Base.getindex(
    A::CuTracedArray, I::Union{Integer,CartesianIndex}...
) = A[Base._to_linear_index(A, to_indices(A, I)...)]
Base.@propagate_inbounds Base.setindex!(
    A::CuTracedArray, x, I::Union{Integer,CartesianIndex}...
) = A[Base._to_linear_index(A, to_indices(A, I)...)] = x

@inline function Base.iterate(A::CuTracedArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end

## ops
for jlop in (
    :(Base.min),
    :(Base.mod),
    :(Base.max),
    :(Base.:+),
    :(Base.:-),
    :(Base.:*),
    :(Base.:/),
    :(Base.:^),
    :(Base.rem),
    :(Base.isless),
    :(Base.:(==)),
    :(Base.:(!=)),
)
    @eval begin
        @inline $jlop(a::CuTracedRNumber, b::CuTracedRNumber) = $jlop(a[], b[])
        @inline $jlop(a::CuTracedRNumber{T,A}, b::Number) where {T,A} = $jlop(a[], b)
        @inline $jlop(a::Number, b::CuTracedRNumber{T,A}) where {T,A} = $jlop(a, b[])
    end
end

@inline Base.ifelse(cond::Bool, a, b::CuTracedRNumber) = ifelse(cond, a, b[])
@inline Base.ifelse(cond::Bool, a::CuTracedRNumber, b) = ifelse(cond, a[], b)
@inline Base.ifelse(cond::Bool, a::CuTracedRNumber, b::CuTracedRNumber) =
    ifelse(cond, a[], b[])
@inline Base.ifelse(cond::CuTracedRNumber, a, b) = ifelse(cond[], a, b)
@inline Base.ifelse(cond::CuTracedRNumber, a::CuTracedRNumber, b) = ifelse(cond[], a[], b)
@inline Base.ifelse(cond::CuTracedRNumber, a, b::CuTracedRNumber) = ifelse(cond[], a, b[])
@inline Base.ifelse(cond::CuTracedRNumber, a::CuTracedRNumber, b::CuTracedRNumber) =
    ifelse(cond[], a[], b[])

Base.@constprop :aggressive @inline Base.:^(
    a::CuTracedRNumber{T,A}, b::Integer
) where {T,A} = ^(a[], b)

@inline Base.unsafe_trunc(::Type{T}, a::CuTracedRNumber) where {T} =
    Base.unsafe_trunc(T, a[])

for jlop in (:(Base.:+), :(Base.:-), :(Base.isnan), :(Base.isfinite), :(Base.isinf))
    @eval begin
        @inline $jlop(a::CuTracedRNumber) = $jlop(a[])
    end
end

Base.OneTo(x::CuTracedRNumber{<:Integer}) = Base.OneTo(x[])

@static if isdefined(Base, :unchecked_oneto)
    function Base.unchecked_oneto(x::CuTracedRNumber{<:Integer})
        return Base.unchecked_oneto(x[])
    end
end

Base.one(a::CuTracedRNumber) = one(a[])
Base.one(::Type{<:CuTracedRNumber{T,A}}) where {T,A} = one(T)
Base.zero(a::CuTracedRNumber) = zero(a[])
Base.zero(::Type{<:CuTracedRNumber{T,A}}) where {T,A} = zero(T)

end
