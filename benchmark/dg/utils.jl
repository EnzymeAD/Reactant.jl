struct Unkown end
struct PartialKnownDims{T, N, Dims, A} <: AbstractArray{T, N}
    data::A
    PartialKnownDims{Dims}(A) where Dims = new{eltype(A), ndims(A), Dims, typeof(A)}(A)
end
function Base.size(a::PartialKnownDims{T, N, Dims, A}) where {T, N, Dims, A}
    dims = (Dims.parameters...,)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        st_dim = dims[i]
        if st_dim isa Unkown
            return size(a.data, i)
        else
            return st_dim
        end
    end
end
Base.eltype(a::PartialKnownDims) = eltype(a.data)
Base.IndexStyle(::Type{<:PartialKnownDims}) = IndexLinear()

Base.Experimental.Const(a::PartialKnownDims{T, N,Dims}) where {T, N, Dims} = PartialKnownDims{Dims}(Base.Experimental.Const(a.data))
Base.@propagate_inbounds Base.getindex(a::PartialKnownDims, i) = Base.getindex(a.data, i)
Base.@propagate_inbounds Base.setindex!(a::PartialKnownDims, val, i) = Base.setindex!(a.data, val, i)

function loopinfo(name, expr, nodes...)
    if expr.head != :for
        error("Syntax error: pragma $name needs a for loop")
    end
    push!(expr.args[2].args, Expr(:loopinfo, nodes...))
    return expr
end

if parse(Bool, get(ENV, "UNROLLING", "true"))
    macro unroll(expr)
        expr = loopinfo("@unroll", expr, (Symbol("llvm.loop.unroll.full"),))
        return esc(expr)
    end
else
    macro unroll(expr)
        return esc(expr)
    end
end


# Poor man's Cassette
# Needed to get FMA
for (jlf, f) in zip((:+, :*, :-), (:add, :mul, :sub))
    for (T, llvmT) in ((Float32, "float"), (Float64, "double"))
        ir = """
            %x = f$f contract nsz $llvmT %0, %1
            ret $llvmT %x
        """
        @eval begin
            # the @pure is necessary so that we can constant propagate.
            Base.@pure function $jlf(a::$T, b::$T)
                Base.@_inline_meta
                Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
            end
        end
    end
    @eval function $jlf(args...)
        Base.$jlf(args...)
    end
end

let (jlf, f) = (:div_arcp, :div)
    for (T, llvmT) in ((Float32, "float"), (Float64, "double"))
        ir = """
            %x = f$f fast $llvmT %0, %1
            ret $llvmT %x
        """
        @eval begin
            # the @pure is necessary so that we can constant propagate.
            Base.@pure function $jlf(a::$T, b::$T)
                @Base._inline_meta
                Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
            end
        end
    end
    @eval function $jlf(args...)
        Base.$jlf(args...)
    end
end
rcp(x) = div_arcp(one(x), x) # still leads to rcp.rn which is also a function call
