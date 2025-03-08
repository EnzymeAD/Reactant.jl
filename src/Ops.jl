# This module reflects the HLO ops defined in the openxla/stablehlo repo (plus some extras).
# If you want to add some check or test, the StableHLO spec should be taken as the source of truth, not the Julia or Reactant semantics.
# Julia and Reactant semantics should be considered on the higher abstractions that use these ops.
module Ops
using ..MLIR: MLIR
using ..MLIR.Dialects: stablehlo, chlo, enzyme
using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    RArray,
    RNumber,
    MissingTracedValue,
    unwrapped_eltype
using ReactantCore: ReactantCore
using Functors: fmap

function mlir_type(x::Union{RNumber,RArray})
    return MLIR.IR.TensorType(size(x), MLIR.IR.Type(unwrapped_eltype(x)))
end

mlir_type(::MissingTracedValue) = MLIR.IR.TensorType((), MLIR.IR.Type(Bool))

function mlir_type(RT::Type{<:RArray{T,N}}, shape) where {T,N}
    @assert length(shape) == N
    return MLIR.IR.TensorType(shape, MLIR.IR.Type(unwrapped_eltype(RT)))
end

function mlir_type(RT::Type{<:RNumber})
    return MLIR.IR.TensorType((), MLIR.IR.Type(unwrapped_eltype(RT)))
end

function mlir_type(::Type{<:MissingTracedValue})
    return MLIR.IR.TensorType((), MLIR.IR.Type(Bool))
end

const DEBUG_MODE::Ref{Bool} = Ref(false)

function with_debug(f)
    old = DEBUG_MODE[]
    DEBUG_MODE[] = true
    try
        return f()
    finally
        DEBUG_MODE[] = old
    end
end

@noinline function mlir_stacktrace(name, file, line)::MLIR.IR.Location
    # calling `stacktrace` can add a lot of time overhead, so let's avoid adding debug info if not used
    if !DEBUG_MODE[]
        return MLIR.IR.Location(name, MLIR.IR.Location(file, line, 0))
    end

    # retrieve current stacktrace, remove this function's frame and translate to MLIR Location
    st = stacktrace()
    deleteat!(st, 1)
    return mapfoldl(MLIR.IR.Location, st) do stackframe
        name = string(stackframe.func)
        file = stackframe.file
        line = stackframe.line
        return MLIR.IR.Location(name, MLIR.IR.Location(file, line, 0))
    end
end

struct Token
    mlir_data::MLIR.IR.Value
end

# constant ops
@noinline function constant(
    x::DenseArray{T,N}; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T,N}
    value = MLIR.IR.DenseElementsAttribute(x)
    output = mlir_type(TracedRArray{T,N}, size(x))
    res = MLIR.IR.result(stablehlo.constant(; output, value, location))
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function constant(
    x::T; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T<:Number}
    x isa TracedRNumber && return x
    res = fill(x; location)
    return TracedRNumber{T}((), res.mlir_data)
end

function fill(
    v, dims::Base.DimOrInd...; location=mlir_stacktrace("fill", @__FILE__, @__LINE__)
)
    return fill(v, dims; location)
end
function fill(
    v,
    dims::NTuple{N,Union{Integer,Base.OneTo}};
    location=mlir_stacktrace("fill", @__FILE__, @__LINE__),
) where {N}
    return fill(v, map(Base.to_dim, dims); location)
end
function fill(
    v, dims::NTuple{N,Integer}; location=mlir_stacktrace("fill", @__FILE__, @__LINE__)
) where {N}
    return fill(v, collect(dims); location)
end
function fill(v, ::Tuple{}; location=mlir_stacktrace("fill", @__FILE__, @__LINE__))
    return fill(v, Int[]; location)
end

function fill(number::TracedRNumber{T}, shape::Vector{Int}; location) where {T}
    return Base.fill(number, Tuple(shape))
end

for (T, mlir_func) in (
    (Bool, :mlirDenseElementsAttrBoolSplatGet),
    (UInt8, :mlirDenseElementsAttrUInt8SplatGet),
    (Int8, :mlirDenseElementsAttrInt8SplatGet),
    (UInt32, :mlirDenseElementsAttrUInt32SplatGet),
    (Int32, :mlirDenseElementsAttrInt32SplatGet),
    (UInt64, :mlirDenseElementsAttrUInt64SplatGet),
    (Int64, :mlirDenseElementsAttrInt64SplatGet),
    (Float32, :mlirDenseElementsAttrFloatSplatGet),
    (Float64, :mlirDenseElementsAttrDoubleSplatGet),
)
    @eval begin
        @noinline function fill(
            number::$T,
            shape::Vector{Int};
            location=mlir_stacktrace("fill", @__FILE__, @__LINE__),
        )
            tt = MLIR.IR.TensorType(shape, MLIR.IR.Type($T); location=location)

            splatattr = MLIR.API.$mlir_func(tt, number)
            cst_op = stablehlo.constant(; output=tt, value=splatattr, location=location)
            cst = MLIR.IR.result(cst_op)
            ta = TracedRArray{$T,length(shape)}((), cst, shape)
            return ta
        end
    end
end

_fill_element_attr(x) = MLIR.IR.Attribute(x)
function _fill_element_attr(x::Complex)
    return MLIR.IR.Attribute([
        MLIR.IR.Attribute(Base.real(x)), MLIR.IR.Attribute(Base.imag(x))
    ])
end

@noinline function fill(
    element::T, shape::Vector{Int}; location=mlir_stacktrace("fill", @__FILE__, @__LINE__)
) where {T}
    tt = MLIR.IR.TensorType(shape, MLIR.IR.Type(T))
    splatattr = MLIR.API.mlirDenseElementsAttrSplatGet(tt, _fill_element_attr(element))
    cst_op = stablehlo.constant(; output=tt, value=splatattr, location=location)
    cst = MLIR.IR.result(cst_op)
    ta = TracedRArray{T,length(shape)}((), cst, shape)
    return ta
end

# unary elementwise ops
for (dialect, op) in [
    (:stablehlo, :abs),
    (:stablehlo, :cbrt),
    (:stablehlo, :ceil),
    (:stablehlo, :count_leading_zeros),
    (:stablehlo, :cosine),
    (:stablehlo, :exponential),
    (:stablehlo, :exponential_minus_one),
    (:stablehlo, :floor),
    (:stablehlo, :log),
    (:stablehlo, :log_plus_one),
    (:stablehlo, :logistic),
    (:stablehlo, :negate),
    (:stablehlo, :not),
    (:stablehlo, :popcnt),
    (:stablehlo, :round_nearest_afz),
    (:stablehlo, :round_nearest_even),
    (:stablehlo, :rsqrt),
    (:stablehlo, :sign),
    (:stablehlo, :sine),
    (:stablehlo, :sqrt),
    (:stablehlo, :tan),
    (:stablehlo, :tanh),
    (:chlo, :acos),
    (:chlo, :acosh),
    (:chlo, :asin),
    (:chlo, :asinh),
    (:chlo, :atan),
    (:chlo, :atanh),
    (:chlo, :bessel_i1e),
    (:chlo, :conj),
    (:chlo, :cosh),
    (:chlo, :digamma),
    (:chlo, :erf_inv),
    (:chlo, :erf),
    (:chlo, :erfc),
    (:chlo, :lgamma),
    (:chlo, :sinh),
]
    @eval begin
        @noinline function $op(
            x::TracedRArray{T,N};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T,N}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location
                ),
            )
            return TracedRArray{T,N}((), res, size(x))
        end

        @noinline function $op(
            x::TracedRNumber{T};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location
                ),
            )
            return TracedRNumber{T}((), res)
        end
    end
end

# binary elementwise ops
for (dialect, op) in [
    (:stablehlo, :add),
    (:stablehlo, :and),
    (:stablehlo, :atan2),
    (:stablehlo, :divide),
    (:stablehlo, :maximum),
    (:stablehlo, :minimum),
    (:stablehlo, :multiply),
    (:stablehlo, :or),
    (:stablehlo, :power),
    (:stablehlo, :remainder),
    (:stablehlo, :shift_left),
    (:stablehlo, :shift_right_arithmetic),
    (:stablehlo, :shift_right_logical),
    (:stablehlo, :subtract),
    (:stablehlo, :xor),
    (:chlo, :next_after),
    (:chlo, :polygamma),
    (:chlo, :zeta),
]
    @eval begin
        @noinline function $op(
            a::TracedRArray{T,N},
            b::TracedRArray{T,N};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T,N}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    a.mlir_data,
                    b.mlir_data;
                    result=mlir_type(TracedRArray{T,N}, size(a)),
                    location,
                ),
            )
            return TracedRArray{T,N}((), res, size(a))
        end

        @noinline function $op(
            a::TracedRNumber{T},
            b::TracedRNumber{T};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    a.mlir_data,
                    b.mlir_data;
                    result=mlir_type(TracedRArray{T,0}, ()),
                    location,
                ),
            )
            return TracedRNumber{T}((), res)
        end
    end
end

# is* checks
for (dialect, op) in
    [(:stablehlo, :is_finite), (:chlo, :is_inf), (:chlo, :is_neg_inf), (:chlo, :is_pos_inf)]
    result = dialect == :stablehlo ? :y : :result
    @eval begin
        @noinline function $op(
            x::TracedRArray{T,N};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T,N}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    x.mlir_data;
                    $(result)=mlir_type(TracedRArray{Bool,N}, size(x)),
                    location,
                ),
            )
            return TracedRArray{Bool,N}((), res, size(x))
        end

        @noinline function $op(
            x::TracedRNumber{T};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    x.mlir_data; $(result)=mlir_type(TracedRArray{Bool,0}, ()), location
                ),
            )
            return TracedRNumber{Bool}((), res)
        end
    end
end

# fixes to default automated implementations
@noinline function abs(
    x::TracedRArray{Complex{T},N}; location=mlir_stacktrace("abs", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function abs(
    x::TracedRNumber{Complex{T}}; location=mlir_stacktrace("abs", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

# shape ops
function reshape(x::TracedRArray, dims...; kwargs...)
    return reshape(x, collect(dims); kwargs...)
end

@noinline function reshape(
    x::TracedRArray{T,N},
    dims::Vector{Int};
    location=mlir_stacktrace("reshape", @__FILE__, @__LINE__),
) where {T,N}
    # HLO reshape semantics collapse the opposite way
    res1 = transpose(x, Int64[N:-1:1...])
    restype = mlir_type(TracedRArray{T,length(dims)}, collect(Base.reverse(dims)))
    res = MLIR.IR.result(stablehlo.reshape(res1.mlir_data; result_0=restype, location))
    result = TracedRArray{T,length(dims)}((), res, collect(Base.reverse(dims)))
    # NOTE this last `transpose` is required for consistency with Julia's column-major order
    # do not remove, as it will be optimized away by the compiler
    return transpose(result, Int64[length(dims):-1:1...])
end

@noinline function get_dimension_size(
    x::TracedRArray{T,N},
    dim;
    location=mlir_stacktrace("get_dimension_size", @__FILE__, @__LINE__),
) where {T,N}
    dimension = MLIR.IR.Attribute(dim - 1)
    res = MLIR.IR.result(
        stablehlo.get_dimension_size(
            x.mlir_data; result_0=mlir_type(TracedRArray{Int32,0}, ()), dimension, location
        ),
    )
    return TracedRNumber{Int32}((), res)
end

@noinline function set_dimension_size(
    x::TracedRArray{T,N},
    size::TracedRNumber{Int},
    dim::Int;
    location=mlir_stacktrace("set_dimension_size", @__FILE__, @__LINE__),
) where {T,N}
    dimension = MLIR.IR.Attribute(dim - 1)
    res = MLIR.IR.result(
        stablehlo.set_dimension_size(
            x.mlir_data,
            size.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(x)),
            dimension,
            location,
        ),
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function transpose(
    x::TracedRArray{T,N},
    permutation;
    location=mlir_stacktrace("transpose", @__FILE__, @__LINE__),
) where {T,N}
    rsize = permute!(collect(size(x)), permutation)
    permutation = permutation .- 1
    result = mlir_type(TracedRArray{T,N}, rsize)
    permutation = MLIR.IR.DenseArrayAttribute(permutation)
    res = MLIR.IR.result(stablehlo.transpose(x.mlir_data; result, permutation, location))
    return TracedRArray{T,N}((), res, rsize)
end

# indexing ops
@noinline function pad(
    x::TracedRArray{T,N},
    padding_value::TracedRNumber{T};
    low=Base.fill(0, N),
    high=Base.fill(0, N),
    interior=Base.fill(0, N),
    location=mlir_stacktrace("pad", @__FILE__, @__LINE__),
) where {T,N}
    rsize = size(x) .+ low .+ high .+ max.(size(x) .- 1, 0) .* interior
    res = MLIR.IR.result(
        stablehlo.pad(
            x.mlir_data,
            padding_value.mlir_data;
            edge_padding_low=MLIR.IR.DenseArrayAttribute(low),
            edge_padding_high=MLIR.IR.DenseArrayAttribute(high),
            interior_padding=MLIR.IR.DenseArrayAttribute(interior),
            location,
        ),
    )
    return TracedRArray{T,N}((), res, rsize)
end

@noinline function slice(
    x::TracedRArray{T,N},
    start_indices,
    limit_indices;
    strides=nothing,
    location=mlir_stacktrace("slice", @__FILE__, @__LINE__),
) where {T,N}
    start_indices = start_indices .- 1
    limit_indices = limit_indices
    rsize = limit_indices .- start_indices
    @assert all(rsize .> 0) "Invalid slice dimensions"
    strides = isnothing(strides) ? [1, size(x)[1:(end - 1)]...] : strides
    res = MLIR.IR.result(
        stablehlo.slice(
            x.mlir_data;
            result_0=mlir_type(TracedRArray{T,N}, rsize),
            start_indices=MLIR.IR.DenseArrayAttribute(start_indices),
            limit_indices=MLIR.IR.DenseArrayAttribute(limit_indices),
            strides=MLIR.IR.DenseArrayAttribute(strides),
            location,
        ),
    )
    return TracedRArray{T,N}((), res, rsize)
end

# numerics
@noinline function complex(
    real::TracedRArray{T,N},
    imag::TracedRArray{T,N};
    location=mlir_stacktrace("complex", @__FILE__, @__LINE__),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.complex(
            real.mlir_data,
            imag.mlir_data;
            result=mlir_type(TracedRArray{Complex{T},N}, size(real)),
            location,
        ),
    )
    return TracedRArray{Complex{T},N}((), res, size(real))
end

@noinline function complex(
    real::TracedRNumber{T},
    imag::TracedRNumber{T};
    location=mlir_stacktrace("complex", @__FILE__, @__LINE__),
) where {T}
    res = MLIR.IR.result(
        stablehlo.complex(
            real.mlir_data,
            imag.mlir_data;
            result=mlir_type(TracedRArray{Complex{T},0}, ()),
            location,
        ),
    )
    return TracedRNumber{Complex{T}}((), res)
end

@noinline function real(
    x::TracedRArray{Complex{T},N}; location=mlir_stacktrace("real", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.real(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function real(
    x::TracedRNumber{Complex{T}}; location=mlir_stacktrace("real", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.real(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

@noinline function imag(
    x::TracedRArray{Complex{T},N}; location=mlir_stacktrace("imag", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.imag(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function imag(
    x::TracedRNumber{Complex{T}}; location=mlir_stacktrace("imag", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.imag(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

# function bitcast_convert(
#     ::Type{TracedRArray{U,N}},
#     x::TracedRArray{T,N};
#     location=mlir_stacktrace(
#         "bitcast_convert", @__FILE__, @__LINE__
#     ),
# ) where {T,N}
#     res = MLIR.IR.result(
#         stablehlo.bitcast_convert(
#             x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location
#         ),
#     )
#     return TracedRArray{T,N}((), res, size(x))
# end
@noinline function bitcast_convert(
    ::Type{U},
    x::TracedRNumber{T};
    location=mlir_stacktrace("bitcast_convert", @__FILE__, @__LINE__),
) where {T,U}
    res = MLIR.IR.result(
        stablehlo.bitcast_convert(
            x.mlir_data; result_0=mlir_type(TracedRArray{U,0}, ()), location
        ),
    )
    return TracedRNumber{U}((), res)
end

@noinline function fft(
    x::TracedRArray{T,N};
    type::String,
    length,
    location=mlir_stacktrace("fft", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 <= Base.length(length) <= 3 "fft only supports up to rank 3"

    if type ∈ ("FFT", "IFFT")
        @assert T <: Complex
        Tout = T
        rsize = size(x)
    elseif type == "RFFT"
        @assert T <: Real
        Tout = Complex{T}
        rsize = let rsize = collect(size(x))
            rsize[end] = rsize[end] == 0 ? 0 : rsize[end] ÷ 2 + 1
            Tuple(rsize)
        end
    elseif type == "IRFFT"
        @assert T <: Complex
        Tout = Base.real(T)
        rsize = let rsize = collect(size(x))
            rsize[(end - Base.length(length) + 1):end] = length
            Tuple(rsize)
        end
    else
        error("Invalid FFT type: $type")
    end

    res = MLIR.IR.result(
        stablehlo.fft(
            x.mlir_data;
            result_0=mlir_type(TracedRArray{Tout,N}, rsize),
            fft_type=MLIR.API.stablehloFftTypeAttrGet(MLIR.IR.context(), type),
            fft_length=MLIR.IR.DenseArrayAttribute(length),
            location,
        ),
    )
    return TracedRArray{Tout,N}((), res, rsize)
end

@noinline function cholesky(
    x::TracedRArray{T,N};
    lower::Bool=false,
    location=mlir_stacktrace("cholesky", @__FILE__, @__LINE__),
) where {T,N}
    lower = MLIR.IR.Attribute(lower)
    res = MLIR.IR.result(
        stablehlo.cholesky(
            x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), lower, location
        ),
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function clamp(
    min::Union{TracedRNumber{T},TracedRArray{T,N}},
    x::TracedRArray{T,N},
    max::Union{TracedRNumber{T},TracedRArray{T,N}};
    location=mlir_stacktrace("clamp", @__FILE__, @__LINE__),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.clamp(
            min.mlir_data,
            x.mlir_data,
            max.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(x)),
            location,
        ),
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function clamp(
    min::TracedRNumber{T},
    x::TracedRNumber{T},
    max::TracedRNumber{T};
    location=mlir_stacktrace("clamp", @__FILE__, @__LINE__),
) where {T}
    res = MLIR.IR.result(
        stablehlo.clamp(
            min.mlir_data,
            x.mlir_data,
            max.mlir_data;
            result=mlir_type(TracedRArray{T,0}, ()),
            location,
        ),
    )
    return TracedRNumber{T}((), res)
end

@noinline function clamp(
    min::T, x::Union{TracedRArray{T,N},TracedRNumber{T}}, max::T
) where {T,N}
    return clamp(constant(min), x, constant(max))
end

# function convolution(
#     lhs::TracedRArray{T,N},
#     rhs::TracedRArray{T,N};
#     dimension_numbers,
#     feature_group_count,
#     batch_group_count,
#     window_strides=nothing,
#     padding=nothing,
#     lhs_dilation=nothing,
#     rhs_dilation=nothing,
#     location=mlir_stacktrace(
#         "convolution", @__FILE__, @__LINE__
#     ),
# ) where {T,N}
#     res = MLIR.IR.result(
#         stablehlo.convolution(
#             lhs.mlir_data,
#             rhs.mlir_data;
#             result=mlir_type(TracedRArray{T,N}, ...), # TODO size of result
#             window_strides, #*MLIR.IR.DenseArrayAttribute(window_strides)*#,
#             padding, #*MLIR.IR.DenseArrayAttribute(padding)*#,
#             lhs_dilation, #*MLIR.IR.DenseArrayAttribute(lhs_dilation)*#,
#             rhs_dilation, #*MLIR.IR.DenseArrayAttribute(rhs_dilation)*#,
#             feature_group_count=feature_group_count,
#             location,
#         ),
#     )
#     return TracedRArray{T,N}((), res, size(lhs))
# end

@noinline function dot_general(
    lhs::TracedRArray{T},
    rhs::TracedRArray{T};
    contracting_dimensions,
    batching_dimensions=(Int[], Int[]),
    precision_config=nothing,
    precision_type=nothing,
    accumulation_type=nothing,
    component_count=nothing,
    num_primitive_operations=nothing,
    allow_imprecise_accumulation=nothing,
    location=mlir_stacktrace("dot_general", @__FILE__, @__LINE__),
) where {T}
    # C1 + C2
    @assert length(batching_dimensions) == 2 && splat(==)(length.(batching_dimensions))
    @assert length(contracting_dimensions) == 2 &&
        splat(==)(length.(contracting_dimensions))

    # C3 + C4
    @assert all(eltype.(contracting_dimensions) .<: Int64)
    @assert all(eltype.(batching_dimensions) .<: Int64)
    @assert all(isdisjoint.(contracting_dimensions, batching_dimensions))

    lhs_contracting_dimensions, rhs_contracting_dimensions = contracting_dimensions
    lhs_batching_dimensions, rhs_batching_dimensions = batching_dimensions

    # C5 + C6 + C7 + C8
    @assert all(lhs_batching_dimensions .<= ndims(lhs))
    @assert all(rhs_batching_dimensions .<= ndims(rhs))
    @assert all(lhs_contracting_dimensions .<= ndims(lhs))
    @assert all(rhs_contracting_dimensions .<= ndims(rhs))

    # C9 + C10
    @assert size.(Ref(lhs), lhs_batching_dimensions) ==
        size.(Ref(rhs), rhs_batching_dimensions)
    @assert size.(Ref(lhs), lhs_contracting_dimensions) ==
        size.(Ref(rhs), rhs_contracting_dimensions)

    # C11
    @assert isnothing(precision_config) || length(precision_config) == 2

    @assert isnothing(precision_type) ||
        length(precision_type) == 2 && eltype(precision_type) <: AbstractFloat
    @assert isnothing(accumulation_type) || accumulation_type <: AbstractFloat

    # C22 + C23
    @assert isnothing(component_count) ||
        length(component_count) == 2 &&
            eltype(component_count) <: Int32 &&
            all(0 .<= component_count)

    # C24
    @assert isnothing(num_primitive_operations) ||
        num_primitive_operations isa Int32 && num_primitive_operations > 0
    @assert isnothing(allow_imprecise_accumulation) || allow_imprecise_accumulation isa Bool

    ctx = MLIR.IR.context()

    # from C12
    lhs_result_dimensions = setdiff(
        1:ndims(lhs), lhs_batching_dimensions, lhs_contracting_dimensions
    )
    rhs_result_dimensions = setdiff(
        1:ndims(rhs), rhs_batching_dimensions, rhs_contracting_dimensions
    )

    ressize = vcat(
        size.(Ref(lhs), lhs_batching_dimensions),
        size.(Ref(lhs), lhs_result_dimensions),
        size.(Ref(rhs), rhs_result_dimensions),
    )

    # fix 1-indexing
    lhs_batching_dimensions = lhs_batching_dimensions .- 1
    rhs_batching_dimensions = rhs_batching_dimensions .- 1
    lhs_contracting_dimensions = lhs_contracting_dimensions .- 1
    rhs_contracting_dimensions = rhs_contracting_dimensions .- 1

    dot_dimension_numbers = GC.@preserve lhs_contracting_dimensions rhs_contracting_dimensions lhs_batching_dimensions rhs_batching_dimensions begin
        MLIR.IR.Attribute(
            MLIR.API.stablehloDotDimensionNumbersGet(
                ctx,
                length(lhs_batching_dimensions),
                lhs_batching_dimensions,
                length(rhs_batching_dimensions),
                rhs_batching_dimensions,
                length(lhs_contracting_dimensions),
                lhs_contracting_dimensions,
                length(rhs_contracting_dimensions),
                rhs_contracting_dimensions,
            ),
        )
    end

    if !isnothing(precision_config)
        precision_config = MLIR.IR.Attribute([
            MLIR.API.stablehloPrecisionAttrGet(ctx, precision_config[1]),
            MLIR.API.stablehloPrecisionAttrGet(ctx, precision_config[2]),
        ])
    end

    # all or nothing: if one is set, all must be set
    # TODO maybe be more flexible, by setting some defaults?
    if any(
        !isnothing,
        (
            precision_type,
            accumulation_type,
            component_count,
            num_primitive_operations,
            allow_imprecise_accumulation,
        ),
    )
        @assert all(
            !isnothing,
            (
                precision_type...,
                accumulation_type,
                component_count...,
                num_primitive_operations,
                allow_imprecise_accumulation,
            ),
        )
        lhs_precision_type, rhs_precision_type = precision_type
        lhs_component_count, rhs_component_count = component_count
        algorithm = GC.@preserve begin
            MLIR.IR.Attribute(
                MLIR.API.stablehloDotAlgorithmGet(
                    ctx,
                    lhs_precision_type,
                    rhs_precision_type,
                    accumulation_type,
                    lhs_component_count,
                    rhs_component_count,
                    num_primitive_operations,
                    allow_imprecise_accumulation,
                ),
            )
        end
    else
        algorithm = nothing
    end

    res = MLIR.IR.result(
        stablehlo.dot_general(
            lhs.mlir_data,
            rhs.mlir_data;
            result_0=mlir_type(TracedRArray{T,length(ressize)}, ressize),
            dot_dimension_numbers,
            precision_config,
            algorithm,
            location,
        ),
    )
    return TracedRArray{T,length(ressize)}((), res, ressize)
end

@noinline function einsum(
    lhs::TracedRArray{T},
    rhs::TracedRArray{T};
    equation::String,
    location=mlir_stacktrace("einsum", @__FILE__, @__LINE__),
) where {T}
    Base.depwarn(
        "`stablehlo.einsum` is on deprecation process; use `dot_general` instead", :einsum
    )
    ins, ic = split(equation, "->")
    ia, ib = split(ins, ",")

    sizea = Dict(c => d for (c, d) in zip(ia, size(lhs)))
    sizeb = Dict(c => d for (c, d) in zip(ib, size(rhs)))
    sizes = mergewith(sizea, sizeb) do da, db
        da == db ? da : error("Invalid dimensions in einsum equation")
    end

    rsize = Tuple(sizes[i] for i in ic)
    result_0 = mlir_type(TracedRArray{T,length(ic)}, rsize)

    res = MLIR.IR.result(
        stablehlo.einsum(
            lhs.mlir_data,
            rhs.mlir_data;
            result_0,
            einsum_config=MLIR.IR.Attribute(equation),
            location,
        ),
    )
    return TracedRArray{T,length(rsize)}((), res, rsize)
end

# function unary_einsum(
#     x::TracedRArray{T};
#     equation::String,
#     location=mlir_stacktrace(
#         "unary_einsum", @__FILE__, @__LINE__
#     ),
# ) where {T}
#     ia, ic = split(equation, "->")
#     sizes = Dict(c => d for (c, d) in zip(ia, size(x)))
#     rsize = Tuple(sizes[i] for i in ic)
#     result_0 = mlir_type(TracedRArray{T,length(ic)}, rsize)

#     res = MLIR.IR.result(
#         stablehlo.unary_einsum(
#             x.mlir_data; result_0, einsum_config=MLIR.IR.Attribute(equation), location
#         ),
#     )
#     if length(rsize) == 0
#         return TracedRNumber{T}((), res)
#     else
#         return TracedRArray{T,length(rsize)}((), res, rsize)
#     end
# end

# paralell ops
@noinline function partition_id(;
    location=mlir_stacktrace("partition_id", @__FILE__, @__LINE__)
)
    res = MLIR.IR.result(stablehlo.partition_id(; location))
    return TracedRNumber{UInt32}((), res)
end

@noinline function replica_id(;
    location=mlir_stacktrace("replica_id", @__FILE__, @__LINE__)
)
    res = MLIR.IR.result(stablehlo.replica_id(; location))
    return TracedRNumber{UInt32}((), res)
end

@noinline function after_all(
    tokens...; location=mlir_stacktrace("after_all", @__FILE__, @__LINE__)
)
    tokens = [token.mlir_data for token in tokens]
    res = MLIR.IR.result(stablehlo.after_all(tokens; location))
    return Token(res)
end

@noinline function optimization_barrier(
    operands::Union{TracedRNumber,TracedRArray}...;
    location=mlir_stacktrace("optimization_barrier", @__FILE__, @__LINE__),
)
    values = [operand.mlir_data for operand in operands]
    op = stablehlo.optimization_barrier(values; location)
    return Tuple(
        map(enumerate(operands)) do (i, operand)
            typ = typeof(operand)
            res = MLIR.IR.result(op, i)
            if typ <: TracedRArray
                return typ((), res, size(operand))
            elseif typ <: TracedRNumber
                return typ((), res)
            else
                error("Invalid type: $typ")
            end
        end,
    )
end

@noinline function outfeed(
    operands::Union{TracedRNumber,TracedRArray}...;
    token,
    config="",
    location=mlir_stacktrace("outfeed", @__FILE__, @__LINE__),
)
    values = [operand.mlir_data for operand in operands]
    outfeed_config = MLIR.IR.Attribute(config)
    res = MLIR.IR.result(
        stablehlo.outfeed(values, token.mlir_data; outfeed_config, location)
    )
    return Token(res)
end

@noinline function send(
    operands::Union{TracedRNumber,TracedRArray}...;
    token,
    channel_id::Int,
    channel_type::Int,
    is_host_transfer=nothing,
    location=mlir_stacktrace("send", @__FILE__, @__LINE__),
)
    values = [operand.mlir_data for operand in operands]
    channel_handle = MLIR.API.stablehloChannelHandleGet(
        MLIR.IR.context(), channel_id, channel_type
    )
    is_host_transfer = if isnothing(is_host_transfer)
        nothing
    else
        MLIR.IR.Attribute(is_host_transfer)
    end
    res = MLIR.IR.result(
        stablehlo.send(values, token.mlir_data; channel_handle, is_host_transfer, location)
    )
    return Token(res)
end

@noinline function recv(
    results::Tuple{Type,Vector{Int}}...;
    token,
    channel_id::Int,
    channel_type::Int,
    is_host_transfer=nothing,
    location=mlir_stacktrace("recv", @__FILE__, @__LINE__),
)
    channel_handle = MLIR.API.stablehloChannelHandleGet(
        MLIR.IR.context(), channel_id, channel_type
    )
    is_host_transfer = if isnothing(is_host_transfer)
        nothing
    else
        MLIR.IR.Attribute(is_host_transfer)
    end
    result_0 = map(results) do (typ, shape)
        MLIR.IR.TensorType(shape, mlir_type(typ))
    end
    op = stablehlo.recv(
        token.mlir_data; result_0, channel_handle, is_host_transfer, location
    )
    return tuple(
        map(enumerate(results)) do (i, (typ, shape))
            typ = MLIR.IR.TensorType(shape, mlir_type(typ))
            res = MLIR.IR.result(op, i)
            if shape === ()
                return TracedRNumber{typ}((), res)
            else
                return TracedRArray{typ,length(shape)}((), res, shape)
            end
        end...,
        Token(MLIR.IR.result(op, length(results) + 1)),
    )
end

# broadcast ops
function broadcast_in_dim(
    x::TracedRArray{T,N},
    dims::Vector{Int},
    result_size::Vector{Int};
    location=mlir_stacktrace("broadcast_in_dim", @__FILE__, @__LINE__),
) where {T,N}
    @assert length(dims) == N

    res = MLIR.IR.result(
        stablehlo.broadcast_in_dim(
            x.mlir_data;
            result_0=MLIR.IR.TensorType(result_size, MLIR.IR.Type(T)),
            broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims .- 1),
            location,
        ),
    )
    return TracedRArray{T,Int64(length(result_size))}((), res, Tuple(result_size))
end

function broadcast_in_dim(
    x::TracedRNumber{T},
    dims::Vector{Int},
    result_size::Vector{Int};
    location=mlir_stacktrace("broadcast_in_dim", @__FILE__, @__LINE__),
) where {T}
    @assert length(dims) == 0

    res = MLIR.IR.result(
        stablehlo.broadcast_in_dim(
            x.mlir_data;
            result_0=MLIR.IR.TensorType(result_size, MLIR.IR.Type(T)),
            broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims .- 1),
            location,
        ),
    )
    return TracedRArray{T,Int64(length(result_size))}((), res, Tuple(result_size))
end

@noinline function sort(
    xs::TracedRArray...;
    comparator,
    dimension=1,
    is_stable=false,
    location=mlir_stacktrace("sort", @__FILE__, @__LINE__),
)
    #C4:
    for x in xs
        @assert 0 < dimension <= ndims(x) "$x invalid dimension"
    end

    sample_inputs = Vector{Reactant.ConcretePJRTNumber}(undef, length(xs) * 2)
    for i in eachindex(xs)
        T = Reactant.unwrapped_eltype(xs[i])
        sample_inputs[2i - 1] = Reactant.ConcretePJRTNumber(T(0))
        sample_inputs[2i] = Reactant.ConcretePJRTNumber(T(0))
    end
    func =
        Reactant.TracedUtils.make_mlir_fn(
            comparator,
            (sample_inputs...,),
            (),
            "comparator";
            args_in_result=:none,
            return_dialect=:stablehlo,
        ).f
    @assert MLIR.IR.nregions(func) == 1
    fn_name = String(
        MLIR.IR.attr(func, String(MLIR.API.mlirSymbolTableGetSymbolAttributeName()))
    )
    #C5:
    @assert fn_name == "comparator" "$comparator: no function generated"
    ftype_attr = MLIR.IR.attr(func, "function_type")
    ftype = MLIR.IR.Type(ftype_attr)
    @assert MLIR.IR.result(ftype) == MLIR.IR.TensorType((), MLIR.IR.Type(Bool)) error(
        "$comparator return type is not tensor<i1>"
    )

    comparator = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(comparator, MLIR.IR.region(func, 1))
    MLIR.IR.rmfromparent!(func)

    dimension = MLIR.IR.Attribute(dimension - 1)
    is_stable = MLIR.IR.Attribute(is_stable)

    op = stablehlo.sort(
        [x.mlir_data for x in xs];
        result_0=[mlir_type(typeof(x), size(x)) for x in xs],
        dimension,
        is_stable,
        comparator,
        location,
    )
    return [
        TracedRArray{Reactant.unwrapped_eltype(xs[i]),ndims(xs[i])}(
            (), MLIR.IR.result(op, i), size(xs[i])
        ) for i in eachindex(xs)
    ]
end

@noinline function top_k(
    x::TracedRArray{T,N},
    k;
    dimension::Integer=N,
    location=mlir_stacktrace("top_k", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 <= dimension <= N
    if dimension != N # chlo.top_k performs the operation along the last dimension
        pdims = collect(Int64, 1:N)
        pdims[dimension] = N
        pdims[N] = dimension
        x = permutedims(x, pdims)
    end

    rsize = [size(x)[1:(end - 1)]..., k]
    values = mlir_type(TracedRArray{T,N}, rsize)
    indices = mlir_type(TracedRArray{Int32,N}, rsize)
    op = chlo.top_k(x.mlir_data; values, indices, k, location)
    indices = add(
        TracedRArray{Int32,N}((), MLIR.IR.result(op, 2), rsize),
        fill(Int32(1), Tuple(rsize)),
    ) # return the 1-indexed index
    indices = convert(TracedRArray{Int64,N}, indices) # julia indexes with Int64 generally
    values = TracedRArray{T,N}((), MLIR.IR.result(op, 1), rsize)

    if dimension != N
        values = permutedims(values, invperm(pdims))
        indices = permutedims(indices, invperm(pdims))
    end

    return (; values, indices)
end

@noinline function iota(
    T::Type,
    shape::Vector{Int};
    iota_dimension,
    location=mlir_stacktrace("iota", @__FILE__, @__LINE__),
)
    N = length(shape)
    output = mlir_type(TracedRArray{T,N}, shape)
    iota_dimension = MLIR.IR.Attribute(iota_dimension - 1)
    res = MLIR.IR.result(stablehlo.iota(; output, iota_dimension, location))
    return TracedRArray{T,N}((), res, shape)
end

@noinline function reverse(
    x::TracedRArray{T,N};
    dimensions,
    location=mlir_stacktrace("reverse", @__FILE__, @__LINE__),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.reverse(
            x.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(x)),
            dimensions=MLIR.IR.DenseArrayAttribute(collect(dimensions .- 1)),
            location,
        ),
    )
    return TracedRArray{T,N}((), res, size(x))
end

# random ops
"""
    rng_bit_generator(
        ::Type{T},
        seed::TracedRArray{UInt64,1},
        shape;
        algorithm::String="DEFAULT",
        location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
    )

Generate a random array of type `T` with the given shape and seed from a uniform random
distribution between 0 and 1. Returns a NamedTuple with the following fields:

- `output_state`: The state of the random number generator after the operation.
- `output`: The generated array.

# Arguments

- `T`: The type of the generated array.
- `seed`: The seed for the random number generator.
- `shape`: The shape of the generated array.
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to
  "DEFAULT". Other options include "PHILOX" and "THREE_FRY".
"""
@noinline function rng_bit_generator(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rng_bit_generator", @__FILE__, @__LINE__),
) where {T<:Integer}
    @assert algorithm in ("DEFAULT", "PHILOX", "THREE_FRY")
    if algorithm == "PHILOX"
        @assert length(seed) ∈ (2, 3)
    elseif algorithm == "THREE_FRY"
        @assert length(seed) == 2
    end

    output = MLIR.IR.TensorType(shape, MLIR.IR.Type(T))
    output_state = MLIR.IR.TensorType(size(seed), MLIR.IR.Type(UInt64))
    rng_algorithm = MLIR.API.stablehloRngAlgorithmAttrGet(MLIR.IR.context(), algorithm)
    op = stablehlo.rng_bit_generator(
        seed.mlir_data; output, output_state, rng_algorithm, location
    )
    return (;
        output_state=TracedRArray{UInt64,1}((), MLIR.IR.result(op, 1), size(seed)),
        output=TracedRArray{T,length(shape)}((), MLIR.IR.result(op, 2), Tuple(shape)),
    )
end

@noinline function rng_bit_generator(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rng_bit_generator", @__FILE__, @__LINE__),
) where {T<:AbstractFloat}
    nbits = sizeof(T) * 8
    uT = nbits == 16 ? UInt16 : (nbits == 32 ? UInt32 : UInt64)
    (; output_state, output) = rng_bit_generator(uT, seed, shape; algorithm, location)
    output = divide(
        convert(TracedRArray{T,ndims(output)}, output),
        fill(T(typemax(uT)), Tuple(shape); location),
    )
    return (; output_state, output)
end

"""
    randn(
        ::Type{T},
        seed::TracedRArray{UInt64,1},
        shape;
        algorithm::String="DEFAULT",
        location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
    )

Generate a random array of type `T` with the given shape and seed from a standard normal
distribution of mean 0 and standard deviation 1. Returns a NamedTuple with the following
fields:

- `output_state`: The state of the random number generator after the operation.
- `output`: The generated array.

# Arguments

- `T`: The type of the generated array.
- `seed`: The seed for the random number generator.
- `shape`: The shape of the generated array.
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to
  "DEFAULT". Other options include "PHILOX" and "THREE_FRY".
"""
@noinline function randn(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
) where {T}
    res = rng_bit_generator(T, seed, shape; algorithm, location)
    rand_uniform = res.output
    seed = res.output_state
    scaled_uniform = subtract(
        multiply(rand_uniform, fill(T(2), size(rand_uniform))),
        fill(T(1), size(rand_uniform)),
    )
    probit = erf_inv(scaled_uniform)
    rand_normal = multiply(probit, fill(Base.sqrt(T(2)), size(rand_uniform)))
    return (; output_state=seed, output=rand_normal)
end

"""
    randexp(
        ::Type{T},
        seed::TracedRArray{UInt64,1},
        shape;
        algorithm::String="DEFAULT",
        location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
    )

Generate a random array of type `T` with the given shape and seed from an exponential
distribution with rate 1. Returns a NamedTuple with the following fields:

- `output_state`: The state of the random number generator after the operation.
- `output`: The generated array.

# Arguments

- `T`: The type of the generated array.
- `seed`: The seed for the random number generator.
- `shape`: The shape of the generated array.
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to
  "DEFAULT". Other options include "PHILOX" and "THREE_FRY".
"""
@noinline function randexp(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
) where {T}
    res = rng_bit_generator(T, seed, shape; algorithm, location)
    rand_uniform = res.output
    seed = res.output_state
    rand_exp = negate(log_plus_one(negate(rand_uniform)))
    return (; output_state=seed, output=rand_exp)
end

# functional ops
@noinline function return_(
    results::Union{TracedRArray,TracedRNumber}...;
    location=mlir_stacktrace("return_", @__FILE__, @__LINE__),
)
    return stablehlo.return_([x.mlir_data for x in results]; location)
end

# control flow ops
@noinline function select(
    pred::Union{TracedRArray{Bool,N},TracedRNumber{Bool}},
    on_true::TracedRArray{T,N},
    on_false::TracedRArray{T,N},
) where {T,N}
    @assert size(on_true) == size(on_false) "`on_true` and `on_false` must have the same size"
    @assert size(pred) == size(on_true) || size(pred) == () "`pred` must have the same size as `on_true`/`on_false` or be a scalar"

    res = MLIR.IR.result(
        stablehlo.select(
            pred.mlir_data,
            on_true.mlir_data,
            on_false.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(on_true)),
        ),
    )
    return TracedRArray{T,N}((), res, size(on_true))
end

@noinline function select(
    pred::TracedRNumber{Bool}, on_true::TracedRNumber{T}, on_false::TracedRNumber{T}
) where {T}
    res = MLIR.IR.result(
        stablehlo.select(
            pred.mlir_data,
            on_true.mlir_data,
            on_false.mlir_data;
            result=mlir_type(TracedRArray{T,0}, ()),
        ),
    )
    return TracedRNumber{T}((), res)
end

# comparison
@noinline function compare(
    lhs::AT,
    rhs::AT;
    comparison_direction::String,
    compare_type=nothing,
    location=mlir_stacktrace("compare", @__FILE__, @__LINE__),
) where {AT<:Union{TracedRArray,TracedRNumber}}
    @assert comparison_direction in ("EQ", "NE", "GE", "GT", "LE", "LT")
    @assert size(lhs) == size(rhs)

    res = MLIR.IR.result(
        stablehlo.compare(
            lhs.mlir_data,
            rhs.mlir_data;
            comparison_direction=MLIR.API.stablehloComparisonDirectionAttrGet(
                MLIR.IR.context(), comparison_direction
            ),
            compare_type,
            location,
        ),
        1,
    )
    lhs isa TracedRNumber && return TracedRNumber{Bool}((), res)
    return TracedRArray{Bool,ndims(lhs)}((), res, size(lhs))
end

# eltype conversion
@noinline function convert(
    ::Type{TracedRArray{T,N}},
    x::TracedRArray;
    location=mlir_stacktrace("convert", @__FILE__, @__LINE__),
) where {T,N}
    @assert N == ndims(x)
    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            stablehlo.convert(
                x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location
            ),
        ),
        size(x),
    )
end

@noinline function convert(
    ::Type{TracedRNumber{T}},
    x::TracedRNumber;
    location=mlir_stacktrace("convert", @__FILE__, @__LINE__),
) where {T}
    return TracedRNumber{T}(
        (),
        MLIR.IR.result(
            stablehlo.convert(x.mlir_data; result=mlir_type(TracedRNumber{T}), location)
        ),
    )
end

# Generate a unique name given a module hash and a function name.
function _hlo_call_name(orig_name, module_suffix)
    return orig_name * "_hlo_call_" * module_suffix
end

"""
    Ops.hlo_call(mlir_code::String, args::Vararg{AnyTracedRArray}...; func_name::String="main") -> NTuple{N, AnyTracedRArray}

Given a MLIR module given as a string, calls the function identified by the `func_name` keyword parameter (default "main")
with the provided arguments and return a tuple for each result of the call.

```julia-repl
julia> Reactant.@jit(
          Ops.hlo_call(
              \"\"\"
              module {
                func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                  %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
                  return %0 : tensor<3xf32>
                }
              }
              \"\"\",
              Reactant.to_rarray(Float32[1, 2, 3]),
              Reactant.to_rarray(Float32[1, 2, 3]),
          )
       )
(ConcretePJRTArray{Float32, 1}(Float32[2.0, 4.0, 6.0]),)
```
"""
@noinline function hlo_call(
    code,
    args...;
    func_name="main",
    location=mlir_stacktrace("hlo_call", @__FILE__, @__LINE__),
)
    module_suffix = string(hash(code); base=16)
    name_to_call = _hlo_call_name(func_name, module_suffix)

    current_module = MLIR.IR.mmodule()
    top_level_block = MLIR.IR.body(current_module)

    symbol_attr_name = String(MLIR.API.mlirSymbolTableGetSymbolAttributeName())

    fn = MLIR.IR.lookup(
        MLIR.IR.SymbolTable(MLIR.IR.Operation(current_module)), name_to_call
    )
    if isnothing(fn)
        new_mod = parse(MLIR.IR.Module, code)
        new_mod_op = MLIR.IR.Operation(new_mod)
        body = MLIR.IR.body(new_mod)

        operations = collect(MLIR.IR.OperationIterator(body))
        for op in operations
            if MLIR.IR.name(op) == "func.func"
                fn_name = String(MLIR.IR.attr(op, symbol_attr_name))
                if fn_name == func_name
                    fn = op
                end

                new_name = _hlo_call_name(fn_name, module_suffix)
                res = MLIR.IR.LogicalResult(
                    MLIR.API.mlirSymbolTableReplaceAllSymbolUses(
                        fn_name, new_name, new_mod_op
                    ),
                )
                @assert res == MLIR.IR.success() "hlo_call: failed to rename $fn_name"

                # Set function private
                MLIR.IR.attr!(
                    op,
                    MLIR.API.mlirSymbolTableGetVisibilityAttributeName(),
                    MLIR.IR.Attribute("private"),
                )

                # Change function name
                MLIR.IR.attr!(op, symbol_attr_name, MLIR.IR.Attribute(new_name))
            end
        end

        for op in operations
            MLIR.IR.rmfromparent!(op)
            push!(top_level_block, op)
        end
    end

    if isnothing(fn)
        error("hlo_call: could not find function $func_name in the provided module")
    end

    ftype_attr = MLIR.IR.attr(fn, "function_type")
    ftype = MLIR.IR.Type(ftype_attr)

    @assert all(Base.Fix2(isa, Reactant.AnyTracedRArray), args) "hlo_call: all inputs to hlo_call should be reactant arrays"
    @assert MLIR.IR.ninputs(ftype) == length(args) "hlo_call: invalid number of arguments for function $func_name"

    for (i, arg) in enumerate(args)
        expected_type = MLIR.IR.input(ftype, i)
        arg_type = MLIR.IR.type(arg.mlir_data)
        @assert expected_type == arg_type "hlo_call: argument #$i has the wrong type (expected $expected_type, got $arg_type)"
    end

    operands = [a.mlir_data for a in args]
    call = MLIR.Dialects.func.call(
        operands;
        result_0=[MLIR.IR.result(ftype, i) for i in 1:MLIR.IR.nresults(ftype)],
        callee=MLIR.IR.FlatSymbolRefAttribute(name_to_call),
        location,
    )

    return ntuple(MLIR.IR.nresults(call)) do i
        out = MLIR.IR.result(call, i)
        ty = MLIR.IR.type(out)
        sz = MLIR.IR.size(ty)
        T = MLIR.IR.julia_type(eltype(ty))
        N = length(sz)
        if N == 0
            Reactant.TracedRNumber{T}((), out)
        else
            Reactant.TracedRArray{T,N}((), out, sz)
        end
    end
end

"""
    scatter_setindex(dest, scatter_indices, updates)

Uses [`MLIR.Dialects.stablehlo.scatter`](@ref) to set the values of `dest` at the indices
specified by `scatter_indices` to the values in `updates`. If the indices are contiguous it
is recommended to directly use [`MLIR.Dialects.stablehlo.dynamic_update_slice`](@ref)
instead.
"""
@noinline function scatter_setindex(
    dest::TracedRArray{T,N},
    scatter_indices::TracedRArray{Int64,2},
    updates::TracedRArray{T2,1},
) where {T,N,T2}
    @assert length(updates) == size(scatter_indices, 1)
    @assert size(scatter_indices, 2) == N

    updates = convert(TracedRArray{T,1}, updates)

    update_computation = MLIR.IR.Region()
    block = MLIR.IR.Block(
        [mlir_type(TracedRNumber{T}), mlir_type(TracedRNumber{T})],
        [MLIR.IR.Location(), MLIR.IR.Location()],
    )
    return_op = MLIR.Dialects.stablehlo.return_([MLIR.IR.argument(block, 2)])
    MLIR.IR.rmfromparent!(return_op)
    push!(block, return_op)
    pushfirst!(update_computation, block)

    #! format: off
    update_window_dims = Int64[]
    inserted_window_dims = collect(Int64, 0:(N - 1))
    input_batching_dims = Int64[]
    scatter_indices_batching_dims = Int64[]
    scatter_dims_to_operand_dims = collect(Int64, 0:(N - 1))
    index_vector_dim = Int64(1)

    scatter_dimension_numbers = MLIR.API.stablehloScatterDimensionNumbersGet(
        MLIR.IR.context(),
        length(update_window_dims), update_window_dims,
        length(inserted_window_dims), inserted_window_dims,
        length(input_batching_dims), input_batching_dims,
        length(scatter_indices_batching_dims), scatter_indices_batching_dims,
        length(scatter_dims_to_operand_dims), scatter_dims_to_operand_dims,
        index_vector_dim,
    )
    #! format: on

    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.scatter(
                [dest.mlir_data],
                scatter_indices.mlir_data,
                [updates.mlir_data];
                result_0=[mlir_type(TracedRArray{T,N}, size(dest))],
                update_computation,
                scatter_dimension_numbers,
            ),
            1,
        ),
        size(dest),
    )
end

"""
    gather_getindex(src, gather_indices)

Uses [`MLIR.Dialects.stablehlo.gather`](@ref) to get the values of `src` at the indices
specified by `gather_indices`. If the indices are contiguous it is recommended to directly
use [`MLIR.Dialects.stablehlo.dynamic_slice`](@ref) instead.
"""
@noinline function gather_getindex(
    src::TracedRArray{T,N}, gather_indices::TracedRArray{Int64,2}
) where {T,N}
    @assert size(gather_indices, 2) == N

    #! format: off
    offset_dims = Int64[1]
    collapsed_slice_dims = collect(Int64, 0:(N - 2))
    operand_batching_dims = Int64[]
    start_indices_batching_dims = Int64[]
    start_index_map = collect(Int64, 0:(N - 1))
    index_vector_dim = Int64(1)

    dimension_numbers = MLIR.API.stablehloGatherDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(length(offset_dims)), offset_dims,
        Int64(length(collapsed_slice_dims)), collapsed_slice_dims,
        Int64(length(operand_batching_dims)), operand_batching_dims,
        Int64(length(start_indices_batching_dims)), start_indices_batching_dims,
        Int64(length(start_index_map)), start_index_map,
        Int64(index_vector_dim),
    )
    #! format: on

    return reshape(
        TracedRArray{T}(
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.gather(
                    src.mlir_data,
                    gather_indices.mlir_data;
                    dimension_numbers,
                    slice_sizes=Base.fill(Int64(1), N),
                    indices_are_sorted=false,
                ),
                1,
            ),
        ),
        size(gather_indices, 1),
    )
end

@noinline function while_loop(cond_fn::CFn, body_fn::BFn, args...) where {CFn,BFn}
    # TODO: detect and prevent mutation within the condition

    # Make all the args traced or concrete
    N = length(args)
    seen_args = Reactant.OrderedIdDict()
    traced_args = Vector{Any}(undef, N)
    for i in 1:N
        @inbounds traced_args[i] = Reactant.make_tracer(
            seen_args, args[i], (), Reactant.NoStopTracedTrack; track_numbers=Number
        )
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seen_args
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
    end

    input_types = [mlir_type(arg) for arg in linear_args]

    cond_fn_compiled =
        Reactant.TracedUtils.make_mlir_fn(
            cond_fn,
            traced_args,
            (),
            string(gensym("cond_fn")),
            false;
            return_dialect=:stablehlo,
            args_in_result=:none,
            do_transpose=false,
        ).f

    body_fn_compiled =
        Reactant.TracedUtils.make_mlir_fn(
            body_fn,
            traced_args,
            (),
            string(gensym("body_fn")),
            false;
            return_dialect=:stablehlo,
            args_in_result=:none,
            do_transpose=false,
        ).f

    cond_reg = Reactant.TracedUtils.__take_region(cond_fn_compiled)
    body_reg = Reactant.TracedUtils.__take_region(body_fn_compiled)

    MLIR.IR.rmfromparent!(cond_fn_compiled)
    MLIR.IR.rmfromparent!(body_fn_compiled)

    while_op = MLIR.Dialects.stablehlo.while_(
        MLIR.IR.Value[Reactant.TracedUtils.get_mlir_data(arg) for arg in linear_args];
        result_0=input_types,
        cond=cond_reg,
        body=body_reg,
    )

    return map(enumerate(linear_args)) do (i, arg)
        Reactant.TracedUtils.set_mlir_data!(arg, MLIR.IR.result(while_op, i))
    end
end

@noinline function if_condition(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args...
) where {TFn,FFn}
    true_fn_names = (gensym(:true_fn_args), gensym(:true_result), gensym(:true_fn_resargs))
    false_fn_names = (
        gensym(:false_fn_args), gensym(:false_result), gensym(:false_fn_resargs)
    )

    # Make all the args traced or concrete
    N = length(args)
    tb_seen_args = Reactant.OrderedIdDict()
    fb_seen_args = Reactant.OrderedIdDict()
    tb_traced_args = Vector{Any}(undef, N)
    fb_traced_args = Vector{Any}(undef, N)
    for i in 1:N
        @inbounds tb_traced_args[i] = Reactant.make_tracer(
            tb_seen_args,
            args[i],
            (true_fn_names[1], i),
            Reactant.TracedSetPath;
            track_numbers=Number,
        )
        @inbounds fb_traced_args[i] = Reactant.make_tracer(
            fb_seen_args,
            args[i],
            (false_fn_names[1], i),
            Reactant.TracedSetPath;
            track_numbers=Number,
        )
    end

    tb_linear_args = Reactant.TracedType[
        v for v in values(tb_seen_args) if v isa Reactant.TracedType
    ]
    fb_linear_args = Reactant.TracedType[
        v for v in values(fb_seen_args) if v isa Reactant.TracedType
    ]

    input_types = [mlir_type(arg) for arg in tb_linear_args]
    sym_visibility = MLIR.IR.Attribute("private")

    # compile the true branch without any returns first
    true_fn_mod = MLIR.IR.mmodule()
    true_func_tmp = MLIR.IR.block!(MLIR.IR.body(true_fn_mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=string(true_fn) * "_tb_tmp",
            function_type=MLIR.IR.FunctionType(input_types, []),
            body=MLIR.IR.Region(),
            sym_visibility,
        )
    end
    true_fn_body = MLIR.IR.Block()
    push!(MLIR.IR.region(true_func_tmp, 1), true_fn_body)

    true_fn_args = true_fn_names[1]

    MLIR.IR.activate!(true_fn_body)
    tb_result = try
        for (i, arg) in enumerate(tb_linear_args)
            # find the right path to index the traced arg.
            path = nothing
            for p in Reactant.TracedUtils.get_paths(arg)
                if length(p) > 0 && p[1] == true_fn_args
                    path = p[2:end]
                end
            end
            if isnothing(path)
                error("if_condition: could not find path for linear arg $i")
            end
            Reactant.TracedUtils.set_mlir_data!(
                arg,
                only(
                    Reactant.TracedUtils.push_val!(
                        [], tb_traced_args[path[1]], path[2:end]
                    ),
                ),
            )
        end
        Reactant.call_with_reactant(true_fn, tb_traced_args...)
    finally
        MLIR.IR.deactivate!(true_fn_body)
    end

    seen_true_results = Reactant.OrderedIdDict()
    traced_true_results = Reactant.make_tracer(
        seen_true_results,
        tb_result,
        (true_fn_names[2],),
        Reactant.NoStopTracedTrack;
        track_numbers=Number,
    )
    for i in eachindex(tb_linear_args)
        Reactant.make_tracer(
            seen_true_results,
            tb_linear_args[i],
            (true_fn_names[3], i),
            Reactant.NoStopTracedTrack;
            track_numbers=Number,
        )
    end

    tb_linear_results = Reactant.TracedType[
        v for v in values(seen_true_results) if v isa Reactant.TracedType
    ]

    # compile the false branch without any returns similar to the true branch
    false_fn_mod = MLIR.IR.mmodule()
    false_func_tmp = MLIR.IR.block!(MLIR.IR.body(false_fn_mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=string(false_fn) * "_fb_tmp",
            function_type=MLIR.IR.FunctionType(input_types, []),
            body=MLIR.IR.Region(),
            sym_visibility,
        )
    end
    false_fn_body = MLIR.IR.Block()
    push!(MLIR.IR.region(false_func_tmp, 1), false_fn_body)

    false_fn_args = false_fn_names[1]
    MLIR.IR.activate!(false_fn_body)
    fb_result = try
        for (i, arg) in enumerate(fb_linear_args)
            # find the right path to index the traced arg.
            path = nothing
            for p in Reactant.TracedUtils.get_paths(arg)
                if length(p) > 0 && p[1] == false_fn_args
                    path = p[2:end]
                end
            end
            if isnothing(path)
                error("if_condition: could not find path for linear arg $i")
            end
            Reactant.TracedUtils.set_mlir_data!(
                arg,
                only(
                    Reactant.TracedUtils.push_val!(
                        [], fb_traced_args[path[1]], path[2:end]
                    ),
                ),
            )
        end
        Reactant.call_with_reactant(false_fn, fb_traced_args...)
    finally
        MLIR.IR.deactivate!(false_fn_body)
    end

    seen_false_results = Reactant.OrderedIdDict()
    traced_false_results = Reactant.make_tracer(
        seen_false_results,
        fb_result,
        (false_fn_names[2],),
        Reactant.NoStopTracedTrack;
        track_numbers=Number,
    )
    for i in eachindex(fb_linear_args)
        Reactant.make_tracer(
            seen_false_results,
            fb_linear_args[i],
            (false_fn_names[3], i),
            Reactant.NoStopTracedTrack;
            track_numbers=Number,
        )
    end

    fb_linear_results = Reactant.TracedType[
        v for v in values(seen_false_results) if v isa Reactant.TracedType
    ]

    tb_results_dict = Dict{Tuple,Reactant.TracedType}()
    for tr in tb_linear_results
        for path in Reactant.TracedUtils.get_paths(tr)
            if length(path) > 0 &&
                (path[1] == true_fn_names[2] || path[1] == true_fn_names[3])
                tb_results_dict[path] = tr
            end
        end
    end

    fb_results_dict = Dict{Tuple,Reactant.TracedType}()
    for fr in fb_linear_results
        for path in Reactant.TracedUtils.get_paths(fr)
            if length(path) > 0 &&
                (path[1] == false_fn_names[2] || path[1] == false_fn_names[3])
                fb_results_dict[path] = fr
            end
        end
    end

    all_paths = []
    for (path, tr) in tb_results_dict
        if path[1] == true_fn_names[2]
            push!(all_paths, (:result, path[2:end]...))
        elseif path[1] == true_fn_names[3]
            push!(all_paths, (:resarg, path[2:end]...))
        end
    end
    for (path, fr) in fb_results_dict
        if path[1] == false_fn_names[2]
            push!(all_paths, (:result, path[2:end]...))
        elseif path[1] == false_fn_names[3]
            push!(all_paths, (:resarg, path[2:end]...))
        end
    end
    all_paths = sort!(unique!(all_paths))
    tb_paths = [
        if path[1] == :result
            (true_fn_names[2], path[2:end]...)
        else
            (true_fn_names[3], path[2:end]...)
        end for path in all_paths
    ]
    fb_paths = [
        if path[1] == :result
            (false_fn_names[2], path[2:end]...)
        else
            (false_fn_names[3], path[2:end]...)
        end for path in all_paths
    ]

    # finalize the true branch by adding the missing values
    MLIR.IR.activate!(true_fn_body)
    tb_corrected_linear_results = Reactant.TracedType[]
    try
        for (i, path) in enumerate(tb_paths)
            if haskey(tb_results_dict, tb_paths[i])
                push!(tb_corrected_linear_results, tb_results_dict[tb_paths[i]])
            else
                push!(tb_corrected_linear_results, zero(fb_results_dict[fb_paths[i]]))
            end
        end
    finally
        MLIR.IR.deactivate!(true_fn_body)
    end

    # finalize the false branch by adding the missing values
    MLIR.IR.activate!(false_fn_body)
    fb_corrected_linear_results = Reactant.TracedType[]
    try
        for (i, path) in enumerate(fb_paths)
            if haskey(fb_results_dict, fb_paths[i])
                push!(fb_corrected_linear_results, fb_results_dict[fb_paths[i]])
            else
                push!(fb_corrected_linear_results, zero(tb_results_dict[tb_paths[i]]))
            end
        end
    finally
        MLIR.IR.deactivate!(false_fn_body)
    end

    # All MissingTracedValues must be replaced with zeroes
    @assert length(tb_corrected_linear_results) == length(fb_corrected_linear_results)

    result_types = MLIR.IR.Type[]
    for (i, (tr, fr)) in
        enumerate(zip(tb_corrected_linear_results, fb_corrected_linear_results))
        if tr isa MissingTracedValue && fr isa MissingTracedValue
            continue # Don't insert into IR
        end
        res = if tr isa MissingTracedValue
            @assert !(fr isa MissingTracedValue)
            MLIR.IR.activate!(true_fn_body)
            try
                tb_corrected_linear_results[i] = zero(fr)
            finally
                MLIR.IR.deactivate!(true_fn_body)
            end
            fr
        elseif fr isa MissingTracedValue
            @assert !(tr isa MissingTracedValue)
            MLIR.IR.activate!(false_fn_body)
            try
                fb_corrected_linear_results[i] = zero(tr)
            finally
                MLIR.IR.deactivate!(false_fn_body)
            end
            tr
        else
            if typeof(tr) != typeof(fr)
                @assert typeof(tr) == typeof(fr) "$(typeof(tr)) vs $(typeof(fr))"
            end
            tr
        end
        push!(result_types, mlir_type(res))
    end

    MLIR.IR.activate!(true_fn_body)
    try
        vals = MLIR.IR.Value[
            Reactant.TracedUtils.get_mlir_data(res) for
            res in tb_corrected_linear_results if !(res isa MissingTracedValue)
        ]
        MLIR.Dialects.stablehlo.return_(vals)
    finally
        MLIR.IR.deactivate!(true_fn_body)
    end

    MLIR.IR.activate!(false_fn_body)
    try
        vals = MLIR.IR.Value[
            Reactant.TracedUtils.get_mlir_data(res) for
            res in fb_corrected_linear_results if !(res isa MissingTracedValue)
        ]
        MLIR.Dialects.stablehlo.return_(vals)
    finally
        MLIR.IR.deactivate!(false_fn_body)
    end

    # With the corrected results, we can compile the true and false branches
    tb_out_types = [mlir_type(tr) for tr in tb_corrected_linear_results]

    true_fn_compiled = MLIR.IR.block!(MLIR.IR.body(true_fn_mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=Reactant.TracedUtils.__lookup_unique_name_in_module(
                true_fn_mod, string(true_fn) * "_tb"
            ),
            function_type=MLIR.IR.FunctionType(input_types, tb_out_types),
            body=MLIR.IR.Region(),
            sym_visibility,
        )
    end
    MLIR.API.mlirRegionTakeBody(
        MLIR.IR.region(true_fn_compiled, 1), MLIR.IR.region(true_func_tmp, 1)
    )
    MLIR.API.mlirOperationDestroy(true_func_tmp.operation)
    true_func_tmp.operation = MLIR.API.MlirOperation(C_NULL)

    fb_out_types = [mlir_type(fr) for fr in fb_corrected_linear_results]

    false_fn_compiled = MLIR.IR.block!(MLIR.IR.body(false_fn_mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=Reactant.TracedUtils.__lookup_unique_name_in_module(
                false_fn_mod, string(false_fn) * "_fb"
            ),
            function_type=MLIR.IR.FunctionType(input_types, fb_out_types),
            body=MLIR.IR.Region(),
            sym_visibility,
        )
    end
    MLIR.API.mlirRegionTakeBody(
        MLIR.IR.region(false_fn_compiled, 1), MLIR.IR.region(false_func_tmp, 1)
    )
    MLIR.API.mlirOperationDestroy(false_func_tmp.operation)
    false_func_tmp.operation = MLIR.API.MlirOperation(C_NULL)

    tb_region = Reactant.TracedUtils.__take_region(true_fn_compiled)
    fb_region = Reactant.TracedUtils.__take_region(false_fn_compiled)

    MLIR.IR.rmfromparent!(true_fn_compiled)
    MLIR.IR.rmfromparent!(false_fn_compiled)

    if_compiled = MLIR.Dialects.stablehlo.if_(
        cond.mlir_data; true_branch=tb_region, false_branch=fb_region, result_0=result_types
    )

    corrected_traced_results = fmap(traced_false_results, traced_true_results) do fr, tr
        if fr isa MissingTracedValue && tr isa MissingTracedValue
            error("Both false and true branches are missing")
        elseif fr isa MissingTracedValue
            return tr
        else
            return fr
        end
    end

    for (residx, path) in enumerate(all_paths)
        if path[1] == :result
            Reactant.TracedUtils.set!(
                corrected_traced_results, path[2:end], MLIR.IR.result(if_compiled, residx)
            )
        elseif path[1] == :resarg
            # The resarg path is with respect to the linear args, not the traced args.
            # We find the path into traced args by searching for it in the linear args.
            # Concretely, we look into tb_linear_args, but we could also look into fb_linear_args, they contain the same arg path.
            @assert length(path) == 2
            argpath = nothing
            for p in Reactant.TracedUtils.get_paths(tb_linear_args[path[2]])
                if length(p) > 0 && p[1] == true_fn_names[1]
                    argpath = p[2:end]
                end
            end
            if isnothing(argpath)
                error("if_condition: could not find path for resarg $path")
            end
            Reactant.TracedUtils.set!(args, argpath, MLIR.IR.result(if_compiled, residx))
        end
    end

    return corrected_traced_results
end

@noinline function call(f, args...)
    seen_cache = Reactant.OrderedIdDict()
    Reactant.make_tracer(
        seen_cache,
        args,
        (), # we have to insert something here, but we remove it immediately below.
        Reactant.TracedTrack;
        toscalar=false,
    )
    linear_args = []
    mlir_caller_args = Reactant.MLIR.IR.Value[]
    for (k, v) in seen_cache
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
        push!(mlir_caller_args, v.mlir_data)
        # make tracer inserted `()` into the path, here we remove it:
        v.paths = v.paths[1:(end - 1)]
    end

    seen = Dict()
    cache_key = []
    Reactant.make_tracer(seen, (f, args...), cache_key, Reactant.TracedToTypes)
    cache = Reactant.Compiler.callcache()
    if haskey(cache, cache_key)
        # cache lookup:
        (; f_name, mlir_result_types, traced_result, mutated_args) = cache[cache_key]
    else
        f_name = String(gensym(Symbol(f)))
        temp = Reactant.TracedUtils.make_mlir_fn(
            f, args, (), f_name, false; args_in_result=:mutated, do_transpose=false
        )
        (; traced_result, ret, mutated_args) = temp
        mlir_result_types = [
            MLIR.IR.type(MLIR.IR.operand(ret, i)) for i in 1:MLIR.IR.noperands(ret)
        ]
        cache[cache_key] = (; f_name, mlir_result_types, traced_result, mutated_args)
    end

    call_op = MLIR.Dialects.func.call(
        mlir_caller_args;
        result_0=mlir_result_types,
        callee=MLIR.IR.FlatSymbolRefAttribute(f_name),
    )

    seen_results = Reactant.OrderedIdDict()
    traced_result = Reactant.make_tracer(
        seen_results,
        traced_result,
        (), # we have to insert something here, but we remove it immediately below.
        Reactant.TracedSetPath;
        toscalar=false,
    )
    i = 1
    for (k, v) in seen_results
        v isa Reactant.TracedType || continue
        # this mutates `traced_result`, which is what we want:
        v.mlir_data = MLIR.IR.result(call_op, i)
        # make tracer inserted `()` into the path, here we remove it:
        v.paths = v.paths[1:(end - 1)]
        i += 1
    end
    nres = MLIR.IR.nresults(call_op)
    # mutated args are included as the last ones in the call op results
    for (result_i, arg_i) in zip((nres - length(mutated_args)):nres, mutated_args)
        Reactant.TracedUtils.set_mlir_data!(
            linear_args[arg_i], MLIR.IR.result(call_op, result_i + 1)
        )
    end
    return traced_result
end

# Shardy Ops
"""
    mesh(
        mesh::Reactant.Sharding.Mesh; mod::MLIR.IR.Module=MLIR.IR.mmodule(),
        sym_name::String="mesh",
        location=mlir_stacktrace("mesh", @__FILE__, @__LINE__)
    )
    mesh(
        mesh_axes::Vector{<:Pair{<:Union{String,Symbol},Int64}},
        device_ids::Vector{Int64};
        sym_name::String="mesh",
        mod::MLIR.IR.Module=MLIR.IR.mmodule(),
        location=mlir_stacktrace("mesh", @__FILE__, @__LINE__)
    )

Produces a [`Reactant.MLIR.Dialects.sdy.mesh`](@ref) operation with the given `mesh` and
`device_ids`.

Based on the provided `sym_name``, we generate a unique name for the mesh in the module's
`SymbolTable`. Note that users shouldn't use this sym_name directly, instead they should
use the returned `sym_name` to refer to the mesh in the module.

!!! warning

    The `device_ids` argument are the logical device ids, not the physical device ids.
    For example, if the physical device ids are `[2, 4, 123, 293]`, the corresponding
    logical device ids are `[0, 1, 2, 3]`.

## Returned Value

We return a NamedTuple with the following fields:

- `sym_name`: The unique name of the mesh in the module's `SymbolTable`.
- `mesh_attr`: `sdy::mlir::MeshAttr` representing the mesh.
- `mesh_op`: The `sdy.mesh` operation.
"""
@noinline function mesh(
    m::Reactant.Sharding.Mesh;
    mod::MLIR.IR.Module=MLIR.IR.mmodule(),
    sym_name::String="mesh",
    location=mlir_stacktrace("mesh", @__FILE__, @__LINE__),
)
    cache = Reactant.Compiler.sdycache(; throw_error=ReactantCore.within_compile())
    cache !== nothing && haskey(cache, m) && return cache[m]
    result = mesh(
        [k => Int64(v) for (k, v) in zip(m.axis_names, size(m))],
        m.logical_device_ids;
        mod,
        sym_name,
        location,
    )
    cache !== nothing && (cache[m] = result)
    return result
end

@noinline function mesh(
    mesh_axes::Vector{<:Pair{<:Union{String,Symbol},Int64}},
    device_ids::AbstractVector{Int64};
    mod::MLIR.IR.Module=MLIR.IR.mmodule(),
    sym_name::String="mesh",
    location=mlir_stacktrace("mesh", @__FILE__, @__LINE__),
)
    # See https://github.com/openxla/shardy/blob/f9d83e779a58b811b848c4edfaf68e88b636787d/shardy/dialect/sdy/ir/verifiers.cc#L647-L699 for the checks
    ndevices = prod(last, mesh_axes)

    @assert allunique(first, mesh_axes) "mesh_axes must be unique"
    @assert ndevices == length(device_ids) "length(device_ids) should be same as \
                                            prod(last, mesh_axes)"
    @assert all(Base.Fix2(≥, 0), device_ids) "device_ids must be non-negative"
    @assert Base.sort(device_ids) == 0:(ndevices - 1) "sorted device_ids must be the same \
                                                       as iota(product(axes)), got \
                                                       $(Base.sort(device_ids))"

    # error: if the ordered device ids are the same as iota(product(axes)), no need to
    # specify them for simplicity
    issorted(device_ids) && (device_ids = Int64[])

    ctx = MLIR.IR.context()
    mesh_axis_attrs = [
        MLIR.API.sdyMeshAxisAttrGet(ctx, String(name), size) for (name, size) in mesh_axes
    ]
    mesh_attr = MLIR.API.sdyMeshAttrGet(
        ctx,
        Int64(length(mesh_axis_attrs)),
        mesh_axis_attrs,
        Int64(length(device_ids)),
        collect(Int64, device_ids),
    )

    sym_name = Reactant.TracedUtils.__lookup_unique_name_in_module(mod, sym_name)

    mesh_op = MLIR.IR.mmodule!(mod) do
        return MLIR.Dialects.sdy.mesh(; sym_name, mesh=mesh_attr, location)
    end

    # mesh_op needs to be moved to the beginning of the module
    mesh_op = MLIR.IR.rmfromparent!(mesh_op)
    mod_body = MLIR.IR.body(mod)
    pushfirst!(mod_body, mesh_op)

    # We return the name of the mesh, since the operation is a Symbol op
    return (;
        sym_name=MLIR.IR.FlatSymbolRefAttribute(sym_name; context=ctx),
        mesh_attr=MLIR.IR.Attribute(mesh_attr),
        mesh_op=mesh_op,
    )
end

"""
    sharding_constraint(
        input::Union{TracedRArray,TracedRNumber},
        sharding::Reactant.Sharding.AbstractSharding;
        location=mlir_stacktrace("sharding_constraint", @__FILE__, @__LINE__)
    )

Produces a [`Reactant.MLIR.Dialects.sdy.sharding_constraint`](@ref) operation with the given
`input` and `sharding`.
"""
@noinline function sharding_constraint(
    input::Union{AbstractArray,Number},
    sharding::Reactant.Sharding.AbstractSharding;
    location=mlir_stacktrace("sharding_constraint", @__FILE__, @__LINE__),
)
    !(input isa TracedRNumber || input isa TracedRArray) &&
        (input = constant(input; location))

    cache = Reactant.Compiler.sdycache()
    haskey(cache, sharding.mesh) || Ops.mesh(sharding.mesh; location)
    (; sym_name, mesh_attr) = cache[sharding.mesh]
    tensor_sharding_attr = Reactant.Sharding.get_shardy_tensor_sharding_attribute(
        sharding, MLIR.IR.context(), sym_name, mesh_attr; do_transpose=true
    )
    resharded_value = MLIR.IR.result(
        MLIR.Dialects.sdy.sharding_constraint(
            input.mlir_data; sharding=tensor_sharding_attr, location
        ),
        1,
    )
    if input isa TracedRNumber
        return TracedRNumber{unwrapped_eltype(input)}(resharded_value)
    else
        return TracedRArray{unwrapped_eltype(input)}(resharded_value)
    end
end

@noinline function reduce(
    x::TracedRArray{T},
    init_values::TracedRNumber{T},
    dimensions::Vector{Int},
    fn::Function,
    location=mlir_stacktrace("reduce", @__FILE__, @__LINE__)
) where {T}
    reduced_shape = Tuple(deleteat!(collect(size(x)), dimensions))

    result_type = mlir_type(TracedRArray{T, length(reduced_shape)}, reduced_shape)
    
    sample_inputs = [Reactant.ConcretePJRTNumber(T(0)), Reactant.ConcretePJRTNumber(T(0))]

    func =
        Reactant.TracedUtils.make_mlir_fn(
            fn,
            (sample_inputs),
            (),
            "reduce_fn";
            args_in_result=:none,
            return_dialect=:stablehlo,
        ).f
    @assert MLIR.IR.nregions(func) == 1
    fn_name = String(
        MLIR.IR.attr(func, String(MLIR.API.mlirSymbolTableGetSymbolAttributeName()))
    )
    @assert fn_name == "reduce_fn"
    ftype_attr = MLIR.IR.attr(func, "function_type")
    ftype = MLIR.IR.Type(ftype_attr)
    @assert MLIR.IR.result(ftype) == MLIR.IR.TensorType((), MLIR.IR.Type(T)) error (
        "$fn return type is not tensor<i1>"
    )
    fn = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(fn, MLIR.IR.region(func, 1))
    MLIR.IR.rmfromparent!(func)
    
    dimensions = MLIR.IR.Attribute(dimensions .- 1) 

    res = MLIR.IR.result(stablehlo.reduce(
        [x.mlir_data], [init_values.mlir_data];
        result_0=[result_type], dimensions=dimensions, body=fn, location=location
    ))

    return TracedRArray{T, length(reduced_shape)}((), res, reduced_shape)
end

end # module Ops
