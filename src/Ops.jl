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
using Functors: Functors, fmap

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
    res = constant(fill(x); location)
    return TracedRNumber{T}((), res.mlir_data)
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
for (dialect, op) in [
    #(:stablehlo, :is_finite),
    (:chlo, :is_inf),
    (:chlo, :is_neg_inf),
    (:chlo, :is_pos_inf),
]
    @eval begin
        @noinline function $op(
            x::TracedRArray{T,N};
            location=mlir_stacktrace($(string(op)), @__FILE__, @__LINE__),
        ) where {T,N}
            res = MLIR.IR.result(
                $(:($dialect.$op))(
                    x.mlir_data; result=mlir_type(TracedRArray{Bool,N}, size(x)), location
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
                    x.mlir_data; result=mlir_type(TracedRArray{Bool,0}, ()), location
                ),
            )
            return TracedRNumber{Bool}((), res)
        end
    end
end

@noinline function is_finite(
    x::TracedRArray{T,N}; location=mlir_stacktrace("is_finite", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.is_finite(
            x.mlir_data; y=mlir_type(TracedRArray{Bool,N}, size(x)), location
        ),
    )
    return TracedRArray{Bool,N}((), res, size(x))
end

@noinline function is_finite(
    x::TracedRNumber{T}; location=mlir_stacktrace("is_finite", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.is_finite(x.mlir_data; y=mlir_type(TracedRArray{Bool,0}, ()), location)
    )
    return TracedRNumber{Bool}((), res)
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
    low=fill(0, N),
    high=fill(0, N),
    interior=fill(0, N),
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

    sample_inputs = Vector{Reactant.ConcreteRNumber}(undef, length(xs) * 2)
    for i in eachindex(xs)
        T = Reactant.unwrapped_eltype(xs[i])
        sample_inputs[2i - 1] = Reactant.ConcreteRNumber(T(0))
        sample_inputs[2i] = Reactant.ConcreteRNumber(T(0))
    end
    func = Reactant.TracedUtils.make_mlir_fn(
        comparator,
        (sample_inputs...,),
        (),
        "comparator";
        no_args_in_result=true,
        return_dialect=:stablehlo,
    )[2]
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
        constant(fill(Int32(1), Tuple(rsize))),
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
        constant(fill(T(typemax(uT)), Tuple(shape)); location),
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
        multiply(rand_uniform, constant(fill(T(2), size(rand_uniform)))),
        constant(fill(T(1), size(rand_uniform))),
    )
    probit = erf_inv(scaled_uniform)
    rand_normal = multiply(probit, constant(fill(Base.sqrt(T(2)), size(rand_uniform))))
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
(ConcreteRArray{Float32, 1}(Float32[2.0, 4.0, 6.0]),)
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
    updates::TracedRArray{T,1},
) where {T,N}
    @assert length(updates) == size(scatter_indices, 1)
    @assert size(scatter_indices, 2) == N

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
                    slice_sizes=fill(Int64(1), N),
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

    (_, cond_fn_compiled, _, _, _, _, _, _, _) = Reactant.TracedUtils.make_mlir_fn(
        cond_fn,
        traced_args,
        (),
        string(gensym("cond_fn")),
        false;
        return_dialect=:stablehlo,
        no_args_in_result=true,
        do_transpose=false,
    )

    (_, body_fn_compiled, _, _, _, _, _, _, _) = Reactant.TracedUtils.make_mlir_fn(
        body_fn,
        traced_args,
        (),
        string(gensym("body_fn")),
        false;
        return_dialect=:stablehlo,
        no_args_in_result=true,
        do_transpose=false,
    )

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

    MLIR.IR.activate!(true_fn_body)
    tb_result = try
        for (i, arg) in enumerate(tb_linear_args)
            Reactant.TracedUtils.set_mlir_data!(
                arg, Reactant.TracedUtils.get_mlir_data(tb_traced_args[i])
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
        Reactant.TracedTrack;
        track_numbers=Number,
    )
    for i in 1:length(tb_linear_args)
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

    MLIR.IR.activate!(false_fn_body)
    fb_result = try
        for (i, arg) in enumerate(fb_linear_args)
            Reactant.TracedUtils.set_mlir_data!(
                arg, Reactant.TracedUtils.get_mlir_data(fb_traced_args[i])
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
        Reactant.TracedTrack;
        track_numbers=Number,
    )
    for i in 1:length(fb_linear_args)
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
                @show tr.mlir_data
                @show fr.mlir_data
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
        else
            Reactant.TracedUtils.set!(
                args, path[2:end], MLIR.IR.result(if_compiled, residx)
            )
        end
    end

    return corrected_traced_results
end

"""
    batch(
        inputs::Vector{<:Union{<:TracedRArray,<:MLIR.IR.Value}},
        output_types::Vector{<:MLIR.IR.Type},
        batch_shape::Vector{Int64};
        fn,
        location=mlir_stacktrace("batch", @__FILE__, @__LINE__),
    )

Generates a Reactant.MLIR.Dialects.enzyme.batch operation. It is recommended to use
`Ops.batch(f, args, batch_dims, result_dims)` or `Ops.elem_apply(f, args...)` instead
of calling this directly.

!!! warning

    This function batches the inputs based on the starting dimensions of the inputs. This
    aligns with the default ordering in Python frameworks like JAX and PyTorch, but is
    opposite to the default ordering in Julia.
"""
@noinline function batch(
    inputs::Vector{<:Union{<:TracedRArray,<:MLIR.IR.Value}},
    output_types::Vector{<:MLIR.IR.Type},
    batch_shape::Vector{Int64};
    fn,
    location=mlir_stacktrace("batch", @__FILE__, @__LINE__),
)
    op = MLIR.Dialects.enzyme.batch(
        [i isa TracedRArray ? i.mlir_data : i for i in inputs];
        outputs=output_types,
        fn=MLIR.IR.FlatSymbolRefAttribute(
            String(Reactant.TracedUtils.get_attribute_by_name(fn, "sym_name"))
        ),
        batch_shape=MLIR.IR.DenseArrayAttribute(batch_shape),
        location,
    )

    return [
        TracedRArray{MLIR.IR.julia_type(eltype(out_type)),ndims(out_type)}(
            (), MLIR.IR.result(op, i), size(out_type)
        ) for (i, out_type) in enumerate(output_types)
    ]
end

# This function assumes that the last dimension of each element is the batch dimension by
# default. This is the standard Julia ordering for batching. We permutedims the ordering to
# make sure the first dimension is the batch dimension when calling `batch_internal` below.
# XXX: Mutation inside a batched function is not supported yet (need to set the results
#      correctly)
@noinline function batch(f, args...; batch_dims=nothing, result_dims=nothing)
    batch_sizes = Int64[]
    batching_dims = if batch_dims === nothing
        fmap(args) do x
            tmp = ndims(x)
            push!(batch_sizes, size(x, tmp))
            return tmp
        end
    else
        fmap(args, batch_dims) do x, dim
            dim !== nothing && push!(batch_sizes, size(x, dim))
            @assert dim isa Integer || dim === nothing
            dim
        end
    end

    batch_sizes_no_ones = filter(x -> x != 1, batch_sizes)
    @assert allequal(batch_sizes) "batching dimensions must be equal"
    B = length(batch_sizes_no_ones) == 0 ? 1 : first(batch_sizes_no_ones)

    corrected_args = fmap(args, batching_dims) do arg, dim
        if dim === nothing # repeat the input along dim=0
            return broadcast_in_dim(arg, collect(1:ndims(arg)) .+ 1, Int64[B, size(arg)...])
        end
        if size(arg, dim) == 1 && size(arg, dim) != B # If batch_dim is 1, then expand that dim
            new_dims = collect(Int64, size(arg))
            new_dims[dim] = B
            arg = broadcast_in_dim(arg, collect(1:ndims(arg)), new_dims)
        end
        order = collect(Int64, 1:ndims(arg))
        order[dim] = 1
        order[1] = dim
        return permutedims(arg, order) # Ensure batch dim is moved to the first position
    end

    results = batch_internal(f, corrected_args...)

    if result_dims === nothing
        return fmap(results) do result
            order = Int64[2:ndims(result)..., 1]
            return permutedims(result, order)
        end
    end

    return fmap(results, result_dims) do result, dim
        order = collect(Int64, 1:ndims(result))
        order[dim] = 1
        order[1] = dim
        return permutedims(result, order)
    end
end

"""
    elem_apply(f, args...)

This is equivalent to `f.(args...)` but generates optimized code using
Reactant.MLIR.Dialects.enzyme.batch.
"""
@noinline function elem_apply(f, args::Vararg)
    return batch_internal(f, args...; batchmode=Reactant.BatchScalar)
end

@noinline function elem_apply(
    ::Type{T}, x::TracedRArray{T}
) where {T<:Reactant.ReactantPrimitive}
    return x
end

@noinline function elem_apply(
    ::Type{T}, x::TracedRArray
) where {T<:Reactant.ReactantPrimitive}
    # Special Path to prevent going down a despecialized path
    return elem_apply(Reactant.TracedUtils.TypeCast{T}(), x)
end

@noinline function batch_internal(f, args::Vararg; batchmode=Reactant.BatchArray)
    @assert batchmode != Reactant.BatchNone

    if batchmode == Reactant.BatchScalar
        if all(iszero ∘ ndims, args)
            scalar_args = map(args) do arg
                return Reactant.TracedUtils.promote_to(
                    TracedRNumber{Reactant.unwrapped_eltype(arg)}, arg
                )
            end
            return f(scalar_args...)
        end
    end

    fnwrap, func2, _, result, seen_args, _, linear_args, _, linear_results = Reactant.TracedUtils.make_mlir_fn(
        f,
        args,
        (),
        string(f) * (batchmode == Reactant.BatchArray ? "_batch" : "_broadcast_scalar"),
        false;
        batchmode,
        no_args_in_result=batchmode == Reactant.BatchScalar,
        do_transpose=false,
    )

    if batchmode == Reactant.BatchArray
        batch_sizes = [size(k, 1) for k in keys(seen_args) if k isa Reactant.TracedType]
        @assert allequal(batch_sizes) "batching dimensions must be equal"
        B = first(batch_sizes)
    else
        input_shapes = [size(k) for k in keys(seen_args) if k isa Reactant.TracedType]
        @assert allequal(input_shapes) "input shapes are $(input_shapes)"
        output_shape = first(input_shapes)
    end

    batch_inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = Reactant.TracedUtils.get_argidx(a)
        if idx == 1 && fnwrap
            Reactant.TracedUtils.push_val!(batch_inputs, f, path[3:end])
        else
            fnwrap && (idx -= 1)
            Reactant.TracedUtils.push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    res = batch(
        batch_inputs,
        [
            MLIR.IR.TensorType(
                batchmode == Reactant.BatchArray ? (B, size(arg)...) : output_shape,
                MLIR.IR.Type(Reactant.unwrapped_eltype(arg)),
            ) for arg in linear_results
        ],
        batchmode == Reactant.BatchArray ? Int64[B] : collect(Int64, output_shape);
        fn=func2,
    )

    residx = 1
    for a in linear_results
        if Reactant.TracedUtils.has_residx(a)
            path = Reactant.TracedUtils.get_residx(a)
            Reactant.TracedUtils.set!(result, path[2:end], res[residx])
            residx += 1
        else
            idx, path = Reactant.TracedUtils.get_argidx(a)
            if idx == 1 && fnwrap
                Reactant.TracedUtils.set!(f, path[3:end], res[residx])
                residx += 1
            else
                fnwrap && (idx -= 1)
                Reactant.TracedUtils.set!(args[idx], path[3:end], res[residx])
                residx += 1
            end
        end
    end

    traced2_result = Reactant.make_tracer(
        Reactant.OrderedIdDict(),
        result,
        (),
        Reactant.TracedSetPath;
        tobatch=batchmode == Reactant.BatchArray ? (B,) : output_shape,
        batchmode,
    )
    func2.operation = MLIR.API.MlirOperation(C_NULL)

    return traced2_result
end

end # module Ops
