# This module reflects the HLO ops defined in the openxla/stablehlo repo (plus some extras).
# If you want to add some check or test, the StableHLO spec should be taken as the source of truth, not the Julia or Reactant semantics.
# Julia and Reactant semantics should be considered on the higher abstractions that use these ops.
module Ops
using ..MLIR: MLIR
using ..MLIR.Dialects: stablehlo, chlo, enzyme
using ..Reactant:
    Reactant,
    ConcreteRArray,
    ConcreteRNumber,
    TracedRArray,
    TracedRNumber,
    mlir_type,
    mlir_stacktrace

struct Token
    mlir_data::MLIR.IR.Value
end

# constant ops
function constant(
    x::DenseArray{T,N}; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T,N}
    value = MLIR.IR.DenseElementsAttribute(x)
    output = mlir_type(TracedRArray{T,N}, size(x))
    res = MLIR.IR.result(stablehlo.constant(; output, value, location))
    return TracedRArray{T,N}((), res, size(x))
end

function constant(x::ConcreteRArray; kwargs...)
    return stablehlo.constant(convert(Array, x); kwargs...)
end

function constant(
    x::T; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T<:Number}
    res = constant(fill(x); location)
    return TracedRNumber{T}((), res.mlir_data)
end

function constant(
    x::ConcreteRNumber{T}; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T}
    output = mlir_type(TracedRArray{T,0}, ())
    value = MLIR.IR.DenseElementsAttribute(fill(MLIR.IR.Attribute(convert(T, x)), output))
    res = MLIR.IR.result(stablehlo.constant(; output, value, location))
    return TracedRNumber{T,N}((), res)
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
        function $op(
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

        function $op(
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
        function $op(
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

        function $op(
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
        function $op(
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

        function $op(
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

function is_finite(
    x::TracedRArray{T,N}; location=mlir_stacktrace("is_finite", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.is_finite(
            x.mlir_data; y=mlir_type(TracedRArray{Bool,N}, size(x)), location
        ),
    )
    return TracedRArray{Bool,N}((), res, size(x))
end

function is_finite(
    x::TracedRNumber{T}; location=mlir_stacktrace("is_finite", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.is_finite(x.mlir_data; y=mlir_type(TracedRArray{Bool,0}, ()), location)
    )
    return TracedRNumber{Bool}((), res)
end

# fixes to default automated implementations
function abs(
    x::TracedRArray{Complex{T},N}; location=mlir_stacktrace("abs", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

function abs(
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

function reshape(
    x::TracedRArray{T,N},
    dims::Vector{Int};
    location=mlir_stacktrace("reshape", @__FILE__, @__LINE__),
) where {T,N}
    restype = mlir_type(TracedRArray{T,length(dims)}, dims)
    res = MLIR.IR.result(stablehlo.reshape(x.mlir_data; result_0=restype, location))
    result = TracedRArray{T,length(dims)}((), res, dims)
    # NOTE this last `transpose` is required for consistency with Julia's column-major order
    # do not remove, as it will be optimized away by the compiler
    return transpose(result, [length(dims):-1:1...])
end

function get_dimension_size(
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

function set_dimension_size(
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

function transpose(
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
function pad(
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

function slice(
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
function complex(
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

function complex(
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

function real(
    x::TracedRArray{Complex{T},N}; location=mlir_stacktrace("real", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.real(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

function real(
    x::TracedRNumber{Complex{T}}; location=mlir_stacktrace("real", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.real(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

function imag(
    x::TracedRArray{Complex{T},N}; location=mlir_stacktrace("imag", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.imag(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

function imag(
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

function fft(
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
        end
    elseif type == "IRFFT"
        @assert T <: Complex
        Tout = real(T)
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

function cholesky(
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

function clamp(
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

function clamp(min::T, x::TracedRArray{T,N}, max::T) where {T,N}
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

function dot_general(
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

function einsum(
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
function partition_id(; location=mlir_stacktrace("partition_id", @__FILE__, @__LINE__))
    res = MLIR.IR.result(stablehlo.partition_id(; location))
    return TracedRNumber{UInt32}((), res)
end

function replica_id(; location=mlir_stacktrace("replica_id", @__FILE__, @__LINE__))
    res = MLIR.IR.result(stablehlo.replica_id(; location))
    return TracedRNumber{UInt32}((), res)
end

function after_all(tokens...; location=mlir_stacktrace("after_all", @__FILE__, @__LINE__))
    tokens = [token.mlir_data for token in tokens]
    res = MLIR.IR.result(stablehlo.after_all(tokens; location))
    return Token(res)
end

function optimization_barrier(
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

function outfeed(
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

function send(
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

function recv(
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
# function broadcast_in_dim(
#     x::TracedRArray{T,N},
#     dims::Vector{Int};
#     location=mlir_stacktrace(
#         "broadcast_in_dim", @__FILE__, @__LINE__
#     ),
# ) where {T,N}
#     rsize = restype = MLIR.IR.TensorType([...], mlir_type(T)) # mlir_type(TracedRArray{T,N}, size(x))
#     res = MLIR.IR.result(
#         stablehlo.broadcast_in_dim(
#             x.mlir_data;
#             result_0=restype,
#             broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims),
#             location,
#         ),
#     )
#     return TracedRArray{T,N}((), res, size(x))
# end

# sorting ops
# TODO need to trace over `comparator`
# function sort(
#     x::TracedRArray{T,N};
#     comparator,
#     dimension=-1,
#     is_stable=false,
#     location=mlir_stacktrace("sort", @__FILE__, @__LINE__),
# ) where {T,N}
#     dimension = MLIR.IR.Attribute(dimension)
#     is_stable = MLIR.IR.Attribute(is_stable)
#     res = MLIR.IR.result(
#         stablehlo.sort(
#             x.mlir_data;
#             result=mlir_type(TracedRArray{T,N}, size(x)),
#             dimension,
#             is_stable,
#             location,
#         ),
#     )
#     return TracedRArray{T,N}((), res, size(x))
# end

function top_k(
    x::TracedRArray{T,N}, k; location=mlir_stacktrace("top_k", @__FILE__, @__LINE__)
) where {T,N}
    rsize = [size(x)[1:(end - 1)]..., k]
    values = mlir_type(TracedRArray{T,N}, rsize)
    indices = mlir_type(TracedRArray{Int32,N}, rsize)
    op = chlo.top_k(x.mlir_data; values, indices, k, location)
    return (;
        values=TracedRArray{T,N}((), MLIR.IR.result(op, 1), rsize),
        indices=TracedRArray{Int32,N}((), MLIR.IR.result(op, 2), rsize),
    )
end

function iota(
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

function reverse(
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
function rng_bit_generator(
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rng_bit_generator", @__FILE__, @__LINE__),
)
    output = MLIR.IR.TensorType(TracedRArray{UInt64,1}, shape)
    rng_algorithm = MLIR.API.stablehloRngAlgorithmAttrGet(MLIR.IR.context(), algorithm)
    op = stablehlo.rng_bit_generator(seed.mlir_data; output, rng_algorithm, location)
    return (;
        output_state=TracedRArray{UInt64,1}((), MLIR.IR.result(op, 1), MLIR.IR.size(seed)),
        output=TracedRArray{T,length(shape)}((), MLIR.IR.result(op, 2), shape),
    )
end

# functional ops
function return_(
    results::Union{TracedRArray,TracedRNumber}...;
    location=mlir_stacktrace("return_", @__FILE__, @__LINE__),
)
    return stablehlo.return_([x.mlir_data for x in results]; location)
end

# control flow ops
function select(
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

function select(
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
function compare(
    lhs::Union{TracedRArray{T},TracedRNumber{T}},
    rhs::Union{TracedRArray{T},TracedRNumber{T}};
    comparison_direction::String,
    compare_type=nothing,
    location=mlir_stacktrace("compare", @__FILE__, @__LINE__),
) where {T}
    @assert comparison_direction in ("EQ", "NE", "GE", "GT", "LE", "LT")
    @assert size(lhs) == size(rhs)
    if lhs isa TracedRNumber
        @assert rhs isa TracedRNumber
    else
        @assert rhs isa TracedRArray
    end

    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.compare(
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

end
