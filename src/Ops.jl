module Ops
using ..MLIR: MLIR
using ..MLIR.Dialects: stablehlo, chlo, enzyme
using ..Reactant: ConcreteRArray, ConcreteRNumber, TracedRArray, TracedRNumber

struct Token
    mlir_data::MLIR.IR.Value
end

@enum RngAlgorithm begin
    RngDefault = 0
    RngThreeFry = 1
    RngPhilox = 2
end

# constant ops
function constant(
    x::DenseArray{T,N};
    location=MLIR.IR.Location(
        "stablehlo.constant", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
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
    x::ConcreteRNumber{T};
    location=MLIR.IR.Location(
        "stablehlo.constant", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T}
    output = mlir_type(TracedRArray{T,0}, ())
    value = MLIR.IR.DenseElementsAttribute(fill(MLIR.IR.Attribute(convert(T, x)), output))
    res = MLIR.IR.result(stablehlo.constant(; output, value, location))
    return TracedRNumber{T,N}((), res)
end

# unary elementwise ops
for (dialect, op) in [
    (stablehlo, :abs),
    (stablehlo, :cbrt),
    (stablehlo, :ceil),
    (stablehlo, :count_leading_zeros),
    (stablehlo, :cosine),
    (stablehlo, :exponential),
    (stablehlo, :exponential_minus_one),
    (stablehlo, :floor),
    (stablehlo, :imag),
    (stablehlo, :log),
    (stablehlo, :log_plus_one),
    (stablehlo, :logistic),
    (stablehlo, :negate),
    (stablehlo, :not),
    (stablehlo, :popcnt),
    (stablehlo, :real),
    (stablehlo, :round_nearest_afz),
    (stablehlo, :round_nearest_even),
    (stablehlo, :rsqrt),
    (stablehlo, :sign),
    (stablehlo, :sine),
    (stablehlo, :sqrt),
    (stablehlo, :tan),
    (stablehlo, :tanh),
    (chlo, :acos),
    (chlo, :acosh),
    (chlo, :asin),
    (chlo, :asinh),
    (chlo, :atan),
    (chlo, :atanh),
    (chlo, :bessel_i1e),
    (chlo, :conj),
    (chlo, :cosh),
    (chlo, :digamma),
    (chlo, :erf_inv),
    (chlo, :erf),
    (chlo, :erfc),
    (chlo, :lgamma),
    (chlo, :sinh),
    # (chlo, :tan),
    (chlo, :zeta),
]
    @eval begin
        function $op(
            x::TracedRArray{T,N};
            location=MLIR.IR.Location(
                string("$dialect.$op"), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T,N}
            res = MLIR.IR.result(
                $dialect.$op(
                    x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location
                ),
            )
            return TracedRArray{T,N}((), res, size(x))
        end

        function $op(
            x::TracedRNumber{T};
            location=MLIR.IR.Location(
                string("$dialect.$op"), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T}
            res = MLIR.IR.result(
                $dialect.$op(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
            )
            return TracedRNumber{T}((), res)
        end
    end
end

# binary elementwise ops
for (dialect, op) in [
    (stablehlo, :add),
    (stablehlo, :and),
    (stablehlo, :atan2),
    (stablehlo, :divide),
    (stablehlo, :maximum),
    (stablehlo, :minimum),
    (stablehlo, :multiply),
    (stablehlo, :or),
    (stablehlo, :power),
    (stablehlo, :remainder),
    (stablehlo, :shift_left),
    (stablehlo, :shift_right_arithmetic),
    (stablehlo, :shift_right_logical),
    (stablehlo, :subtract),
    (stablehlo, :xor),
    (chlo, :next_after),
    (chlo, :polygamma),
]
    @eval begin
        function $op(
            a::TracedRArray{T,N},
            b::TracedRArray{T,N};
            location=MLIR.IR.Location(
                string("$dialect.$op"), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T,N}
            res = MLIR.IR.result(
                $dialect.$op(
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
            location=MLIR.IR.Location(
                string("$dialect.$op"), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T}
            res = MLIR.IR.result(
                $dialect.$op(
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
    [(stablehlo, :is_finite), (chlo, :is_inf), (chlo, :is_neg_inf), (chlo, :is_pos_inf)]
    @eval begin
        function $op(
            x::TracedRArray{T,N};
            location=MLIR.IR.Location(
                string("$dialect.$op"), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T,N}
            res = MLIR.IR.result(
                $dialect.$op(
                    x.mlir_data; result=mlir_type(TracedRArray{Bool,N}, size(x)), location
                ),
            )
            return TracedRArray{Bool,N}((), res, size(x))
        end

        function $op(
            x::TracedRNumber{T};
            location=MLIR.IR.Location(
                string("$dialect.$op"), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T}
            res = MLIR.IR.result(
                $dialect.$op(
                    x.mlir_data; result=mlir_type(TracedRArray{Bool,0}, ()), location
                ),
            )
            return TracedRNumber{Bool}((), res)
        end
    end
end

# fixes to default automated implementations
function abs(
    x::TracedRArray{Complex{T},N};
    location=MLIR.IR.Location("stablehlo.abs", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

function abs(
    x::TracedRNumber{Complex{T}};
    location=MLIR.IR.Location("stablehlo.abs", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
) where {T}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

# shape ops
function reshape(x::TracedRArray, dims...; kwargs...)
    return stablehlo.reshape(x, collect(dims); kwargs...)
end

function reshape(
    x::TracedRArray{T,N},
    dims::Vector{Int};
    location=MLIR.IR.Location(
        "stablehlo.reshape", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T,N}
    restype = MLIR.IR.TensorType(dims, mlir_type(T))
    res = MLIR.IR.result(stablehlo.reshape(x.mlir_data; result_0=restype, location))
    return TracedRArray{T,N}((), res, dims)
end

function get_dimension_size(
    x::TracedRArray{T,N},
    dim;
    location=MLIR.IR.Location(
        "stablehlo.get_dimension_size", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T,N}
    dimension = MLIR.IR.Attribute(dim)
    res = MLIR.IR.result(
        stablehlo.get_dimension_size(
            x.mlir_data; result=mlir_type(TracedRNumber{Int}, ()), dimension, location
        ),
    )
    return TracedRNumber{Int}((), res)
end

function set_dimension_size(
    x::TracedRArray{T,N},
    size::TracedRNumber{Int},
    dim::Int;
    location=MLIR.IR.Location(
        "stablehlo.set_dimension_size", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T,N}
    dimension = MLIR.IR.Attribute(dim)
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
    location=MLIR.IR.Location(
        "stablehlo.transpose", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T,N}
    rsize = permute!(size(x), permutation)
    result = mlir_type(TracedRArray{T,N}, rsize)
    permutation = MLIR.IR.DenseArrayAttribute(permutation)
    res = MLIR.IR.result(stablehlo.transpose(x.mlir_data; result, permutation, location))
    return TracedRArray{T,N}((), res, rsize)
end

# numerics
function complex(
    real::TracedRArray{T,N},
    imag::TracedRArray{T,N};
    location=MLIR.IR.Location(
        "stablehlo.complex", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.complex(
            real.mlir_data,
            imag.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(real)),
            location,
        ),
    )
    return TracedRArray{Complex{T},N}((), res, size(real))
end

function complex(
    real::TracedRNumber{T},
    imag::TracedRNumber{T};
    location=MLIR.IR.Location(
        "stablehlo.complex", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T}
    res = MLIR.IR.result(
        stablehlo.complex(
            real.mlir_data,
            imag.mlir_data;
            result=mlir_type(TracedRArray{T,0}, ()),
            location,
        ),
    )
    return TracedRNumber{Complex{T}}((), res)
end

# function bitcast_convert(
#     ::Type{TracedRArray{U,N}},
#     x::TracedRArray{T,N};
#     location=MLIR.IR.Location(
#         "stablehlo.bitcast_convert", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
#     ),
# ) where {T,N}
#     res = MLIR.IR.result(
#         stablehlo.bitcast_convert(
#             x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location
#         ),
#     )
#     return TracedRArray{T,N}((), res, size(x))
# end

function cholesky(
    x::TracedRArray{T,N};
    lower::Bool=false,
    location=MLIR.IR.Location(
        "stablehlo.cholesky", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
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
    min::TracedRArray{T,N},
    x::TracedRArray{T,N},
    max::TracedRArray{T,N};
    location=MLIR.IR.Location("stablehlo.clamp", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
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
#     location=MLIR.IR.Location(
#         "stablehlo.convolution", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
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

# function dot_general(
#     lhs::TracedRArray{T,N},
#     rhs::TracedRArray{T,N};
#     dimension_numbers,
#     lhs_contracting_dimensions,
#     rhs_contracting_dimensions,
#     result_permutation,
#     location=MLIR.IR.Location(
#         "stablehlo.dot_general", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
#     ),
# ) where {T,N}
#     res = MLIR.IR.result(
#         stablehlo.dot_general(
#             lhs.mlir_data,
#             rhs.mlir_data;
#             result=mlir_type(TracedRArray{T,N}, ...), # TODO size of result
#             dimension_numbers,
#             lhs_contracting_dimensions,
#             rhs_contracting_dimensions,
#             result_permutation,
#             location,
#         ),
#     )
#     return TracedRArray{T,N}((), res, size(lhs))
# end

function einsum(
    lhs::TracedRArray{T},
    rhs::TracedRArray{T};
    equation::String,
    location=MLIR.IR.Location(
        "stablehlo.einsum", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T}
    res = MLIR.IR.result(
        stablehlo.einsum(
            lhs.mlir_data,
            rhs.mlir_data;
            einsum_config=MLIR.IR.Attribute(equation),
            location,
        ),
    )
    # computing the result size is not trivial
    rsize = size(res)
    return TracedRArray{T,length(rsize)}((), res, rsize)
end

function unary_einsum(
    x::Union{TracedRNumber,TracedRArray};
    equation::String,
    location=MLIR.IR.Location(
        "stablehlo.unary_einsum", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    res = MLIR.IR.result(
        stablehlo.unary_einsum(
            x.mlir_data; einsum_config=MLIR.IR.Attribute(equation), location
        ),
    )
    # computing the result size is not trivial
    return TracedRArray{Float64,1}((), res, (1,))
end

# paralell ops
function partition_id(;
    location=MLIR.IR.Location(
        "stablehlo.partition_id", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    res = MLIR.IR.result(stablehlo.partition_id(; location))
    return TracedRNumber{UInt32}((), res)
end

function replica_id(;
    location=MLIR.IR.Location(
        "stablehlo.replica_id", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    res = MLIR.IR.result(stablehlo.replica_id(; location))
    return TracedRNumber{UInt32}((), res)
end

function optimization_barrier(
    operands::Union{TracedRNumber,TracedRArray}...;
    location=MLIR.IR.Location(
        "stablehlo.optimization_barrier", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
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
    location=MLIR.IR.Location(
        "stablehlo.outfeed", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    values = [operand.mlir_data for operand in operands]
    outfeed_config = MLIR.IR.Attribute(config)
    res = MLIR.IR.result(
        stablehlo.outfeed(values, token.mlir_data; outfeed_config, location)
    )
    return Token(res)
end

# function send(
#     operands::Union{TracedRNumber,TracedRArray}...;
#     token,
#     channel_id,
#     channel_type,
#     location=MLIR.IR.Location("stablehlo.send", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
# )
#     values = [operand.mlir_data for operand in operands]
#     channel_handle = ... # MLIR.IR.Attribute(channel_id)
#     res = MLIR.IR.result(stablehlo.send(values, token.mlir_data; send_config, location))
#     return Token(res)
# end

# broadcast ops
# function broadcast_in_dim(
#     x::TracedRArray{T,N},
#     dims::Vector{Int};
#     location=MLIR.IR.Location(
#         "stablehlo.broadcast_in_dim", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
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
#     location=MLIR.IR.Location("stablehlo.sort", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
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

function chlo.top_k(
    x::TracedRArray{T,N},
    k;
    location=MLIR.IR.Location("chlo.top_k", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
) where {T,N}
    rsize = [size(x)[1:(end - 1)]..., k]
    values = MLIR.IR.TensorType(rsize, mlir_type(T))
    indices = MLIR.IR.TensorType(rsize, mlir_type(Int))
    op = chlo.top_k(x.mlir_data; values, indices, location)
    return (;
        values=TracedRArray{T,N}((), MLIR.IR.result(op, 1), rsize),
        indices=TracedRArray{Int,N}((), MLIR.IR.result(op, 2), rsize),
    )
end

function iota(
    T::Type,
    shape::Vector{Int};
    iota_dimension,
    location=MLIR.IR.Location("stablehlo.iota", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
)
    N = length(shape)
    output = mlir_type(TracedRArray{T,N}, shape)
    iota_dimension = MLIR.IR.Attribute(iota_dimension)
    res = MLIR.IR.result(stablehlo.iota(; output, iota_dimension, location))
    return TracedRArray{T,N}((), res, shape)
end

function reverse(
    x::TracedRArray{T,N};
    dimensions,
    location=MLIR.IR.Location(
        "stablehlo.reverse", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.reverse(
            x.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(x)),
            dimensions=MLIR.IR.DenseArrayAttribute(dimensions),
            location,
        ),
    )
    return TracedRArray{T,N}((), res, size(x))
end

# random ops
function rng_bit_generator(
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm=RngDefault,
    location=MLIR.IR.Location(
        "stablehlo.rng_bit_generator", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    output = MLIR.IR.TensorType(TracedRArray{UInt64,1}, shape)
    rng_algorithm = MLIR.IR.Attribute(algorithm)
    op = stablehlo.rng_bit_generator(seed.mlir_data; output, rng_algorithm, location)
    return (;
        output_state=TracedRArray{UInt64,1}((), MLIR.IR.result(op, 1), MLIR.IR.size(seed)),
        output=TracedRArray{T,length(shape)}((), MLIR.IR.result(op, 2), shape),
    )
end

# functional ops
function return_(
    results::Union{TracedRArray,TracedRNumber}...;
    location=MLIR.IR.Location(
        "stablehlo.return_", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    return stablehlo.return_([x.mlir_data for x in results]; location)
end

end
