using .MLIR.Dialects: stablehlo, chlo

## stablehlo
# [x] abs
# [x] add
# [ ] after_all
# [ ] all_gather
# [ ] all_reduce
# [ ] all_to_all
# [x] and
# [x] atan2
# [ ] batch_norm_grad
# [ ] batch_norm_inference
# [ ] batch_norm_training
# [ ] bitcast_convert
# [ ] broadcast_in_dim
# [-] broadcast -> on deprecation process
# [ ] case
# [x] cbrt
# [x] ceil
# [ ] cholesky
# [ ] clamp
# [x] count_leading_zeros
# [ ] collective_broadcast
# [ ] collective_permute
# [ ] compare
# [x] complex
# [ ] composite
# [ ] concatenate
# [x] constant
# [ ] convert
# [ ] convolution
# [x] cosine
# [-] create_token -> on deprecation process
# [-] cross_replica_sum -> on deprecation process
# [ ] custom_call
# [x] divide
# [ ] dot_general
# [-] dot -> on deprecation process
# [ ] dynamic_broadcast_in_dim
# [ ] dynamic_conv
# [ ] dynamic_gather
# [ ] dynamic_iota
# [ ] dynamic_pad
# [ ] dynamic_reshape
# [ ] dynamic_slice
# [ ] dynamic_update_slice
# [ ] einsum -> on deprecation process, but used by Tenet
# [x] exponential
# [x] exponential_minus_one
# [ ] fft
# [x] floor
# [ ] gather
# [ ] get_dimension_size
# [-] get_tuple_element -> on deprecation process
# [ ] if_
# [x] imag
# [ ] infeed
# [ ] iota
# [x] is_finite
# [x] log_plus_one
# [x] log
# [x] logistic
# [-] map -> on deprecation process
# [x] maximum
# [x] minimum
# [x] multiply
# [x] negate
# [x] not
# [x] optimization_barrier
# [x] or
# [ ] outfeed
# [ ] pad
# [x] partition_id
# [x] popcnt
# [x] power
# [-] real_dynamic_slice -> on deprecation process
# [x] real
# [ ] recv
# [ ] reduce
# [ ] reduce_precision
# [ ] reduce_scatter
# [ ] reduce_window
# [x] remainder
# [x] replica_id
# [ ] reshape
# [ ] reverse
# [ ] rng_bit_generator
# [-] rng -> on deprecation process
# [x] round_nearest_even
# [x] round_nearest_afz
# [x] rsqrt
# [ ] scatter
# [ ] select_and_scatter
# [ ] select
# [ ] send
# [ ] set_dimension_size
# [x] shift_left
# [x] shift_right_arithmetic
# [x] shift_right_logical
# [x] sign
# [x] sine
# [ ] slice
# [ ] sort
# [x] sqrt
# [x] subtract
# [x] tan
# [x] tanh
# [-] torch_index_select -> on deprecation process
# [ ] transpose
# [ ] triangular_solve
# [-] tuple -> on deprecation process
# [ ] unary_einsum -> on deprecation process, but used by Tenet
# [ ] uniform_dequantize
# [ ] uniform_quantize
# [ ] while_
# [x] xor

## chlo
# [x] acos
# [x] acosh
# [-] _asin_acos_kernel --> should not be used directly
# [x] asin
# [x] asinh
# [x] atan
# [x] atanh
# [x] bessel_i1e
# [ ] broadcast_add
# [ ] broadcast_and
# [ ] broadcast_atan2
# [ ] broadcast_compare
# [ ] broadcast_complex
# [ ] broadcast_divide
# [ ] broadcast_maximum
# [ ] broadcast_minimum
# [ ] broadcast_multiply
# [ ] broadcast_next_after
# [ ] broadcast_or
# [ ] broadcast_polygamma
# [ ] broadcast_power
# [ ] broadcast_remainder
# [ ] broadcast_select
# [ ] broadcast_shift_left
# [ ] broadcast_shift_right_arithmetic
# [ ] broadcast_shift_right_logical
# [ ] broadcast_subtract
# [ ] broadcast_xor
# [ ] broadcast_zeta
# [x] conj
# [ ] constant_like
# [ ] constant
# [x] cosh
# [x] digamma
# [x] erf_inv
# [x] erf
# [x] erfc
# [x] is_inf
# [x] is_neg_inf
# [x] is_pos_inf
# [x] lgamma
# [x] next_after
# [x] polygamma
# [x] sinh
# [x] tan
# [ ] top_k
# [x] zeta

# zeroary ops
function stablehlo.constant(
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

function stablehlo.constant(x::ConcreteRArray; kwargs...)
    return stablehlo.constant(convert(Array, x); kwargs...)
end

function stablehlo.constant(
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
for op in [
    :(stablehlo.abs),
    :(stablehlo.cbrt),
    :(stablehlo.ceil),
    :(stablehlo.count_leading_zeros),
    :(stablehlo.cosine),
    :(stablehlo.exponential),
    :(stablehlo.exponential_minus_one),
    :(stablehlo.floor),
    :(stablehlo.imag),
    :(stablehlo.log),
    :(stablehlo.log_plus_one),
    :(stablehlo.logistic),
    :(stablehlo.negate),
    :(stablehlo.not),
    :(stablehlo.popcnt),
    :(stablehlo.real),
    :(stablehlo.round_nearest_afz),
    :(stablehlo.round_nearest_even),
    :(stablehlo.rsqrt),
    :(stablehlo.sign),
    :(stablehlo.sine),
    :(stablehlo.sqrt),
    :(stablehlo.tan),
    :(stablehlo.tanh),
    :(chlo.acos),
    :(chlo.acosh),
    :(chlo.asin),
    :(chlo.asinh),
    :(chlo.atan),
    :(chlo.atanh),
    :(chlo.bessel_i1e),
    :(chlo.conj),
    :(chlo.cosh),
    :(chlo.digamma),
    :(chlo.erf_inv),
    :(chlo.erf),
    :(chlo.erfc),
    :(chlo.lgamma),
    :(chlo.sinh),
    :(chlo.tan),
    :(chlo.zeta),
]
    @eval begin
        function $op(
            x::TracedRArray{T,N};
            location=MLIR.IR.Location(
                string($op), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T,N}
            res = MLIR.IR.result(
                $op(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
            )
            return TracedRArray{T,N}((), res, size(x))
        end

        function $op(
            x::TracedRNumber{T};
            location=MLIR.IR.Location(
                string($op), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T}
            res = MLIR.IR.result(
                $op(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
            )
            return TracedRNumber{T}((), res)
        end
    end
end

# binary elementwise ops
for op in [
    :(stablehlo.add),
    :(stablehlo.and),
    :(stablehlo.atan2),
    :(stablehlo.divide),
    :(stablehlo.maximum),
    :(stablehlo.minimum),
    :(stablehlo.multiply),
    :(stablehlo.or),
    :(stablehlo.power),
    :(stablehlo.remainder),
    :(stablehlo.shift_left),
    :(stablehlo.shift_right_arithmetic),
    :(stablehlo.shift_right_logical),
    :(stablehlo.subtract),
    :(stablehlo.xor),
    :(chlo.next_after),
    :(chlo.polygamma),
]
    @eval begin
        function $op(
            a::TracedRArray{T,N},
            b::TracedRArray{T,N};
            location=MLIR.IR.Location(
                string($op), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T,N}
            res = MLIR.IR.result(
                $op(
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
                string($op), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T}
            res = MLIR.IR.result(
                $op(
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
for op in [:(stablehlo.is_finite), :(chlo.is_inf), :(chlo.is_neg_inf), :(chlo.is_pos_inf)]
    @eval begin
        function $op(
            x::TracedRArray{T,N};
            location=MLIR.IR.Location(
                string($op), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T,N}
            res = MLIR.IR.result(
                $op(x.mlir_data; result=mlir_type(TracedRArray{Bool,N}, size(x)), location)
            )
            return TracedRArray{Bool,N}((), res, size(x))
        end

        function $op(
            x::TracedRNumber{T};
            location=MLIR.IR.Location(
                string($op), MLIR.IR.Location(@__FILE__, @__LINE__, 0)
            ),
        ) where {T}
            res = MLIR.IR.result(
                $op(x.mlir_data; result=mlir_type(TracedRArray{Bool,0}, ()), location)
            )
            return TracedRNumber{Bool}((), res)
        end
    end
end

# fixes to default automated implementations
function stablehlo.abs(
    x::TracedRArray{Complex{T},N};
    location=MLIR.IR.Location("stablehlo.abs", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

function stablehlo.abs(
    x::TracedRNumber{Complex{T}};
    location=MLIR.IR.Location("stablehlo.abs", MLIR.IR.Location(@__FILE__, @__LINE__, 0)),
) where {T}
    res = MLIR.IR.result(
        stablehlo.abs(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

# miscelaneous
function stablehlo.complex(
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

function stablehlo.complex(
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

function stablehlo.cholesky(
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

function stablehlo.clamp(
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

# paralell ops
function stablehlo.partition_id(;
    location=MLIR.IR.Location(
        "stablehlo.partition_id", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    res = MLIR.IR.result(stablehlo.partition_id(; location))
    return TracedRNumber{UInt32}((), res)
end

function stablehlo.replica_id(;
    location=MLIR.IR.Location(
        "stablehlo.replica_id", MLIR.IR.Location(@__FILE__, @__LINE__, 0)
    ),
)
    res = MLIR.IR.result(stablehlo.replica_id(; location))
    return TracedRNumber{UInt32}((), res)
end

function stablehlo.optimization_barrier(
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
