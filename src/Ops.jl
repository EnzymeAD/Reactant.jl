using .MLIR.Dialects: stablehlo, chlo

## stablehlo
# [x] abs
# [ ] add
# [ ] after_all
# [ ] all_gather
# [ ] all_reduce
# [ ] all_to_all
# [ ] and
# [ ] atan2
# [ ] batch_norm_grad
# [ ] batch_norm_inference
# [ ] batch_norm_training
# [ ] bitcast_convert
# [ ] broadcast_in_dim
# [ ] broadcast
# [ ] case
# [x] cbrt
# [x] ceil
# [ ] cholesky
# [ ] clamp
# [x] count_leading_zeros
# [ ] collective_broadcast
# [ ] collective_permute
# [ ] compare
# [ ] complex
# [ ] composite
# [ ] concatenate
# [ ] constant
# [ ] convert
# [ ] convolution
# [x] cosine
# [ ] create_token
# [ ] cross_replica_sum
# [ ] custom_call
# [ ] divide
# [ ] dot_general
# [ ] dot
# [ ] dynamic_broadcast_in_dim
# [ ] dynamic_conv
# [ ] dynamic_gather
# [ ] dynamic_iota
# [ ] dynamic_pad
# [ ] dynamic_reshape
# [ ] dynamic_slice
# [ ] dynamic_update_slice
# [ ] einsum
# [x] exponential
# [x] exponential_minus_one
# [ ] fft
# [ ] floor
# [ ] gather
# [ ] get_dimension_size
# [ ] get_tuple_element
# [ ] if_
# [x] imag
# [ ] infeed
# [ ] iota
# [ ] is_finite
# [x] log_plus_one
# [x] log
# [x] logistic
# [ ] map
# [ ] maximum
# [ ] minimum
# [ ] multiply
# [x] negate
# [x] not
# [ ] optimization_barrier
# [ ] or
# [ ] outfeed
# [ ] pad
# [ ] partition_id
# [x] popcnt
# [ ] power
# [ ] real_dynamic_slice
# [x] real
# [ ] recv
# [ ] reduce
# [ ] reduce_precision
# [ ] reduce_scatter
# [ ] reduce_window
# [ ] remainder
# [ ] replica_id
# [ ] reshape
# [ ] reverse
# [ ] rng_bit_generator
# [ ] rng
# [x] round_nearest_even
# [x] round_nearest_afz
# [x] rsqrt
# [ ] scatter
# [ ] select_and_scatter
# [ ] select
# [ ] send
# [ ] set_dimension_size
# [ ] shift_left
# [ ] shift_right_arithmetic
# [ ] shift_right_logical
# [x] sign
# [x] sine
# [ ] slice
# [ ] sort
# [x] sqrt
# [ ] subtract
# [x] tan
# [x] tanh
# [ ] torch_index_select
# [ ] transpose
# [ ] triangular_solve
# [ ] tuple
# [ ] unary_einsum
# [ ] uniform_dequantize
# [ ] uniform_quantize
# [ ] while_
# [ ] xor

## chlo
# [x] acos
# [x] acosh
# [ ] _asin_acos_kernel
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
# [ ] is_inf
# [ ] is_neg_inf
# [ ] is_pos_inf
# [x] lgamma
# [ ] next_after
# [ ] polygamma
# [x] sinh
# [x] tan
# [ ] top_k
# [x] zeta

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
