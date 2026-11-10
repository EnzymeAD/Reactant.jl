# Legalization of a module for the Metal backend (`Reactant.Accelerators.Metal`, the
# `jax-graft` PJRT plugin)
module Metal

using ..Reactant: Reactant, MLIR, TracedRArray
using ..Reactant.Ops: @opcall

"""
    legalize!(mod::MLIR.IR.Module)

Rewrite the ops of `mod` that XLA:GPU can't compile on Metal into the custom calls of the
`jax-graft` plugin. It runs after the optimization pipeline (in particular after
differentiation), in place of `lower-enzymexla-linalg` and `lower-enzymexla-lapack`:

  - `enzymexla.linalg.{lu,svd,tridiagonal_solve}` become Accelerate LAPACK calls
    (`metal\$lapack_{getrf,gesdd,gtsv}`), `Float32` only;
  - `stablehlo.fft` becomes one `metal\$fft` per axis (`Float32` / `ComplexF32`).

`stablehlo.cholesky` and `stablehlo.triangular_solve` are handled by the plugin itself.
"""
function legalize!(mod::MLIR.IR.Module)
    MLIR.IR.activate(mod)
    try
        legalize!(MLIR.IR.body(mod))
    finally
        MLIR.IR.deactivate(mod)
    end
    return mod
end

function legalize!(block::MLIR.IR.Block)
    for op in collect(block)
        for region in op, inner_block in region
            legalize!(inner_block)
        end
        name = MLIR.IR.name(op)
        haskey(LEGALIZATIONS, name) && replace_op!(LEGALIZATIONS[name], op)
    end
    return nothing
end

# Build the replacement of `op` with `Ops` in a detached block, then splice it in
function replace_op!(f::F, op::MLIR.IR.Operation) where {F}
    block = MLIR.IR.Block()
    MLIR.IR.activate(block)
    results = try
        f(op)
    finally
        MLIR.IR.deactivate(block)
    end
    for new_op in collect(block)
        MLIR.IR.move_before!(new_op, op)
    end
    for (i, res) in enumerate(results)
        MLIR.API.mlirValueReplaceAllUsesOfWith(MLIR.IR.result(op, i), res.mlir_data)
    end
    MLIR.IR.dispose(MLIR.IR.rmfromparent!(op))
    MLIR.IR.dispose(block)
    return nothing
end

operand(op, i) = TracedRArray(MLIR.IR.operand(op, i))
result_type(op, i) = MLIR.IR.type(MLIR.IR.result(op, i))
function is_used(op, i)
    return !MLIR.API.mlirOpOperandIsNull(
        MLIR.API.mlirValueGetFirstUse(MLIR.IR.result(op, i))
    )
end

function check_float32(x::TracedRArray, what)
    T = Reactant.unwrapped_eltype(x)
    T == Float32 || error("Metal: $(what) only supports Float32 (got $(T)).")
    return nothing
end

# LAPACK's layout: the matrix (last two dims) in column-major order, batch dims outermost
function lapack_layout(N)
    N < 2 && return default_layout(N)
    return [N - 2, N - 1, (N - 3):-1:0...]
end
default_layout(N) = collect(Int64, (N - 1):-1:0)

function layouts_attr(layouts)
    return MLIR.IR.Attribute([
        MLIR.IR.DenseIndexElementsAttribute(collect(Int64, l)) for l in layouts
    ])
end

# `aliases` are (result, operand) pairs; the result index is a tuple index only with several
# results
function aliases_attr(aliases, nresults)
    return parse(
        MLIR.IR.Attribute,
        "[" *
        join(
            [
                "#stablehlo.output_operand_alias<output_tuple_indices = \
                 [$(nresults > 1 ? o : "")], operand_index = $(i), \
                 operand_tuple_indices = []>" for (o, i) in aliases
            ],
            ", ",
        ) *
        "]",
    )
end

# A typed FFI custom call into the plugin
function ffi_call(
    target::String,
    operands::Vector{<:TracedRArray},
    result_types::Vector{MLIR.IR.Type};
    attributes=Dict{String,MLIR.IR.Attribute}(),
    operand_layouts=[default_layout(ndims(x)) for x in operands],
    result_layouts=[default_layout(ndims(t)) for t in result_types],
    aliases=(),
)
    op = MLIR.Dialects.stablehlo.custom_call(
        MLIR.IR.Value[x.mlir_data for x in operands];
        result_0=result_types,
        call_target_name=target,
        backend_config=MLIR.IR.Attribute(attributes),
        api_version=Int32(4),
        operand_layouts=layouts_attr(operand_layouts),
        result_layouts=layouts_attr(result_layouts),
        output_operand_aliases=if isempty(aliases)
            nothing
        else
            aliases_attr(aliases, length(result_types))
        end,
    )
    return [TracedRArray(MLIR.IR.result(op, i)) for i in 1:length(result_types)]
end

tensor_type(T, shape) = MLIR.IR.TensorType(collect(Int64, shape), MLIR.IR.Type(T))

function convert_to(x::TracedRArray{T,N}, type::MLIR.IR.Type) where {T,N}
    U = eltype(MLIR.IR.julia_type(type))
    return T == U ? x : @opcall(convert(TracedRArray{U,N}, x))
end

function zeros_like(type::MLIR.IR.Type)
    return @opcall fill(zero(eltype(MLIR.IR.julia_type(type))), collect(Int64, size(type)))
end

# LAPACK's `info` of `getrf`: the (1-based) index of the first zero on the diagonal of U
function getrf_info(lu::TracedRArray{T,N}) where {T,N}
    shape = collect(Int64, size(lu))
    row = @opcall iota(Int32, shape; iota_dimension=N - 1)
    col = @opcall iota(Int32, shape; iota_dimension=N)
    zero_diagonal = @opcall and(
        @opcall(compare(row, col; comparison_direction="EQ")),
        @opcall(compare(lu, @opcall(fill(zero(T), shape)); comparison_direction="EQ")),
    )
    none = typemax(Int32)
    index = @opcall select(
        zero_diagonal,
        @opcall(add(row, @opcall(fill(Int32(1), shape)))),
        @opcall(fill(none, shape)),
    )
    first_index = reduce_min(index, [N - 1, N])
    info_shape = collect(Int64, size(first_index))
    return @opcall select(
        @opcall(
            compare(first_index, @opcall(fill(none, info_shape)); comparison_direction="EQ")
        ),
        @opcall(fill(Int32(0), info_shape)),
        first_index,
    )
end

# `Ops.reduce` traces its reducer, which needs the tracing caches
function reduce_min(x::TracedRArray{T}, dims::Vector{Int}) where {T}
    scalar = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(T))
    body = MLIR.IR.Region()
    block = MLIR.IR.Block([scalar, scalar], [MLIR.IR.Location(), MLIR.IR.Location()])
    push!(body, block)
    MLIR.IR.activate(block)
    try
        res = MLIR.Dialects.stablehlo.minimum(
            MLIR.IR.argument(block, 1), MLIR.IR.argument(block, 2); result=scalar
        )
        MLIR.Dialects.stablehlo.return_([MLIR.IR.result(res)])
    finally
        MLIR.IR.deactivate(block)
    end
    op = MLIR.Dialects.stablehlo.reduce(
        [x.mlir_data],
        [@opcall(fill(typemax(T), Int64[])).mlir_data];
        result_0=[tensor_type(T, deleteat!(collect(Int64, size(x)), dims))],
        dimensions=MLIR.IR.DenseArrayAttribute(dims .- 1),
        body,
    )
    return TracedRArray(MLIR.IR.result(op))
end

function legalize_lu(op)
    x = operand(op, 1)
    check_float32(x, "LU factorization")
    N = ndims(x)
    m, n = size(x, N - 1), size(x, N)
    batch = size(x)[1:(N - 2)]
    lu, pivots, permutation = ffi_call(
        "metal\$lapack_getrf",
        [x],
        [
            MLIR.IR.type(x.mlir_data),
            tensor_type(Int32, (batch..., min(m, n))),
            tensor_type(Int32, (batch..., m)),
        ];
        operand_layouts=[lapack_layout(N)],
        result_layouts=[lapack_layout(N), default_layout(N - 1), default_layout(N - 1)],
        aliases=((0, 0),),
    )
    # LAPACK's 1-based pivots and permutation
    pivots = @opcall add(pivots, @opcall(fill(Int32(1), collect(Int64, size(pivots)))))
    permutation = @opcall add(
        permutation, @opcall(fill(Int32(1), collect(Int64, size(permutation))))
    )
    return (
        lu,
        convert_to(pivots, result_type(op, 2)),
        convert_to(permutation, result_type(op, 3)),
        convert_to(getrf_info(lu), result_type(op, 4)),
    )
end

function legalize_svd(op)
    x = operand(op, 1)
    check_float32(x, "SVD")
    N = ndims(x)
    U_type, S_type, Vt_type, info_type = (result_type(op, i) for i in 1:4)
    if is_used(op, 1) || is_used(op, 3)
        _, S, U, Vt = ffi_call(
            "metal\$lapack_gesdd",
            [x],
            [MLIR.IR.type(x.mlir_data), S_type, U_type, Vt_type];
            attributes=Dict("full_matrices" => MLIR.IR.getattr(op, "full")),
            operand_layouts=[lapack_layout(N)],
            result_layouts=[
                lapack_layout(N), default_layout(N - 1), lapack_layout(N), lapack_layout(N)
            ],
            aliases=((0, 0),),
        )
    else
        _, S = ffi_call(
            "metal\$lapack_gesdd_novec",
            [x],
            [MLIR.IR.type(x.mlir_data), S_type];
            operand_layouts=[lapack_layout(N)],
            result_layouts=[lapack_layout(N), default_layout(N - 1)],
            aliases=((0, 0),),
        )
        U, Vt = zeros_like(U_type), zeros_like(Vt_type)
    end
    # A failed factorization makes the outputs NaN
    return (U, S, Vt, zeros_like(info_type))
end

function legalize_tridiagonal_solve(op)
    dl, d, du, b = (operand(op, i) for i in 1:4)
    check_float32(b, "tridiagonal solve")
    N = ndims(b)
    (x,) = ffi_call(
        "metal\$lapack_gtsv",
        [dl, d, du, b],
        [MLIR.IR.type(b.mlir_data)];
        operand_layouts=[
            default_layout(N - 1),
            default_layout(N - 1),
            default_layout(N - 1),
            lapack_layout(N),
        ],
        result_layouts=[lapack_layout(N)],
        aliases=((0, 3),),
    )
    # A singular system makes the solution NaN
    return (x, zeros_like(result_type(op, 2)))
end

# `metal$fft` (`metal_pjrt/fft/fft_plan.h`): powers of two up to 2^24, other lengths up
# to 2^23 - 1, and a workspace that must have exactly `fft_workspace_bytes(n, rows)` bytes
const MAX_FFT = 1 << 24
const MAX_FFT_CHUNK_ELEMENTS = 1 << 24

function fft_supported(n)
    (n < 1 || n > MAX_FFT) && return false
    return n <= 4096 || ispow2(n) || nextpow(2, 2n - 1) <= MAX_FFT
end

function fft_workspace_bytes(n, rows)
    rows == 0 && return 0
    if n > 4096 && ispow2(n)
        width, per_row = n, 8n
    elseif n > 4096 || (n > 2048 && !is_13_smooth(n))
        width = nextpow(2, 2n - 1)
        per_row = 16width
    else
        return 0
    end
    return min(max(MAX_FFT_CHUNK_ELEMENTS ÷ width, 1), rows) * per_row
end

function is_13_smooth(n)
    for p in (2, 3, 5, 7, 11, 13)
        while n % p == 0
            n ÷= p
        end
    end
    return n == 1
end

# One `metal$fft` over the last axis of `x`: `kind` is "fft", "ifft", "rfft" or "irfft"
function metal_fft_last(x::TracedRArray{T,N}, kind, n) where {T,N}
    fft_supported(n) || error(
        "Metal: FFT of length $(n) is not supported: the kernels take powers of two up \
         to 2^24 and other lengths up to 2^23 - 1."
    )
    out_length = kind == "rfft" ? n ÷ 2 + 1 : n
    out_T = kind == "irfft" ? Float32 : ComplexF32
    rows = prod(size(x)[1:(N - 1)]; init=1)
    out, _ = ffi_call(
        "metal\$fft",
        [x],
        [
            tensor_type(out_T, (size(x)[1:(N - 1)]..., out_length)),
            tensor_type(UInt8, (fft_workspace_bytes(n, rows),)),
        ];
        attributes=Dict(
            "fft_type" => MLIR.IR.Attribute(kind), "n" => MLIR.IR.Attribute(Int64(n))
        ),
    )
    return out
end

function metal_fft_axis(x::TracedRArray{T,N}, axis, kind, n) where {T,N}
    axis == N && return metal_fft_last(x, kind, n)
    perm = collect(1:N)
    perm[axis], perm[N] = N, axis
    return @opcall transpose(metal_fft_last(@opcall(transpose(x, perm)), kind, n), perm)
end

function legalize_fft(op)
    x = operand(op, 1)
    T = Reactant.unwrapped_eltype(x)
    T in (Float32, ComplexF32) ||
        error("Metal: FFT only supports Float32 and ComplexF32 (got $(T)).")
    fft_type = only(match(r"fft_type (\w+)", string(MLIR.IR.getattr(op, "fft_type"))))
    fft_length = MLIR.IR.getattr(op, "fft_length")
    lengths = Int64[fft_length[i] for i in 0:(length(fft_length) - 1)]
    N = ndims(x)
    axes = (N - length(lengths) + 1):N
    if fft_type == "RFFT"
        x = metal_fft_axis(x, N, "rfft", lengths[end])
        for (axis, n) in zip(axes[1:(end - 1)], lengths[1:(end - 1)])
            x = metal_fft_axis(x, axis, "fft", n)
        end
    elseif fft_type == "IRFFT"
        for (axis, n) in zip(axes[1:(end - 1)], lengths[1:(end - 1)])
            x = metal_fft_axis(x, axis, "ifft", n)
        end
        x = metal_fft_axis(x, N, "irfft", lengths[end])
    else
        kind = fft_type == "FFT" ? "fft" : "ifft"
        for (axis, n) in zip(axes, lengths)
            x = metal_fft_axis(x, axis, kind, n)
        end
    end
    return (x,)
end

const LEGALIZATIONS = Dict{String,Any}(
    "enzymexla.linalg.lu" => legalize_lu,
    "enzymexla.linalg.svd" => legalize_svd,
    "enzymexla.linalg.tridiagonal_solve" => legalize_tridiagonal_solve,
    "stablehlo.fft" => legalize_fft,
)

end
