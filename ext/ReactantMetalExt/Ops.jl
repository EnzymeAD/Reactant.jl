"""
    handle(::Val{opname}, graph::MPSGraph, operands::Tuple, attributes::NamedTuple, results::Tuple[, regions::Tuple, vars::Dict])

Add to `graph` the nodes that compute the MLIR op `opname` (e.g. `Symbol("stablehlo.add")`) and return the tensors of
its results.

- `operands`: `MPSGraphTensor`s, which like MLIR tensors are row-major, so they have their dimensions.
- `attributes`: the op attributes, decoded by `decode_attribute`. Dense elements are Julia arrays with reversed
  dimensions, i.e. their memory is in the row-major order of MLIR.
- `results`: the type of each result, as `(; eltype, size)` with `size` in MLIR order.
- `regions`: the block of each region, for ops with regions. For `func.call` and `stablehlo.composite`, the block of
  the function they call.
- `vars`: the tensors of the MLIR values, for ops with regions, which are translated by `codegen!(graph, block, vars)`.
  Like in MLIR, the ops of a region can use the values of the enclosing blocks.

Like MLIR values, the return value depends on the number of results: it is the tensor itself if there is one, and it
must be a tuple with one tensor per result if there are several. Their shapes must be exactly the ones in `results`,
since `execute` writes the outputs into buffers of those shapes.
"""
function handle end

# permute the dimensions of `a`: dimension `i` of the result is dimension `perm[i]` of `a` (0-based, like in MLIR)
function permute(graph, a, perm)
    perm == 0:(length(perm) - 1) && return a
    return @objc [
        graph::id{MPSGraph} transposeTensor:(a::id{MPSGraphTensor})
        permutation:(NSArray(NSNumber.(perm))::id{NSArray})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# name of the op of a region that only combines its 2 arguments (e.g. `stablehlo.add` for the body of a sum).
# MPSGraph can't run a region for each element, so the ops that take one are mapped to the node that does the same
function reducer(block)
    ops = collect(block)
    if length(ops) == 2 && IR.nargs(block) == 2 && IR.noperands(ops[1]) == 2
        args = Set(IR.argument(block, i).ref.ptr for i in 1:2)
        operands = Set(IR.operand(ops[1], i).ref.ptr for i in 1:2)
        returned = [IR.operand(ops[2], i).ref.ptr for i in 1:IR.noperands(ops[2])]
        if args == operands && returned == [IR.result(ops[1], 1).ref.ptr]
            return IR.name(ops[1])
        end
    end
    throw(unimplemented("regions with several ops are not supported by `MetalClient`"))
end

# element-wise unary ops that are a node of the graph
for (op, selector) in (
    ("ceil", :ceilWithTensor),
    ("cosine", :cosWithTensor),
    ("exp2", :exponentBase2WithTensor),
    ("exponential", :exponentWithTensor),
    ("floor", :floorWithTensor),
    ("imag", :imaginaryPartOfTensor),
    ("is_finite", :isFiniteWithTensor),
    ("log", :logarithmWithTensor),
    ("log2", :logarithmBase2WithTensor),
    ("logistic", :sigmoidWithTensor),
    ("negate", :negativeWithTensor),
    ("popcnt", :bitwisePopulationCountWithTensor),
    ("real", :realPartOfTensor),
    ("round_nearest_even", :rintWithTensor),
    ("rsqrt", :reciprocalSquareRootWithTensor),
    ("sine", :sinWithTensor),
    ("sqrt", :squareRootWithTensor),
    ("tan", :tanWithTensor),
    ("tanh", :tanhWithTensor),
)
    @eval function handle(::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a,), _, _)
        return @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor}) name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

function handle(::Val{Symbol("stablehlo.abs")}, graph::MPSGraph, (a,), _, _)
    y = @objc [
        graph::id{MPSGraph} absoluteWithTensor:(a::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor

    # MPSGraphs's `absolute` kernel returns complex on complex operands
    MPS.jl_mps_to_typ[a.dataType] <: Complex || return y
    return @objc [
        graph::id{MPSGraph} realPartOfTensor:(y::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# element-wise binary ops that are a node of the graph
for (op, selector) in (
    ("add", :additionWithPrimaryTensor),
    ("atan2", :atan2WithPrimaryTensor),
    ("divide", :divisionWithPrimaryTensor),
    ("multiply", :multiplicationWithPrimaryTensor),
    ("subtract", :subtractionWithPrimaryTensor),
)
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a, b), _, _
    )
        return @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

function handle(::Val{Symbol("stablehlo.complex")}, graph::MPSGraph, (a, b), _, _)
    return @objc [
        graph::id{MPSGraph} complexTensorWithRealTensor:(a::id{MPSGraphTensor})
        imaginaryTensor:(b::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

function handle(::Val{Symbol("stablehlo.select")}, graph::MPSGraph, (x, t, f), _, _)
    return @objc [
        graph::id{MPSGraph} selectWithPredicateTensor:(x::id{MPSGraphTensor})
        truePredicateTensor:(t::id{MPSGraphTensor})
        falsePredicateTensor:(f::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# like in StableHLO, the result is NaN if any operand is. MPSGraph only propagates NaN on floats
for (op, selector, nan_selector) in (
    ("maximum", :maximumWithPrimaryTensor, :maximumWithNaNPropagationWithPrimaryTensor),
    ("minimum", :minimumWithPrimaryTensor, :minimumWithNaNPropagationWithPrimaryTensor),
)
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a, b), _, (result,)
    )
        if result.eltype <: AbstractFloat
            return @objc [
                graph::id{MPSGraph} ($nan_selector):(a::id{MPSGraphTensor})
                secondaryTensor:(b::id{MPSGraphTensor})
                name:(nil::id{NSString})
            ]::MPSGraphTensor
        end
        return @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

# StableHLO uses the same op for booleans and integers, but MPSGraph has logical ops for the former and bitwise ops for
# the latter
for (op, logical_selector, bitwise_selector) in (
    ("and", :logicalANDWithPrimaryTensor, :bitwiseANDWithPrimaryTensor),
    ("or", :logicalORWithPrimaryTensor, :bitwiseORWithPrimaryTensor),
    ("xor", :logicalXORWithPrimaryTensor, :bitwiseXORWithPrimaryTensor),
)
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a, b), _, (result,)
    )
        if result.eltype == Bool
            return @objc [
                graph::id{MPSGraph} ($logical_selector):(a::id{MPSGraphTensor})
                secondaryTensor:(b::id{MPSGraphTensor})
                name:(nil::id{NSString})
            ]::MPSGraphTensor
        end
        return @objc [
            graph::id{MPSGraph} ($bitwise_selector):(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

function handle(::Val{Symbol("stablehlo.not")}, graph::MPSGraph, (a,), _, (result,))
    if result.eltype == Bool
        return @objc [
            graph::id{MPSGraph} notWithTensor:(a::id{MPSGraphTensor}) name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
    return @objc [
        graph::id{MPSGraph} bitwiseNOTWithTensor:(a::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# `power` goes through `exp` and `log`, so its result is refined by a Newton step, `y + (x / y^2 - y) / 3`.
# `x / y^2` is only not finite for ±0, ±Inf and NaN, which are their own cube root.
# Tested within 2 ulp for `Float32` and 0.5 ulp for `Float16`, except for `Float32` subnormals: Metal flushes them to
# zero, so `y` is 0 and the result is `x` itself
function handle(::Val{Symbol("stablehlo.cbrt")}, graph::MPSGraph, (x,), _, (result,))
    T = result.eltype
    T <: AbstractFloat ||
        throw(unimplemented("`stablehlo.cbrt` of `$T` is not supported by `MetalClient`"))

    sgn = @objc [
        graph::id{MPSGraph} signWithTensor:(x::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor
    absx = handle(Val(Symbol("stablehlo.abs")), graph, (x,), (;), ())
    third = MPSGraphs.constantWithScalar(graph, 1 / 3, T)
    y = @objc [
        graph::id{MPSGraph} powerWithPrimaryTensor:(absx::id{MPSGraphTensor})
        secondaryTensor:(third::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    y = handle(Val(Symbol("stablehlo.multiply")), graph, (sgn, y), (;), ())

    y2 = handle(Val(Symbol("stablehlo.multiply")), graph, (y, y), (;), ())
    q = handle(Val(Symbol("stablehlo.divide")), graph, (x, y2), (;), ())
    step = handle(Val(Symbol("stablehlo.subtract")), graph, (q, y), (;), ())
    three = MPSGraphs.constantWithScalar(graph, 3, T)
    step = handle(Val(Symbol("stablehlo.divide")), graph, (step, three), (;), ())
    refined = handle(Val(Symbol("stablehlo.add")), graph, (y, step), (;), ())
    finite = handle(Val(Symbol("stablehlo.is_finite")), graph, (q,), (;), ())
    return handle(Val(Symbol("stablehlo.select")), graph, (finite, refined, x), (;), ())
end

# the bits below the highest set one are set, so that the leading zeros are the bits that are still not set. Negative
# integers become all ones with both arithmetic and logical shifts
function handle(
    ::Val{Symbol("stablehlo.count_leading_zeros")}, graph::MPSGraph, (x,), _, (result,)
)
    T = result.eltype
    nbits = 8 * sizeof(T)
    k = 1
    while k < nbits
        shift = MPSGraphs.constantWithScalar(graph, k, T)
        shifted = @objc [
            graph::id{MPSGraph} bitwiseRightShiftWithPrimaryTensor:(x::id{MPSGraphTensor})
            secondaryTensor:(shift::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        x = @objc [
            graph::id{MPSGraph} bitwiseORWithPrimaryTensor:(x::id{MPSGraphTensor})
            secondaryTensor:(shifted::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        k *= 2
    end
    nset = handle(Val(Symbol("stablehlo.popcnt")), graph, (x,), (;), ())
    width = MPSGraphs.constantWithScalar(graph, nbits, T)
    return handle(Val(Symbol("stablehlo.subtract")), graph, (width, nset), (;), ())
end

# Kahan's `(u - 1) * x / log(u)` with `u = exp(x)`. The plain `exp(x) - 1` loses `x` near 0, since `u` is rounded to
# the spacing of the floats around 1 (2^-23 for `Float32`): its relative error is about 6e-8 / |x| (e.g. 2e-1 for
# `x = 1e-7`), and it returns 0 for |x| below 6e-8. This formula cancels the rounding error of `u`, for 12% (`Float32`)
# to 27% (`Float16`) more runtime than `exp(x) - 1` alone. `x / log(u)` is computed first, since it is close to 1 while
# `(u - 1) * x` may overflow.
# Tested within 4.7 ulp for `Float32` and 0.5 ulp for `Float16`. The error comes from `exp` and `log` of MPSGraph, so
# away from 0 the plain `exp(x) - 1` is more precise (2.8 ulp for `|x| >= 0.5`). A port of XLA's polynomial approach
# (`xla/codegen/intrinsic/cpp/expm1.cc`) is within 0.8 ulp everywhere, but 7x slower
function handle(
    ::Val{Symbol("stablehlo.exponential_minus_one")}, graph::MPSGraph, (x,), _, (result,)
)
    T = result.eltype
    T <: AbstractFloat || throw(
        unimplemented(
            "`stablehlo.exponential_minus_one` of `$T` is not supported by `MetalClient`",
        ),
    )

    one_ = MPSGraphs.constantWithScalar(graph, 1, T)
    minus_one = MPSGraphs.constantWithScalar(graph, -1, T)
    inf = MPSGraphs.constantWithScalar(graph, Inf, T)
    u = handle(Val(Symbol("stablehlo.exponential")), graph, (x,), (;), ())
    um1 = handle(Val(Symbol("stablehlo.subtract")), graph, (u, one_), (;), ())
    logu = handle(Val(Symbol("stablehlo.log")), graph, (u,), (;), ())
    ratio = handle(Val(Symbol("stablehlo.divide")), graph, (x, logu), (;), ())
    y = handle(Val(Symbol("stablehlo.multiply")), graph, (um1, ratio), (;), ())

    # the formula fails for `u == 1` (0 / 0), `u - 1 == -1` (`log(u)` may be -Inf) and `u == Inf` (Inf / Inf)
    for (value, input, output) in ((u, one_, x), (um1, minus_one, minus_one), (u, inf, inf))
        cond = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(value::id{MPSGraphTensor})
            secondaryTensor:(input::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        y = handle(Val(Symbol("stablehlo.select")), graph, (cond, output, y), (;), ())
    end
    return y
end

# Goldberg's `log(u) * x / (u - 1)` with `u = 1 + x`. The plain `log(1 + x)` loses `x` near 0, since `u` is rounded to
# the spacing of the floats around 1 (2^-23 for `Float32`): its relative error is about 6e-8 / |x| (e.g. 2e-1 for
# `x = 1e-7`), and it returns 0 for |x| below 6e-8. This formula cancels the rounding error of `u`, for 3% (`Float32`)
# to 13% (`Float16`) more runtime than `log(1 + x)` alone.
# Tested within 3.4 ulp for `Float32` and 0.5 ulp for `Float16`. Most of the error comes from `log` of MPSGraph (1.8 ulp
# for `Float32`). A port of XLA's rational approximation (`xla/codegen/intrinsic/log1p.cc`) is within 1.2 ulp for
# `|x| < sqrt(2) - 1`, but 4x slower
function handle(
    ::Val{Symbol("stablehlo.log_plus_one")}, graph::MPSGraph, (x,), _, (result,)
)
    T = result.eltype
    T <: AbstractFloat || throw(
        unimplemented("`stablehlo.log_plus_one` of `$T` is not supported by `MetalClient`"),
    )

    one_ = MPSGraphs.constantWithScalar(graph, 1, T)
    inf = MPSGraphs.constantWithScalar(graph, Inf, T)
    u = handle(Val(Symbol("stablehlo.add")), graph, (one_, x), (;), ())
    um1 = handle(Val(Symbol("stablehlo.subtract")), graph, (u, one_), (;), ())
    ratio = handle(Val(Symbol("stablehlo.divide")), graph, (x, um1), (;), ())
    logu = handle(Val(Symbol("stablehlo.log")), graph, (u,), (;), ())
    y = handle(Val(Symbol("stablehlo.multiply")), graph, (logu, ratio), (;), ())

    # the formula fails for `u == 1` (0 / 0) and `u == Inf` (Inf / Inf)
    for (input, output) in ((one_, x), (inf, inf))
        cond = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(u::id{MPSGraphTensor})
            secondaryTensor:(input::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        y = handle(Val(Symbol("stablehlo.select")), graph, (cond, output, y), (;), ())
    end
    return y
end

# `modulo` is floored like `mod` (the result has the sign of `b`), but StableHLO truncates like `rem` (the sign of
# `a`), so the results with a different sign than `a` are shifted by `b`. Zero results take the sign of `a` instead
function handle(::Val{Symbol("stablehlo.remainder")}, graph::MPSGraph, (a, b), _, (result,))
    T = result.eltype
    T <: Complex && throw(
        unimplemented("`stablehlo.remainder` of `$T` is not supported by `MetalClient`")
    )

    zero_ = MPSGraphs.constantWithScalar(graph, 0, T)
    r = @objc [
        graph::id{MPSGraph} moduloWithPrimaryTensor:(a::id{MPSGraphTensor})
        secondaryTensor:(b::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    rneg = @objc [
        graph::id{MPSGraph} lessThanWithPrimaryTensor:(r::id{MPSGraphTensor})
        secondaryTensor:(zero_::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    aneg = @objc [
        graph::id{MPSGraph} lessThanWithPrimaryTensor:(a::id{MPSGraphTensor})
        secondaryTensor:(zero_::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    shift = @objc [
        graph::id{MPSGraph} logicalXORWithPrimaryTensor:(rneg::id{MPSGraphTensor})
        secondaryTensor:(aneg::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    shifted = handle(Val(Symbol("stablehlo.subtract")), graph, (r, b), (;), ())
    y = handle(Val(Symbol("stablehlo.select")), graph, (shift, shifted, r), (;), ())

    # `a * 0` is a zero with the sign of `a`. This also undoes the shift of zeros when `a` is negative
    rzero = @objc [
        graph::id{MPSGraph} equalWithPrimaryTensor:(r::id{MPSGraphTensor})
        secondaryTensor:(zero_::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    signed_zero = handle(Val(Symbol("stablehlo.multiply")), graph, (a, zero_), (;), ())
    return handle(Val(Symbol("stablehlo.select")), graph, (rzero, signed_zero, y), (;), ())
end

# `round` of MPSGraph rounds halfway cases away from zero, but returns +0 for the negative inputs that round to 0, which
# get the sign of `x` back from `x * 0`
function handle(
    ::Val{Symbol("stablehlo.round_nearest_afz")}, graph::MPSGraph, (x,), _, (result,)
)
    y = @objc [
        graph::id{MPSGraph} roundWithTensor:(x::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor
    zero_ = MPSGraphs.constantWithScalar(graph, 0, result.eltype)
    iszero_ = @objc [
        graph::id{MPSGraph} equalWithPrimaryTensor:(y::id{MPSGraphTensor})
        secondaryTensor:(zero_::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    signed_zero = handle(Val(Symbol("stablehlo.multiply")), graph, (x, zero_), (;), ())
    return handle(
        Val(Symbol("stablehlo.select")), graph, (iszero_, signed_zero, y), (;), ()
    )
end

# like in StableHLO, ±0 and NaN are their own sign (kept by `x * 0`), while `sign` of MPSGraph returns +0 for them.
# Metal flushes subnormals to zero, so their sign is 0 too
function handle(::Val{Symbol("stablehlo.sign")}, graph::MPSGraph, (x,), _, (result,))
    T = result.eltype
    T <: Complex &&
        throw(unimplemented("`stablehlo.sign` of `$T` is not supported by `MetalClient`"))
    s = @objc [
        graph::id{MPSGraph} signWithTensor:(x::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor
    if T <: AbstractFloat
        zero_ = MPSGraphs.constantWithScalar(graph, 0, T)
        iszero_ = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(x::id{MPSGraphTensor})
            secondaryTensor:(zero_::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        isnan = @objc [
            graph::id{MPSGraph} isNaNWithTensor:(x::id{MPSGraphTensor}) name:(nil::id{NSString})
        ]::MPSGraphTensor
        keep = handle(
            Val(Symbol("stablehlo.or")), graph, (iszero_, isnan), (;), ((; eltype=Bool),)
        )
        kept = handle(Val(Symbol("stablehlo.multiply")), graph, (x, zero_), (;), ())
        return handle(Val(Symbol("stablehlo.select")), graph, (keep, kept, s), (;), ())
    end
    return s
end

# like `minimum(maximum(x, lo), hi)`, which propagate NaN unlike `clamp` of MPSGraph. Scalar bounds are broadcast
function handle(::Val{Symbol("stablehlo.clamp")}, graph::MPSGraph, (lo, x, hi), _, results)
    x = handle(Val(Symbol("stablehlo.maximum")), graph, (x, lo), (;), results)
    return handle(Val(Symbol("stablehlo.minimum")), graph, (x, hi), (;), results)
end

# all directions are built from `lessThan` and `equal`, which are false for NaN like in StableHLO.
# `compare_type` TOTALORDER orders floats as -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN, which is the order of
# their bits as signed integers once the bits of negative floats other than the sign are flipped
function handle(::Val{Symbol("stablehlo.compare")}, graph::MPSGraph, (a, b), attributes, _)
    T = MPS.jl_mps_to_typ[a.dataType]
    dir = attributes.comparison_direction
    if get(attributes, :compare_type, nothing) == :TOTALORDER && T <: AbstractFloat
        S = sizeof(T) == 2 ? Int16 : Int32
        zero_ = MPSGraphs.constantWithScalar(graph, 0, S)
        flip = MPSGraphs.constantWithScalar(graph, typemax(S), S)
        a, b = map((a, b)) do x
            bits = @objc [
                graph::id{MPSGraph} reinterpretCastTensor:(x::id{MPSGraphTensor})
                toType:(S::MPSDataType)
                name:(nil::id{NSString})
            ]::MPSGraphTensor
            neg = @objc [
                graph::id{MPSGraph} lessThanWithPrimaryTensor:(bits::id{MPSGraphTensor})
                secondaryTensor:(zero_::id{MPSGraphTensor})
                name:(nil::id{NSString})
            ]::MPSGraphTensor
            flipped = handle(
                Val(Symbol("stablehlo.xor")), graph, (bits, flip), (;), ((; eltype=S),)
            )
            handle(Val(Symbol("stablehlo.select")), graph, (neg, flipped, bits), (;), ())
        end
    end

    if dir in (:EQ, :NE)
        eq = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        dir == :EQ && return eq
        return handle(Val(Symbol("stablehlo.not")), graph, (eq,), (;), ((; eltype=Bool),))
    end
    a, b = dir in (:GT, :GE) ? (b, a) : (a, b)
    lt = @objc [
        graph::id{MPSGraph} lessThanWithPrimaryTensor:(a::id{MPSGraphTensor})
        secondaryTensor:(b::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    dir in (:LT, :GT) && return lt
    eq = @objc [
        graph::id{MPSGraph} equalWithPrimaryTensor:(a::id{MPSGraphTensor})
        secondaryTensor:(b::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return handle(Val(Symbol("stablehlo.or")), graph, (lt, eq), (;), ((; eltype=Bool),))
end

# MPSGraph has no fused multiply-add, so `a * b + c` is rounded twice. 16-bit floats are computed in `Float32`, where
# their product is exact, so only the final conversion rounds (twice, if the sum is rounded too)
function handle(::Val{Symbol("stablehlo.fma")}, graph::MPSGraph, (a, b, c), _, (result,))
    T = result.eltype
    widen = T <: AbstractFloat && sizeof(T) == 2
    if widen
        a, b, c = map(x -> MPSGraphs.castTensor(graph, x, Float32), (a, b, c))
    end
    ab = handle(Val(Symbol("stablehlo.multiply")), graph, (a, b), (;), ())
    y = handle(Val(Symbol("stablehlo.add")), graph, (ab, c), (;), ())
    return widen ? MPSGraphs.castTensor(graph, y, T) : y
end

# `power` of MPSGraph is exact for integers (wrapping on overflow) and follows StableHLO for negative exponents, except
# that it raises 0 to them to the largest integer instead of 0
function handle(::Val{Symbol("stablehlo.power")}, graph::MPSGraph, (x, y), _, (result,))
    z = @objc [
        graph::id{MPSGraph} powerWithPrimaryTensor:(x::id{MPSGraphTensor})
        secondaryTensor:(y::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    T = result.eltype
    if T <: Signed
        zero_ = MPSGraphs.constantWithScalar(graph, 0, T)
        xzero = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(x::id{MPSGraphTensor})
            secondaryTensor:(zero_::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        yneg = @objc [
            graph::id{MPSGraph} lessThanWithPrimaryTensor:(y::id{MPSGraphTensor})
            secondaryTensor:(zero_::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        vanish = handle(
            Val(Symbol("stablehlo.and")), graph, (xzero, yneg), (;), ((; eltype=Bool),)
        )
        return handle(Val(Symbol("stablehlo.select")), graph, (vanish, zero_, z), (;), ())
    end
    return z
end

# StableHLO shifts every bit out when the amount, read as unsigned, is at least the bit width, while the shifts of
# MPSGraph are undefined there. Logical shifts of signed integers and arithmetic shifts of unsigned ones are done on the
# bits reinterpreted with the other signedness
for (op, selector) in (
    ("shift_left", :bitwiseLeftShiftWithPrimaryTensor),
    ("shift_right_logical", :bitwiseRightShiftWithPrimaryTensor),
    ("shift_right_arithmetic", :bitwiseRightShiftWithPrimaryTensor),
)
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a, b), _, (result,)
    )
        T = result.eltype
        nbits = 8 * sizeof(T)
        U = unsigned(T)
        bu = if T <: Signed
            (@objc [
                graph::id{MPSGraph} reinterpretCastTensor:(b::id{MPSGraphTensor})
                toType:(U::MPSDataType)
                name:(nil::id{NSString})
            ]::MPSGraphTensor)
        else
            b
        end
        width = MPSGraphs.constantWithScalar(graph, nbits, U)
        out = @objc [
            graph::id{MPSGraph} greaterThanOrEqualToWithPrimaryTensor:(bu::id{MPSGraphTensor})
            secondaryTensor:(width::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor

        # the arithmetic shift by `nbits - 1` fills every bit with the sign
        if $op == "shift_right_arithmetic"
            b = handle(
                Val(Symbol("stablehlo.select")),
                graph,
                (out, MPSGraphs.constantWithScalar(graph, nbits - 1, T), b),
                (;),
                (),
            )
        end
        S = if $op == "shift_right_logical"
            U
        elseif $op == "shift_right_arithmetic"
            signed(T)
        else
            T
        end
        if S != T
            a, b = map((a, b)) do x
                @objc [
                    graph::id{MPSGraph} reinterpretCastTensor:(x::id{MPSGraphTensor})
                    toType:(S::MPSDataType)
                    name:(nil::id{NSString})
                ]::MPSGraphTensor
            end
        end
        y = @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        if S != T
            y = @objc [
                graph::id{MPSGraph} reinterpretCastTensor:(y::id{MPSGraphTensor})
                toType:(T::MPSDataType)
                name:(nil::id{NSString})
            ]::MPSGraphTensor
        end
        $op == "shift_right_arithmetic" && return y
        zero_ = MPSGraphs.constantWithScalar(graph, 0, T)
        return handle(Val(Symbol("stablehlo.select")), graph, (out, zero_, y), (;), ())
    end
end

# MPSGraph scales the last dimension by the ratio of the element sizes, while StableHLO adds or removes a dimension of
# that size, so the result is reshaped
function handle(
    ::Val{Symbol("stablehlo.bitcast_convert")}, graph::MPSGraph, (a,), _, (result,)
)
    y = @objc [
        graph::id{MPSGraph} reinterpretCastTensor:(a::id{MPSGraphTensor})
        toType:(result.eltype::MPSDataType)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return MPSGraphs.reshapeTensor(graph, y, convert(MPSShape, result.size))
end

# the barrier only constrains the scheduling of XLA, so the operands are passed through
function handle(
    ::Val{Symbol("stablehlo.optimization_barrier")}, graph::MPSGraph, operands, _, _
)
    return length(operands) == 1 ? only(operands) : operands
end

# MPSGraph builds the loop by calling 2 blocks once: `before` computes the condition from the values of an iteration
# and passes them on, to `after` if it holds and as the results otherwise, and `after` computes the next values.
# A Julia exception can't unwind through MPSGraph, so an error in a region is rethrown once the loop is built, and
# the blocks pass their arguments through meanwhile
function handle(
    ::Val{Symbol("stablehlo.while")},
    graph::MPSGraph,
    operands,
    _,
    results,
    (cond, body),
    vars,
)
    err = Ref{Any}(nothing)
    # the second argument is an `NSMutableArray`, which has no wrapper
    before = @objcblock(id{MPSGraphTensor}, (id{NSArray}, id{NSArray})) do inputs, outputs
        @objc [outputs::id{NSArray} addObjectsFromArray:(inputs::id{NSArray})]::Nothing
        try
            for (i, t) in enumerate(convert(Vector{MPSGraphTensor}, NSArray(inputs)))
                vars[IR.argument(cond, i).ref.ptr] = t
            end
            return only(codegen!(graph, cond, vars)).ptr
        catch e
            isnothing(err[]) && (err[] = e)
            return MPSGraphs.constantWithScalar(graph, 0, Bool).ptr
        end
    end
    after = @objcblock(id{NSArray}, (id{NSArray},)) do inputs
        try
            for (i, t) in enumerate(convert(Vector{MPSGraphTensor}, NSArray(inputs)))
                vars[IR.argument(body, i).ref.ptr] = t
            end
            return NSArray(codegen!(graph, body, vars)).ptr
        catch e
            isnothing(err[]) && (err[] = e)
            return inputs
        end
    end

    outputs = @objc [
        graph::id{MPSGraph} whileWithInitialInputs:(NSArray(collect(operands))::id{NSArray})
        before:(before::id{NSBlock})
        after:(after::id{NSBlock})
        name:(nil::id{NSString})
    ]::id{NSArray}
    # the blocks keep their closures alive until they are released
    release(before)
    release(after)
    isnothing(err[]) || throw(err[])

    outputs = convert(Vector{MPSGraphTensor}, NSArray(outputs))
    return length(results) == 1 ? only(outputs) : Tuple(outputs)
end

# MPSGraph builds the 2 branches of an `if` by calling a block for each, so `case` is a chain of them: branch 0 if
# `index == 0`, and otherwise the `case` of the other branches with `index - 1`. Indices out of range end up in the
# last branch, like in StableHLO. As in `stablehlo.while`, an error in a branch is rethrown once the `if` is built,
# and the branch returns zeros meanwhile
function handle(
    ::Val{Symbol("stablehlo.case")}, graph::MPSGraph, (index,), _, results, branches, vars
)
    if length(branches) == 1
        outputs = codegen!(graph, only(branches), vars)
        return length(results) == 1 ? only(outputs) : Tuple(outputs)
    end

    err = Ref{Any}(nothing)
    function zeros_()
        return NSArray(
            MPSGraphTensor[
                MPSGraphs.broadcastTensor(
                    graph,
                    MPSGraphs.constantWithScalar(graph, 0, result.eltype),
                    convert(MPSShape, result.size),
                ) for result in results
            ],
        )
    end
    first_ = @objcblock(id{NSArray}, ()) do
        try
            return NSArray(codegen!(graph, first(branches), vars)).ptr
        catch e
            isnothing(err[]) && (err[] = e)
            return zeros_().ptr
        end
    end
    rest = @objcblock(id{NSArray}, ()) do
        try
            one_ = MPSGraphs.constantWithScalar(graph, 1, Int32)
            next = handle(Val(Symbol("stablehlo.subtract")), graph, (index, one_), (;), ())
            outputs = handle(
                Val(Symbol("stablehlo.case")),
                graph,
                (next,),
                (;),
                results,
                branches[2:end],
                vars,
            )
            return NSArray(outputs isa Tuple ? MPSGraphTensor[outputs...] : [outputs]).ptr
        catch e
            isnothing(err[]) && (err[] = e)
            return zeros_().ptr
        end
    end

    # StableHLO indices are `si32`
    zero_ = MPSGraphs.constantWithScalar(graph, 0, Int32)
    pred = @objc [
        graph::id{MPSGraph} equalWithPrimaryTensor:(index::id{MPSGraphTensor})
        secondaryTensor:(zero_::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    outputs = @objc [
        graph::id{MPSGraph} ifWithPredicateTensor:(pred::id{MPSGraphTensor})
        thenBlock:(first_::id{NSBlock})
        elseBlock:(rest::id{NSBlock})
        name:(nil::id{NSString})
    ]::id{NSArray}
    # the blocks keep their closures alive until they are released
    release(first_)
    release(rest)
    isnothing(err[]) || throw(err[])

    outputs = convert(Vector{MPSGraphTensor}, NSArray(outputs))
    return length(results) == 1 ? only(outputs) : Tuple(outputs)
end

# `if` is a `case` with index 0 if the predicate holds and 1 otherwise
function handle(
    ::Val{Symbol("stablehlo.if")}, graph::MPSGraph, (pred,), _, results, branches, vars
)
    index = @objc [
        graph::id{MPSGraph} notWithTensor:(pred::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor
    index = MPSGraphs.castTensor(graph, index, Int32)
    return handle(
        Val(Symbol("stablehlo.case")), graph, (index,), (;), results, branches, vars
    )
end

# `func` is the body of the function named by the `callee` (`func.call`) or `decomposition` (`stablehlo.composite`)
# attribute, which is inlined: its arguments are bound to the operands and its returned values are the results. The
# function is isolated from above, so it only uses its arguments, and another op calling it rebinds them
for op in ("func.call", "stablehlo.composite")
    @eval function handle(
        ::Val{Symbol($op)}, graph::MPSGraph, operands, _, results, (func,), vars
    )
        for (i, operand) in enumerate(operands)
            vars[IR.argument(func, i).ref.ptr] = operand
        end
        outputs = codegen!(graph, func, vars)
        return length(results) == 1 ? only(outputs) : Tuple(outputs)
    end
end

# executables run on a single device, as replica 0 of partition 0
for op in ("partition_id", "replica_id")
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, _, _, (result,)
    )
        zero_ = MPSGraphs.constantWithScalar(graph, 0, result.eltype)
        return MPSGraphs.reshapeTensor(graph, zero_, convert(MPSShape, result.size))
    end
end

# shapes are static (see `tensor_type`)
function handle(
    ::Val{Symbol("stablehlo.get_dimension_size")},
    graph::MPSGraph,
    (a,),
    attributes,
    (result,),
)
    n = MPSGraphs.constantWithScalar(
        graph, size(a)[attributes.dimension + 1], result.eltype
    )
    return MPSGraphs.reshapeTensor(graph, n, convert(MPSShape, result.size))
end

# like the reference interpreter of StableHLO: the mantissa is rounded to nearest even by adding a bias to the bits, and
# exponents out of the reduced range become Inf or 0. The masks and biases are constants of the unsigned integer with
# the bits of the float
function handle(
    ::Val{Symbol("stablehlo.reduce_precision")},
    graph::MPSGraph,
    (x,),
    attributes,
    (result,),
)
    T = result.eltype
    nbits = 8 * sizeof(T)
    src_mbits = if T == Float16
        10
    elseif T == Float32
        23
    else
        7
    end
    src_ebits = nbits - src_mbits - 1
    ebits, mbits = attributes.exponent_bits, attributes.mantissa_bits
    U = unsigned(sizeof(T) == 2 ? Int16 : Int32)
    c(v) = MPSGraphs.constantWithScalar(graph, v, U)
    bits = @objc [
        graph::id{MPSGraph} reinterpretCastTensor:(x::id{MPSGraphTensor})
        toType:(U::MPSDataType)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    and(a, b) = handle(Val(Symbol("stablehlo.and")), graph, (a, b), (;), ((; eltype=U),))
    select(p, a, b) = handle(Val(Symbol("stablehlo.select")), graph, (p, a, b), (;), ())

    if mbits < src_mbits
        last_bit_mask = 1 << (src_mbits - mbits)
        last_bit = and(bits, c(last_bit_mask))
        last_bit = @objc [
            graph::id{MPSGraph} bitwiseRightShiftWithPrimaryTensor:(last_bit::id{MPSGraphTensor})
            secondaryTensor:(c(src_mbits - mbits)::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        bias = handle(
            Val(Symbol("stablehlo.add")),
            graph,
            (last_bit, c((last_bit_mask >> 1) - 1)),
            (;),
            (),
        )
        bits = handle(Val(Symbol("stablehlo.add")), graph, (bits, bias), (;), ())
        bits = and(bits, c(~(last_bit_mask - 1) & (2^nbits - 1)))
    end
    if ebits < src_ebits
        exp_mask = ((1 << src_ebits) - 1) << src_mbits
        exp_bias = (1 << (src_ebits - 1)) - 1
        reduced_bias = (1 << (ebits - 1)) - 1
        exponent = and(bits, c(exp_mask))
        overflows = @objc [
            graph::id{MPSGraph} greaterThanWithPrimaryTensor:(exponent::id{MPSGraphTensor})
            secondaryTensor:(c((exp_bias + reduced_bias) << src_mbits)::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        underflows = @objc [
            graph::id{MPSGraph} lessThanOrEqualToWithPrimaryTensor:(exponent::id{MPSGraphTensor})
            secondaryTensor:(c((exp_bias - reduced_bias) << src_mbits)::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        signed_zero = and(bits, c(1 << (nbits - 1)))
        signed_inf = handle(
            Val(Symbol("stablehlo.or")),
            graph,
            (signed_zero, c(exp_mask)),
            (;),
            ((; eltype=U),),
        )
        bits = select(overflows, signed_inf, bits)
        bits = select(underflows, signed_zero, bits)
    end
    y = @objc [
        graph::id{MPSGraph} reinterpretCastTensor:(bits::id{MPSGraphTensor})
        toType:(T::MPSDataType)
        name:(nil::id{NSString})
    ]::MPSGraphTensor

    # NaN is kept if any mantissa bit is left, and becomes Inf otherwise
    isnan = @objc [
        graph::id{MPSGraph} isNaNWithTensor:(x::id{MPSGraphTensor}) name:(nil::id{NSString})
    ]::MPSGraphTensor
    nan = mbits > 0 ? x : MPSGraphs.constantWithScalar(graph, Inf, T)
    return select(isnan, nan, y)
end

# vector of `Int32` with the scalars `indices` clamped to `0:limits[i]`, like StableHLO does for the start of the slices
# of `dynamic_slice` and `dynamic_update_slice`, so that they are always in bounds
function clamped_starts(graph, indices, limits)
    starts = map(enumerate(indices)) do (i, index)
        T = MPS.jl_mps_to_typ[index.dataType]
        lo = MPSGraphs.constantWithScalar(graph, 0, T)
        hi = MPSGraphs.constantWithScalar(graph, limits[i], T)
        index = handle(
            Val(Symbol("stablehlo.clamp")), graph, (lo, index, hi), (;), ((; eltype=T),)
        )
        index = MPSGraphs.castTensor(graph, index, Int32)
        MPSGraphs.reshapeTensor(graph, index, convert(MPSShape, [1]))
    end
    return @objc [
        graph::id{MPSGraph} concatTensors:(NSArray(starts)::id{NSArray})
        dimension:(0::NSInteger)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# the size of the result of `sliceTensor` with a start tensor is only known at run time, so it is reshaped to the static
# one
function handle(
    ::Val{Symbol("stablehlo.dynamic_slice")},
    graph::MPSGraph,
    operands,
    attributes,
    (result,),
)
    a, indices = operands[1], operands[2:end]
    sizes = attributes.slice_sizes
    starts = clamped_starts(graph, indices, collect(size(a)) .- sizes)
    sizes = handle(
        Val(Symbol("stablehlo.constant")),
        graph,
        (),
        (; value=Int32.(sizes)),
        ((; eltype=Int32, size=[length(sizes)]),),
    )
    y = @objc [
        graph::id{MPSGraph} sliceTensor:(a::id{MPSGraphTensor})
        startTensor:(starts::id{MPSGraphTensor})
        sizeTensor:(sizes::id{MPSGraphTensor})
        squeezeMask:(0::UInt32)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return MPSGraphs.reshapeTensor(graph, y, convert(MPSShape, result.size))
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_update_slice")}, graph::MPSGraph, operands, _, _
)
    a, update, indices = operands[1], operands[2], operands[3:end]
    usize = collect(size(update))
    starts = clamped_starts(graph, indices, collect(size(a)) .- usize)
    function vector(v)
        return handle(
            Val(Symbol("stablehlo.constant")),
            graph,
            (),
            (; value=Int32.(v)),
            ((; eltype=Int32, size=[length(v)]),),
        )
    end
    ends = handle(Val(Symbol("stablehlo.add")), graph, (starts, vector(usize)), (;), ())
    strides = vector(ones(Int32, length(usize)))
    return @objc [
        graph::id{MPSGraph} sliceUpdateDataTensor:(a::id{MPSGraphTensor})
        updateTensor:(update::id{MPSGraphTensor})
        startsTensor:(starts::id{MPSGraphTensor})
        endsTensor:(ends::id{MPSGraphTensor})
        stridesTensor:(strides::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# `padTensor` of MPSGraph only takes a constant padding value, so the operand is padded with zeros along with a mask of
# its elements, and the padding value is selected where the mask is 0. Interior padding pads each element along a new
# dimension, which is then merged, and negative padding removes elements by slicing
function handle(::Val{Symbol("stablehlo.pad")}, graph::MPSGraph, (a, value), attributes, _)
    low, high = attributes.edge_padding_low, attributes.edge_padding_high
    interior = attributes.interior_padding
    n = length(low)
    function pad(x, left, right)
        @objc [
            graph::id{MPSGraph} padTensor:(x::id{MPSGraphTensor})
            withPaddingMode:(MPSGraphs.MPSGraphPaddingModeConstant::MPSGraphs.MPSGraphPaddingMode)
            leftPadding:(NSArray(NSNumber.(left))::id{NSArray})
            rightPadding:(NSArray(NSNumber.(right))::id{NSArray})
            constantValue:(0.0::Float64)
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
    function slice(x, starts, limits)
        return handle(
            Val(Symbol("stablehlo.slice")),
            graph,
            (x,),
            (; start_indices=starts, limit_indices=limits, strides=ones(Int64, n)),
            (),
        )
    end

    one_ = MPSGraphs.constantWithScalar(graph, 1, Int8)
    mask = MPSGraphs.broadcastTensor(graph, one_, convert(MPSShape, collect(size(a))))
    tensors = (a, mask)
    for d in 1:n
        interior[d] == 0 && continue
        tensors = map(tensors) do x
            shape = collect(size(x))
            x = MPSGraphs.reshapeTensor(
                graph, x, convert(MPSShape, [shape[1:d]; 1; shape[(d + 1):end]])
            )
            right = zeros(Int64, n + 1)
            right[d + 1] = interior[d]
            x = pad(x, zeros(Int64, n + 1), right)
            shape[d] *= 1 + interior[d]
            x = MPSGraphs.reshapeTensor(graph, x, convert(MPSShape, shape))
            # the padding after the last element is dropped
            shape[d] -= interior[d]
            slice(x, zeros(Int64, n), shape)
        end
    end
    tensors = map(x -> pad(x, max.(low, 0), max.(high, 0)), tensors)
    if any(<(0), low) || any(<(0), high)
        tensors = map(tensors) do x
            slice(x, max.(.-low, 0), collect(size(x)) .- max.(.-high, 0))
        end
    end

    x, mask = tensors
    mask = MPSGraphs.castTensor(graph, mask, Bool)
    return handle(Val(Symbol("stablehlo.select")), graph, (mask, x, value), (;), ())
end

# like `scatter`: `gatherND` takes the indexed dimensions first, and returns the dimensions of the indices (the batch
# dimensions) followed by the other dimensions of the operand. The windows can't span the indexed dimensions, which may
# be collapsed or kept with size 1, and the windows start at 0 in the other dimensions. Like in StableHLO, the indices
# are clamped into the operand
function handle(
    ::Val{Symbol("stablehlo.gather")}, graph::MPSGraph, (a, indices), attributes, (result,)
)
    dn = attributes.dimension_numbers
    sizes = attributes.slice_sizes
    indexed = dn.start_index_map
    if any(!=(1), sizes[indexed .+ 1]) ||
        !isempty(dn.operand_batching_dims) ||
        !isempty(dn.start_indices_batching_dims)
        throw(
            unimplemented(
                "`stablehlo.gather` with these dimension numbers is not supported by `MetalClient`",
            ),
        )
    end
    asize = collect(size(a))
    r = length(asize)
    limits = copy(sizes)
    limits[indexed .+ 1] .= asize[indexed .+ 1]
    if limits != asize
        a = handle(
            Val(Symbol("stablehlo.slice")),
            graph,
            (a,),
            (; start_indices=zeros(Int64, r), limit_indices=limits, strides=ones(Int64, r)),
            (),
        )
    end
    a = permute(graph, a, [indexed; setdiff(0:(r - 1), indexed)])

    # the index vector becomes the last dimension
    isize = collect(size(indices))
    vector_dim = dn.index_vector_dim
    if vector_dim == length(isize)
        indices = MPSGraphs.reshapeTensor(graph, indices, convert(MPSShape, [isize; 1]))
    else
        indices = permute(
            graph, indices, [setdiff(0:(length(isize) - 1), vector_dim); vector_dim]
        )
    end
    T = MPS.jl_mps_to_typ[indices.dataType]
    m = length(indexed)
    hi = handle(
        Val(Symbol("stablehlo.constant")),
        graph,
        (),
        (; value=T.(asize[indexed .+ 1] .- 1)),
        ((; eltype=T, size=[m]),),
    )
    lo = MPSGraphs.constantWithScalar(graph, 0, T)
    indices = handle(
        Val(Symbol("stablehlo.clamp")), graph, (lo, indices, hi), (;), ((; eltype=T),)
    )
    indices = MPSGraphs.castTensor(graph, indices, Int32)
    batch_size = collect(size(indices))[1:(end - 1)]

    g = @objc [
        graph::id{MPSGraph} gatherNDWithUpdatesTensor:(a::id{MPSGraphTensor})
        indicesTensor:(indices::id{MPSGraphTensor})
        batchDimensions:(0::NSUInteger)
        name:(nil::id{NSString})
    ]::MPSGraphTensor

    # the indexed dimensions that aren't collapsed are kept with size 1, and the offset dimensions are moved to their
    # place among the batch ones
    offset = setdiff(0:(r - 1), dn.collapsed_slice_dims)
    g = MPSGraphs.reshapeTensor(
        graph, g, convert(MPSShape, [batch_size; sizes[offset .+ 1]])
    )
    nbatch = length(batch_size)
    perm = Int64[]
    next_batch = 0
    for p in 0:(length(result.size) - 1)
        j = findfirst(==(p), dn.offset_dims)
        if isnothing(j)
            push!(perm, next_batch)
            next_batch += 1
        else
            push!(perm, nbatch + j - 1)
        end
    end
    return permute(graph, g, perm)
end

# the comparator must be a single `compare` of the 2 elements of one operand, the key: `LT` or `LE` of `(lhs, rhs)`
# sorts it ascending and `GT` or `GE` descending (the other way around with the arguments swapped). The other operands
# follow the order of the key, which are gathered with its `argSort`
function handle(
    ::Val{Symbol("stablehlo.sort")},
    graph::MPSGraph,
    operands,
    attributes,
    _,
    (comparator,),
    _,
)
    ops = collect(comparator)
    args = [IR.argument(comparator, i).ref.ptr for i in 1:IR.nargs(comparator)]
    if length(ops) != 2 ||
        IR.name(ops[1]) != "stablehlo.compare" ||
        IR.operand(ops[2], 1).ref.ptr != IR.result(ops[1], 1).ref.ptr
        throw(
            unimplemented(
                "`stablehlo.sort` with this comparator is not supported by `MetalClient`"
            ),
        )
    end
    i = findfirst(==(IR.operand(ops[1], 1).ref.ptr), args)
    j = findfirst(==(IR.operand(ops[1], 2).ref.ptr), args)
    dir = decode_attribute(IR.getattr(ops[1], "comparison_direction"))
    if isnothing(i) ||
        isnothing(j) ||
        !(isodd(min(i, j)) && max(i, j) == min(i, j) + 1) ||
        !(dir in (:LT, :LE, :GT, :GE))
        throw(
            unimplemented(
                "`stablehlo.sort` with this comparator is not supported by `MetalClient`"
            ),
        )
    end
    descending = (dir in (:GT, :GE)) == (i < j)
    axis = attributes.dimension

    length(operands) == 1 &&
        return MPSGraphs.sortWithTensor(graph, only(operands), axis, descending)
    order = MPSGraphs.argSortWithTensor(graph, operands[cld(i, 2)], axis, descending)
    return map(operands) do x
        @objc [
            graph::id{MPSGraph} gatherAlongAxis:(axis::NSInteger)
            withUpdatesTensor:(x::id{MPSGraphTensor})
            indicesTensor:(order::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

# over the innermost dimensions. Like in StableHLO, the inverse transforms are scaled by 1 / N
function handle(::Val{Symbol("stablehlo.fft")}, graph::MPSGraph, (a,), attributes, _)
    type = attributes.fft_type
    lengths = attributes.fft_length
    r = length(size(a))
    axes = NSArray(NSNumber.(collect((r - length(lengths)):(r - 1))))
    inverse = type in (:IFFT, :IRFFT)
    scaling = if inverse
        MPSGraphs.MPSGraphFFTScalingModeSize
    else
        MPSGraphs.MPSGraphFFTScalingModeNone
    end
    desc = MPSGraphs.MPSGraphFFTDescriptor(; inverse, scalingMode=scaling)
    type == :RFFT && return MPSGraphs.realToHermiteanFFTWithTensor(graph, a, axes, desc)
    if type == :IRFFT
        # the length of the last dimension of the result isn't implied by the operand
        desc.roundToOddHermitean = isodd(last(lengths))
        return MPSGraphs.HermiteanToRealFFTWithTensor(graph, a, axes, desc)
    end
    return MPSGraphs.fastFourierTransformWithTensor(graph, a, axes, desc)
end

# the statistics are over every dimension but `feature_index`, along which the parameters are broadcast. The reductions
# keep the reduced dimensions with size 1, so they are broadcast too
function handle(
    ::Val{Symbol("stablehlo.batch_norm_inference")},
    graph::MPSGraph,
    (x, scale, offset, mean, variance),
    attributes,
    _,
)
    shape = ones(Int64, length(size(x)))
    shape[attributes.feature_index + 1] = size(x)[attributes.feature_index + 1]
    scale, offset, mean, variance = map((scale, offset, mean, variance)) do t
        MPSGraphs.reshapeTensor(graph, t, convert(MPSShape, shape))
    end
    T = MPS.jl_mps_to_typ[x.dataType]
    eps = MPSGraphs.constantWithScalar(graph, attributes.epsilon, T)
    rstd = handle(Val(Symbol("stablehlo.add")), graph, (variance, eps), (;), ())
    rstd = handle(Val(Symbol("stablehlo.rsqrt")), graph, (rstd,), (;), ())
    factor = handle(Val(Symbol("stablehlo.multiply")), graph, (rstd, scale), (;), ())
    y = handle(Val(Symbol("stablehlo.subtract")), graph, (x, mean), (;), ())
    y = handle(Val(Symbol("stablehlo.multiply")), graph, (y, factor), (;), ())
    return handle(Val(Symbol("stablehlo.add")), graph, (y, offset), (;), ())
end

# the variance is biased (divided by the number of elements per feature)
function handle(
    ::Val{Symbol("stablehlo.batch_norm_training")},
    graph::MPSGraph,
    (x, scale, offset),
    attributes,
    _,
)
    f = attributes.feature_index
    xsize = collect(size(x))
    shape = ones(Int64, length(xsize))
    shape[f + 1] = xsize[f + 1]
    scale, offset = map(
        t -> MPSGraphs.reshapeTensor(graph, t, convert(MPSShape, shape)), (scale, offset)
    )
    axes = NSArray(NSNumber.(setdiff(0:(length(xsize) - 1), f)))
    T = MPS.jl_mps_to_typ[x.dataType]
    n = MPSGraphs.constantWithScalar(graph, prod(xsize) ÷ xsize[f + 1], T)
    eps = MPSGraphs.constantWithScalar(graph, attributes.epsilon, T)

    mean = MPSGraphs.reductionSumWithTensor(graph, x, axes)
    mean = handle(Val(Symbol("stablehlo.divide")), graph, (mean, n), (;), ())
    centered = handle(Val(Symbol("stablehlo.subtract")), graph, (x, mean), (;), ())
    variance = handle(
        Val(Symbol("stablehlo.multiply")), graph, (centered, centered), (;), ()
    )
    variance = MPSGraphs.reductionSumWithTensor(graph, variance, axes)
    variance = handle(Val(Symbol("stablehlo.divide")), graph, (variance, n), (;), ())

    rstd = handle(Val(Symbol("stablehlo.add")), graph, (variance, eps), (;), ())
    rstd = handle(Val(Symbol("stablehlo.rsqrt")), graph, (rstd,), (;), ())
    factor = handle(Val(Symbol("stablehlo.multiply")), graph, (rstd, scale), (;), ())
    y = handle(Val(Symbol("stablehlo.multiply")), graph, (centered, factor), (;), ())
    y = handle(Val(Symbol("stablehlo.add")), graph, (y, offset), (;), ())
    features = convert(MPSShape, [xsize[f + 1]])
    return (
        y,
        MPSGraphs.reshapeTensor(graph, mean, features),
        MPSGraphs.reshapeTensor(graph, variance, features),
    )
end

# like the expansion of XLA: with `xhat = (x - mean) / sqrt(variance + epsilon)` and `n` elements per feature,
# `grad_offset = sum(dy)`, `grad_scale = sum(dy * xhat)` and
# `grad_operand = scale / sqrt(variance + epsilon) * (dy - (grad_offset + xhat * grad_scale) / n)`
function handle(
    ::Val{Symbol("stablehlo.batch_norm_grad")},
    graph::MPSGraph,
    (x, scale, mean, variance, dy),
    attributes,
    _,
)
    f = attributes.feature_index
    xsize = collect(size(x))
    shape = ones(Int64, length(xsize))
    shape[f + 1] = xsize[f + 1]
    scale, mean, variance = map((scale, mean, variance)) do t
        MPSGraphs.reshapeTensor(graph, t, convert(MPSShape, shape))
    end
    axes = NSArray(NSNumber.(setdiff(0:(length(xsize) - 1), f)))
    T = MPS.jl_mps_to_typ[x.dataType]
    n = MPSGraphs.constantWithScalar(graph, prod(xsize) ÷ xsize[f + 1], T)
    eps = MPSGraphs.constantWithScalar(graph, attributes.epsilon, T)

    rstd = handle(Val(Symbol("stablehlo.add")), graph, (variance, eps), (;), ())
    rstd = handle(Val(Symbol("stablehlo.rsqrt")), graph, (rstd,), (;), ())
    xhat = handle(Val(Symbol("stablehlo.subtract")), graph, (x, mean), (;), ())
    xhat = handle(Val(Symbol("stablehlo.multiply")), graph, (xhat, rstd), (;), ())
    grad_offset = MPSGraphs.reductionSumWithTensor(graph, dy, axes)
    grad_scale = handle(Val(Symbol("stablehlo.multiply")), graph, (dy, xhat), (;), ())
    grad_scale = MPSGraphs.reductionSumWithTensor(graph, grad_scale, axes)

    t = handle(Val(Symbol("stablehlo.multiply")), graph, (xhat, grad_scale), (;), ())
    t = handle(Val(Symbol("stablehlo.add")), graph, (grad_offset, t), (;), ())
    t = handle(Val(Symbol("stablehlo.divide")), graph, (t, n), (;), ())
    t = handle(Val(Symbol("stablehlo.subtract")), graph, (dy, t), (;), ())
    factor = handle(Val(Symbol("stablehlo.multiply")), graph, (scale, rstd), (;), ())
    grad_operand = handle(Val(Symbol("stablehlo.multiply")), graph, (factor, t), (;), ())
    features = convert(MPSShape, [xsize[f + 1]])
    return (
        grad_operand,
        MPSGraphs.reshapeTensor(graph, grad_scale, features),
        MPSGraphs.reshapeTensor(graph, grad_offset, features),
    )
end

# MPSGraph only has a Philox generator, whose state is 7 `Int32`: a word that MPSGraph sets to 1, the 128-bit counter
# and the 64-bit key, from the least significant word. Like in XLA, the 2 or 3 `UInt64` of the StableHLO state are the
# key and the counter (whose high half is 0 for 2 of them), but the bits differ from those of XLA. The output is drawn
# as 16-bit integers of `[0, 2^16)`, since wider ranges aren't uniform bits (`[typemin(Int32), 2^31)` only gives
# `typemin(Int32)`). MPSGraph advances the counter by 256 per integer
function handle(
    ::Val{Symbol("stablehlo.rng_bit_generator")},
    graph::MPSGraph,
    (state,),
    attributes,
    (_, result),
)
    algorithm = attributes.rng_algorithm
    if !(algorithm in (:DEFAULT, :PHILOX))
        throw(
            unimplemented(
                "`stablehlo.rng_bit_generator` with `$algorithm` is not supported by `MetalClient`",
            ),
        )
    end
    n = only(size(state))
    if !(n in (2, 3))
        throw(
            unimplemented(
                "`stablehlo.rng_bit_generator` with a state of size $n is not supported by `MetalClient`",
            ),
        )
    end

    function slice(x, start, stop)
        attributes = (; start_indices=[start], limit_indices=[stop], strides=[1])
        return handle(Val(Symbol("stablehlo.slice")), graph, (x,), attributes, ())
    end
    function concat(xs...)
        return handle(Val(Symbol("stablehlo.concatenate")), graph, xs, (; dimension=0), ())
    end
    function reinterpret_(x, T)
        @objc [
            graph::id{MPSGraph} reinterpretCastTensor:(x::id{MPSGraphTensor})
            toType:(T::MPSDataType)
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end

    # `words` is `[key; counter]` in `Int32`
    words = reinterpret_(state, Int32)
    one_ = MPSGraphs.constantWithScalar(graph, 1, Int32)
    one_ = MPSGraphs.reshapeTensor(graph, one_, convert(MPSShape, [1]))
    high = if n == 2
        zero_ = MPSGraphs.constantWithScalar(graph, 0, Int32)
        (MPSGraphs.broadcastTensor(graph, zero_, convert(MPSShape, [2])),)
    else
        ()
    end
    mps_state = concat(one_, slice(words, 2, 2n), high..., slice(words, 0, 2))

    uniform = MPSGraphs.MPSGraphRandomDistributionUniform
    desc = @objc [
        MPSGraphRandomOpDescriptor descriptorWithDistribution:(uniform::MPSGraphs.MPSGraphRandomDistribution)
        dataType:(Int32::MPSDataType)
    ]::MPSGraphRandomOpDescriptor
    desc.minInteger = 0
    desc.maxInteger = 2^16
    len = prod(result.size)
    nhalves = cld(len * sizeof(result.eltype), 2)
    tensors = @objc [
        graph::id{MPSGraph} randomTensorWithShape:(convert(MPSShape, [nhalves])::id{MPSShape})
        descriptor:(desc::id{MPSGraphRandomOpDescriptor})
        stateTensor:(mps_state::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::id{NSArray}
    halves, mps_state = map((0, 1)) do i
        MPSGraphTensor(
            @objc [tensors::id{NSArray} objectAtIndex:(i::NSUInteger)]::id{MPSGraphTensor}
        )
    end

    bits = reinterpret_(MPSGraphs.castTensor(graph, halves, UInt16), result.eltype)
    # an odd number of bytes leaves one unused
    isodd(len) && sizeof(result.eltype) == 1 && (bits = slice(bits, 0, len))
    output = MPSGraphs.reshapeTensor(graph, bits, convert(MPSShape, result.size))
    new_state = concat(slice(mps_state, 5, 7), slice(mps_state, 1, 2n - 1))
    return (reinterpret_(new_state, UInt64), output)
end

# the memory of `attributes.value` is already in the row-major order of the tensor
function handle(
    ::Val{Symbol("stablehlo.constant")}, graph::MPSGraph, _, attributes, (result,)
)
    value = attributes.value
    data = GC.@preserve value @objc [
        NSData dataWithBytes:(pointer(value)::Ptr{Cvoid}) length:(sizeof(value)::NSUInteger)
    ]::id{NSData}

    # MPSGraph can't create constants without dimensions, so they are created flat
    constant = @objc [
        graph::id{MPSGraph} constantWithData:(data::id{NSData})
        shape:(convert(MPSShape, [length(value)])::id{MPSShape})
        dataType:(result.eltype::MPSDataType)
    ]::MPSGraphTensor
    return MPSGraphs.reshapeTensor(graph, constant, convert(MPSShape, result.size))
end

# the coordinates of MPSGraph are `Int32`
function handle(::Val{Symbol("stablehlo.iota")}, graph::MPSGraph, _, attributes, (result,))
    coords = @objc [
        graph::id{MPSGraph} coordinateAlongAxis:(attributes.iota_dimension::NSInteger)
        withShape:(convert(MPSShape, result.size)::id{MPSShape})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return MPSGraphs.castTensor(graph, coords, result.eltype)
end

# the shape operands of the dynamic ops are ignored, since the shapes of the results are static (see `tensor_type`)
function handle(
    ::Val{Symbol("stablehlo.dynamic_iota")}, graph::MPSGraph, _, attributes, results
)
    return handle(Val(Symbol("stablehlo.iota")), graph, (), attributes, results)
end

function handle(::Val{Symbol("stablehlo.convert")}, graph::MPSGraph, (a,), _, (result,))
    return MPSGraphs.castTensor(graph, a, result.eltype)
end

function handle(::Val{Symbol("stablehlo.reshape")}, graph::MPSGraph, (a,), _, (result,))
    return MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, result.size))
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_reshape")}, graph::MPSGraph, (a, _), attributes, results
)
    return handle(Val(Symbol("stablehlo.reshape")), graph, (a,), attributes, results)
end

function handle(::Val{Symbol("stablehlo.transpose")}, graph::MPSGraph, (a,), attributes, _)
    return permute(graph, a, attributes.permutation)
end

function handle(::Val{Symbol("stablehlo.reverse")}, graph::MPSGraph, (a,), attributes, _)
    return MPSGraphs.reverseTensor(graph, a, attributes.dimensions)
end

function handle(
    ::Val{Symbol("stablehlo.concatenate")}, graph::MPSGraph, operands, attributes, _
)
    return @objc [
        graph::id{MPSGraph} concatTensors:(NSArray(collect(operands))::id{NSArray})
        dimension:(attributes.dimension::NSInteger)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

function handle(::Val{Symbol("stablehlo.slice")}, graph::MPSGraph, (a,), attributes, _)
    return @objc [
        graph::id{MPSGraph} sliceTensor:(a::id{MPSGraphTensor})
        starts:(NSArray(NSNumber.(attributes.start_indices))::id{NSArray})
        ends:(NSArray(NSNumber.(attributes.limit_indices))::id{NSArray})
        strides:(NSArray(NSNumber.(attributes.strides))::id{NSArray})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# dimension `i` of the operand becomes dimension `broadcast_dimensions[i]` of the result
function handle(
    ::Val{Symbol("stablehlo.broadcast_in_dim")},
    graph::MPSGraph,
    (a,),
    attributes,
    (result,),
)
    dims = attributes.broadcast_dimensions

    # the dimensions of the operand are sorted like in the result, and the missing ones are added with size 1
    shape = ones(Int64, length(result.size))
    shape[dims .+ 1] .= size(a)
    a = permute(graph, a, sortperm(dims) .- 1)
    a = MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, shape))
    return MPSGraphs.broadcastTensor(graph, a, convert(MPSShape, result.size))
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_broadcast_in_dim")},
    graph::MPSGraph,
    (a, _),
    attributes,
    results,
)
    return handle(
        Val(Symbol("stablehlo.broadcast_in_dim")), graph, (a,), attributes, results
    )
end

# the contraction is a product of matrices, whose rows and columns are the free and contracted dimensions
function handle(
    ::Val{Symbol("stablehlo.dot_general")}, graph::MPSGraph, (a, b), attributes, (result,)
)
    # MPSGraph aborts on products of matrices of integers
    result.eltype <: Union{AbstractFloat,Complex} || throw(
        unimplemented(
            "`stablehlo.dot_general` of `$(result.eltype)` is not supported by `MetalClient`",
        ),
    )

    dn = attributes.dot_dimension_numbers
    asize, bsize = size(a), size(b)
    abatch, acontract = dn.lhs_batching_dimensions, dn.lhs_contracting_dimensions
    bbatch, bcontract = dn.rhs_batching_dimensions, dn.rhs_contracting_dimensions
    afree = setdiff(0:(length(asize) - 1), abatch, acontract)
    bfree = setdiff(0:(length(bsize) - 1), bbatch, bcontract)

    nbatch = prod(asize[abatch .+ 1]; init=1)
    ncontract = prod(asize[acontract .+ 1]; init=1)
    nafree = prod(asize[afree .+ 1]; init=1)
    nbfree = prod(bsize[bfree .+ 1]; init=1)

    a = permute(graph, a, [abatch; afree; acontract])
    a = MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, [nbatch, nafree, ncontract]))
    b = permute(graph, b, [bbatch; bcontract; bfree])
    b = MPSGraphs.reshapeTensor(graph, b, convert(MPSShape, [nbatch, ncontract, nbfree]))
    c = MPSGraphs.matrixMultiplicationWithPrimaryTensor(graph, a, b)
    return MPSGraphs.reshapeTensor(graph, c, convert(MPSShape, result.size))
end

# for some trivial reduction bodies (i.e. `add`, `multiply`, `maximum` and `minimum`), MPSGraph already provides kernels
# but for more complex reductions, we manually unroll the reduction loop into a tree and parallelize each level of the
# reduction tree with `enzyme-batch`
function handle(
    ::Val{Symbol("stablehlo.reduce")},
    graph::MPSGraph,
    operands,
    attributes,
    results,
    (body,),
    vars,
)
    n = length(results)
    inputs, inits = collect(operands[1:n]), collect(operands[(n + 1):end])
    dims = attributes.dimensions

    # `reducer` fails on bodies with several ops, which are reduced by the tree
    opname = try
        n == 1 ? reducer(body) : nothing
    catch e
        e isa Plugin.Error || rethrow()
        nothing
    end
    reduced = if opname == "stablehlo.add"
        MPSGraphs.reductionSumWithTensor(graph, only(inputs), NSArray(NSNumber.(dims)))
    elseif opname == "stablehlo.multiply"
        MPSGraphs.reductionProductWithTensor(graph, only(inputs), NSArray(NSNumber.(dims)))
    elseif opname == "stablehlo.maximum"
        MPSGraphs.reductionMaximumPropagateNaNWithTensor(
            graph, only(inputs), NSArray(NSNumber.(dims))
        )
    elseif opname == "stablehlo.minimum"
        MPSGraphs.reductionMinimumPropagateNaNWithTensor(
            graph, only(inputs), NSArray(NSNumber.(dims))
        )
    else
        nothing
    end
    if !isnothing(reduced)
        # MPSGraph keeps the reduced dimensions with size 1
        reduced = MPSGraphs.reshapeTensor(
            graph, reduced, convert(MPSShape, only(results).size)
        )
        return handle(Val(Symbol(opname)), graph, (only(inits), reduced), (;), results)
    end

    slice =
        (x, starts, limits) -> handle(
            Val(Symbol("stablehlo.slice")),
            graph,
            (x,),
            (; start_indices=starts, limit_indices=limits, strides=[1, 1]),
            (),
        )

    shape = collect(size(first(inputs)))
    kept = setdiff(0:(length(shape) - 1), dims)
    rows, cols = prod(shape[dims .+ 1]; init=1), prod(shape[kept .+ 1]; init=1)
    # the number of rows at each level
    levels = Int[]
    m = rows
    while m > 1
        push!(levels, m)
        m = m ÷ 2 + isodd(m)
    end

    # `enzyme.batch` calls a function, so the body is copied to `@body` in a module of its own, where `@level_h` applies
    # it to `h` pairs of rows (the last level and the combination with `init` have `h = 1`) through `enzyme.batch`,
    # which `enzyme-batch` turns into ops on whole tensors
    reduce_op = IR.parent_op(body)
    ctx, loc = IR.context(reduce_op), IR.location(reduce_op)
    mod = IR.Module(loc)
    try
        IR.@with_context ctx IR.@with_block IR.body(mod) begin
            scalars = [IR.type(IR.argument(body, i)) for i in 1:n]
            func = Dialects.func.func_(;
                sym_name="body",
                function_type=IR.FunctionType([scalars; scalars], scalars),
                sym_visibility="private",
                body=IR.Region(),
                location=loc,
            )
            # MLIR can only copy whole ops, so the body is taken from a copy of the `stablehlo.reduce`
            clone = copy(reduce_op)
            API.mlirRegionTakeBody(IR.region(func, 1), IR.region(clone, 1))
            IR.dispose(clone)
            block = IR.first_block(IR.region(func, 1))

            # the body can use values defined outside of it, like the constants hoisted by the canonicalizer, which
            # are copied into it
            ops, defined, blocks = IR.Operation[], Set{Ptr{Cvoid}}(), [block]
            while !isempty(blocks)
                b = pop!(blocks)
                foreach(i -> push!(defined, IR.argument(b, i).ref.ptr), 1:IR.nargs(b))
                for op in b
                    push!(ops, op)
                    foreach(
                        i -> push!(defined, IR.result(op, i).ref.ptr), 1:IR.nresults(op)
                    )
                    foreach(region -> append!(blocks, region), op)
                end
            end
            copies = Dict{Ptr{Cvoid},IR.Value}()
            for op in ops, i in 1:IR.noperands(op)
                value = IR.operand(op, i)
                value.ref.ptr in defined && continue
                def = IR.owner(value)
                if !(def isa IR.Operation && IR.name(def) == "stablehlo.constant")
                    msg = "`stablehlo.reduce` whose body uses values defined outside of it, other than constants,"
                    throw(unimplemented("$msg is not supported by `MetalClient`"))
                end
                copied = get!(
                    () -> IR.result(pushfirst!(block, copy(def))), copies, value.ref.ptr
                )
                IR.setoperand!(op, i, copied)
            end

            ret = IR.terminator(block)
            IR.@with_block block Dialects.func.return_(IR.operands(ret); location=loc)
            IR.dispose(ret)

            for h in unique([levels .÷ 2; 1])
                types = [IR.TensorType([h, cols], eltype(t)) for t in scalars]
                entry = IR.Block([types; types], fill(loc, 2n))
                IR.@with_block entry begin
                    batched = Dialects.enzyme.batch(
                        IR.Value[IR.argument(entry, i) for i in 1:(2n)];
                        outputs=types,
                        fn=IR.FlatSymbolRefAttribute("body"),
                        batch_shape=IR.DenseArrayAttribute([h, cols]),
                        location=loc,
                    )
                    Dialects.func.return_(IR.results(batched); location=loc)
                end
                region = IR.Region()
                push!(region, entry)
                Dialects.func.func_(;
                    sym_name="level_$h",
                    function_type=IR.FunctionType([types; types], types),
                    body=region,
                    location=loc,
                )
            end
        end

        pm = IR.PassManager(; context=ctx)
        try
            IR.add_pipeline!(IR.OpPassManager(pm), "enzyme-batch,inline,enzyme-hlo-opt,cse")
            IR.run!(pm, mod)
        catch e
            msg = sprint(showerror, e)
            throw(
                unimplemented("the body of `stablehlo.reduce` could not be batched: $msg")
            )
        finally
            IR.dispose(pm)
        end
        symbols = IR.SymbolTable(IR.Operation(mod))
        level(h) = IR.first_block(IR.region(IR.lookup(symbols, "level_$h"), 1))

        # the inputs are converted to the type of the results, which can be wider
        xs = map(inputs, results) do x, r
            if MPS.jl_mps_to_typ[x.dataType] != r.eltype
                x = MPSGraphs.castTensor(graph, x, r.eltype)
            end
            x = permute(graph, x, [sort(dims); kept])
            return MPSGraphs.reshapeTensor(graph, x, convert(MPSShape, [rows, cols]))
        end
        for m in levels
            half, f = m ÷ 2, level(m ÷ 2)
            for (i, x) in enumerate(xs)
                x = isodd(m) ? slice(x, [0, 0], [2half, cols]) : x
                x = MPSGraphs.reshapeTensor(graph, x, convert(MPSShape, [half, 2cols]))
                vars[IR.argument(f, i).ref.ptr] = slice(x, [0, 0], [half, cols])
                vars[IR.argument(f, n + i).ref.ptr] = slice(x, [0, cols], [half, 2cols])
            end
            ys = codegen!(graph, f, vars)
            if isodd(m)
                ys = map(ys, xs) do y, x
                    last_ = slice(x, [m - 1, 0], [m, cols])
                    attrs = (; dimension=0)
                    return handle(
                        Val(Symbol("stablehlo.concatenate")),
                        graph,
                        (y, last_),
                        attrs,
                        (),
                    )
                end
            end
            xs = ys
        end

        f = level(1)
        for i in 1:n
            vars[IR.argument(f, i).ref.ptr] = MPSGraphs.broadcastTensor(
                graph, inits[i], convert(MPSShape, [1, cols])
            )
            vars[IR.argument(f, n + i).ref.ptr] = xs[i]
        end
        outputs = map(codegen!(graph, f, vars), results) do y, r
            return MPSGraphs.reshapeTensor(graph, y, convert(MPSShape, r.size))
        end
        return n == 1 ? only(outputs) : Tuple(outputs)
    finally
        IR.dispose(mod)
    end
end

# windows are reduced by the pooling of MPSGraph, which works on tensors of 4 dimensions
function handle(
    ::Val{Symbol("stablehlo.reduce_window")},
    graph::MPSGraph,
    operands,
    attributes,
    (result,),
    (body,),
    _,
)
    length(operands) == 2 || throw(
        unimplemented(
            "`stablehlo.reduce_window` of several tensors is not supported by `MetalClient`",
        ),
    )
    a, init = operands
    asize = size(a)
    n = length(asize)

    window = attributes.window_dimensions
    strides = get(attributes, :window_strides, ones(Int64, n))
    dilations = get(attributes, :window_dilations, ones(Int64, n))
    padding = get(attributes, :padding, zeros(Int64, 2, n))
    if any(!=(1), get(attributes, :base_dilations, ones(Int64, n)))
        throw(
            unimplemented(
                "`stablehlo.reduce_window` with base dilations is not supported by `MetalClient`",
            ),
        )
    end

    # the dimensions without a window are merged into the first one
    windowed = filter(1:n) do i
        window[i] != 1 || strides[i] != 1 || dilations[i] != 1 || any(!=(0), padding[:, i])
    end
    nfill = 4 - length(windowed)
    nfill >= 1 || throw(
        unimplemented(
            "`stablehlo.reduce_window` over more than 3 dimensions is not supported by `MetalClient`",
        ),
    )
    perm = [setdiff(1:n, windowed); windowed]
    a = permute(graph, a, perm .- 1)
    shape = [
        prod(asize[setdiff(1:n, windowed)]; init=1)
        ones(Int64, nfill - 1)
        collect(asize[windowed])
    ]
    a = MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, shape))

    kernel = NSArray(NSNumber.([ones(Int64, nfill); window[windowed]]))
    stride = NSArray(NSNumber.([ones(Int64, nfill); strides[windowed]]))
    dilation = NSArray(NSNumber.([ones(Int64, nfill); dilations[windowed]]))
    pad = NSArray(NSNumber.([zeros(Int64, 2 * nfill); vec(padding[:, windowed])]))
    desc = @objc [
        MPSGraphPooling4DOpDescriptor descriptorWithKernelSizes:(kernel::id{NSArray})
        strides:(stride::id{NSArray})
        dilationRates:(dilation::id{NSArray})
        paddingValues:(pad::id{NSArray})
        paddingStyle:(MPSGraphs.MPSGraphPaddingStyleExplicit::MPSGraphs.MPSGraphPaddingStyle)
    ]::MPSGraphPooling4DOpDescriptor

    opname = reducer(body)
    pooled = if opname == "stablehlo.maximum"
        @objc [
            graph::id{MPSGraph} maxPooling4DWithSourceTensor:(a::id{MPSGraphTensor})
            descriptor:(desc::id{MPSGraphPooling4DOpDescriptor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    elseif opname == "stablehlo.add" && result.eltype <: AbstractFloat
        # the sum is the mean (which must count the padding) times the size of the window
        desc.includeZeroPadToAverage = true
        mean = @objc [
            graph::id{MPSGraph} avgPooling4DWithSourceTensor:(a::id{MPSGraphTensor})
            descriptor:(desc::id{MPSGraphPooling4DOpDescriptor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        MPSGraphs.multiplicationWithPrimaryTensor(
            graph,
            mean,
            MPSGraphs.constantWithScalar(graph, prod(window), result.eltype),
        )
    else
        throw(
            unimplemented(
                "`stablehlo.reduce_window` with `$opname` of `$(result.eltype)` is not supported by \
                `MetalClient`",
            ),
        )
    end

    pooled = MPSGraphs.reshapeTensor(graph, pooled, convert(MPSShape, result.size[perm]))
    pooled = permute(graph, pooled, invperm(perm) .- 1)
    return handle(Val(Symbol(opname)), graph, (init, pooled), (;), (result,))
end

# MPSGraph takes the input as (batch, feature, spatial...) and the kernel as (output feature, input feature, spatial...)
function handle(
    ::Val{Symbol("stablehlo.convolution")}, graph::MPSGraph, (a, kernel), attributes, _
)
    dn = attributes.dimension_numbers
    n = length(dn.input_spatial_dimensions)
    strides = get(attributes, :window_strides, ones(Int64, n))
    dilations = get(attributes, :rhs_dilation, ones(Int64, n))
    padding = get(attributes, :padding, zeros(Int64, 2, n))
    groups = attributes.feature_group_count
    if !(n in (2, 3)) ||
        any(!=(1), get(attributes, :lhs_dilation, ones(Int64, n))) ||
        any(get(attributes, :window_reversal, falses(n))) ||
        attributes.batch_group_count != 1 ||
        any(<(0), padding)
        throw(
            unimplemented(
                "`stablehlo.convolution` is only supported by `MetalClient` over 2 or 3 spatial dimensions, and \
                without base dilation, window reversal, batch groups nor negative padding",
            ),
        )
    end

    a = permute(
        graph,
        a,
        [
            dn.input_batch_dimension
            dn.input_feature_dimension
            dn.input_spatial_dimensions
        ],
    )
    kernel = permute(
        graph,
        kernel,
        [
            dn.kernel_output_feature_dimension
            dn.kernel_input_feature_dimension
            dn.kernel_spatial_dimensions
        ],
    )

    # MPSGraph names the spatial dimensions from the last one: X, Y and Z
    conv = if n == 2
        desc = MPSGraphs.MPSGraphConvolution2DOpDescriptor(
            reverse(strides),
            (padding[1, 2], padding[2, 2], padding[1, 1], padding[2, 1]),
            reverse(dilations),
            groups,
        )
        MPSGraphs.convolution2DWithSourceTensor(graph, a, kernel, desc)
    else
        desc = @objc [
            MPSGraphConvolution3DOpDescriptor descriptorWithStrideInX:(strides[3]::NSUInteger)
            strideInY:(strides[2]::NSUInteger)
            strideInZ:(strides[1]::NSUInteger)
            dilationRateInX:(dilations[3]::NSUInteger)
            dilationRateInY:(dilations[2]::NSUInteger)
            dilationRateInZ:(dilations[1]::NSUInteger)
            groups:(groups::NSUInteger)
            paddingLeft:(padding[1, 3]::NSUInteger)
            paddingRight:(padding[2, 3]::NSUInteger)
            paddingTop:(padding[1, 2]::NSUInteger)
            paddingBottom:(padding[2, 2]::NSUInteger)
            paddingFront:(padding[1, 1]::NSUInteger)
            paddingBack:(padding[2, 1]::NSUInteger)
            paddingStyle:(MPSGraphs.MPSGraphPaddingStyleExplicit::MPSGraphs.MPSGraphPaddingStyle)
            dataLayout:(MPSGraphs.MPSGraphTensorNamedDataLayoutNCDHW::MPSGraphs.MPSGraphTensorNamedDataLayout)
            weightsLayout:(MPSGraphs.MPSGraphTensorNamedDataLayoutOIDHW::MPSGraphs.MPSGraphTensorNamedDataLayout)
        ]::MPSGraphConvolution3DOpDescriptor
        @objc [
            graph::id{MPSGraph} convolution3DWithSourceTensor:(a::id{MPSGraphTensor})
            weightsTensor:(kernel::id{MPSGraphTensor})
            descriptor:(desc::id{MPSGraphConvolution3DOpDescriptor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end

    # the result is (batch, feature, spatial...) too
    perm = [
        dn.output_batch_dimension
        dn.output_feature_dimension
        dn.output_spatial_dimensions
    ]
    return permute(graph, conv, invperm(perm .+ 1) .- 1)
end

# the padding is only known at run time, so the input is padded by a gather: element `i` of a padded spatial dimension
# is element `i - low` of the input, or 0 outside of it. Only the elements that the convolution reads are gathered, so
# the high padding follows from the size of the result, and negative padding is supported too
function handle(
    ::Val{Symbol("stablehlo.dynamic_conv")},
    graph::MPSGraph,
    (a, kernel, padding),
    attributes,
    (result,),
)
    dn = attributes.dimension_numbers
    n = length(dn.input_spatial_dimensions)
    strides = get(attributes, :window_strides, ones(Int64, n))
    dilations = get(attributes, :rhs_dilation, ones(Int64, n))
    padding = MPSGraphs.castTensor(graph, padding, Int32)
    pad_value = MPSGraphs.constantWithScalar(graph, 0, result.eltype)
    imin = MPSGraphs.constantWithScalar(graph, 0, Int32)
    for i in 1:n
        dim = dn.input_spatial_dimensions[i]
        window = size(kernel)[dn.kernel_spatial_dimensions[i] + 1]
        out = result.size[dn.output_spatial_dimensions[i] + 1]
        shape = collect(Int64, size(a))
        imax = MPSGraphs.constantWithScalar(graph, shape[dim + 1] - 1, Int32)
        shape[dim + 1] = (out - 1) * strides[i] + (window - 1) * dilations[i] + 1

        # `padding` has shape (n, 2), so `low` has shape (1, 1)
        low = handle(
            Val(Symbol("stablehlo.slice")),
            graph,
            (padding,),
            (; start_indices=[i - 1, 0], limit_indices=[i, 1], strides=[1, 1]),
            (),
        )
        coords = @objc [
            graph::id{MPSGraph} coordinateAlongAxis:(dim::NSInteger)
            withShape:(convert(MPSShape, shape)::id{MPSShape})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        indices = handle(Val(Symbol("stablehlo.subtract")), graph, (coords, low), (;), ())
        clamped = @objc [
            graph::id{MPSGraph} clampWithTensor:(indices::id{MPSGraphTensor})
            minValueTensor:(imin::id{MPSGraphTensor})
            maxValueTensor:(imax::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        inside = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(indices::id{MPSGraphTensor})
            secondaryTensor:(clamped::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        gathered = @objc [
            graph::id{MPSGraph} gatherAlongAxis:(dim::NSInteger)
            withUpdatesTensor:(a::id{MPSGraphTensor})
            indicesTensor:(clamped::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        a = handle(
            Val(Symbol("stablehlo.select")), graph, (inside, gathered, pad_value), (;), ()
        )
    end
    return handle(
        Val(Symbol("stablehlo.convolution")), graph, (a, kernel), attributes, (result,)
    )
end

# MPSGraph scatters like `tf.scatter_nd`: the last dimension of `indices` holds the index of an element along the
# first dimensions of the operand, and the dimensions of `updates` are the other ones of `indices` followed by the
# rest of the operand
function handle(
    ::Val{Symbol("stablehlo.scatter")}, graph::MPSGraph, operands, attributes, _, (body,), _
)
    length(operands) == 3 || throw(
        unimplemented(
            "`stablehlo.scatter` of several tensors is not supported by `MetalClient`"
        ),
    )
    a, indices, updates = operands
    dn = attributes.scatter_dimension_numbers

    # the window of an update can't span the indexed dimensions
    indexed = dn.scatter_dims_to_operand_dims
    if sort(indexed) != dn.inserted_window_dims ||
        !isempty(dn.input_batching_dims) ||
        !isempty(dn.scatter_indices_batching_dims)
        throw(
            unimplemented(
                "`stablehlo.scatter` with these dimension numbers is not supported by `MetalClient`",
            ),
        )
    end

    perm = [indexed; setdiff(0:(length(size(a)) - 1), indexed)]
    a = permute(graph, a, perm)
    vector_dim = dn.index_vector_dim
    indices = permute(
        graph, indices, [setdiff(0:(length(size(indices)) - 1), vector_dim); vector_dim]
    )
    window = dn.update_window_dims
    updates = permute(
        graph, updates, [setdiff(0:(length(size(updates)) - 1), window); window]
    )

    # the region returns the new value of an element from its current value and the update
    value = IR.operand(last(collect(body)), 1)
    mode = if value.ref.ptr == IR.argument(body, 2).ref.ptr
        MPSGraphs.MPSGraphScatterModeSet
    elseif IR.is_op_res(value) && IR.name(IR.op_owner(value)) == "stablehlo.constant"
        # every element is set to the same constant (e.g. the one-hot seeds of forward-mode AD)
        value = decode_attribute(IR.getattr(IR.op_owner(value), "value"))
        constant = handle(
            Val(Symbol("stablehlo.constant")),
            graph,
            (),
            (; value),
            ((; eltype=eltype(value), size=Int64[]),),
        )
        updates = MPSGraphs.broadcastTensor(
            graph, constant, convert(MPSShape, size(updates))
        )
        MPSGraphs.MPSGraphScatterModeSet
    else
        opname = reducer(body)
        if opname == "stablehlo.add"
            MPSGraphs.MPSGraphScatterModeAdd
        elseif opname == "stablehlo.multiply"
            MPSGraphs.MPSGraphScatterModeMul
        elseif opname == "stablehlo.maximum"
            MPSGraphs.MPSGraphScatterModeMax
        elseif opname == "stablehlo.minimum"
            MPSGraphs.MPSGraphScatterModeMin
        else
            throw(
                unimplemented(
                    "`stablehlo.scatter` with `$opname` is not supported by `MetalClient`",
                ),
            )
        end
    end
    scattered = @objc [
        graph::id{MPSGraph} scatterNDWithDataTensor:(a::id{MPSGraphTensor})
        updatesTensor:(updates::id{MPSGraphTensor})
        indicesTensor:(indices::id{MPSGraphTensor})
        batchDimensions:(0::NSUInteger)
        mode:(mode::MPSGraphs.MPSGraphScatterMode)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return permute(graph, scattered, invperm(perm .+ 1) .- 1)
end
