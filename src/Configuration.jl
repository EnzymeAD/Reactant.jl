using ScopedValues: ScopedValues, ScopedValue

export with_config
export DotGeneralAlgorithmPreset, PrecisionConfig, DotGeneralAlgorithm

"""
    with_config(f; kwargs...)

Run the function `f` within a dynamic scope such that all uses of the config within this
scope will use the provided values.

# Extended Help

## Configuration Options

### Lowering

  - `lower_partialsort_to_approx_top_k`: Whether to lower `partialsort` and
    `partialsortperm` to `Ops.approx_top_k`. Note that XLA only supports lowering
    `ApproxTopK` for TPUs unless `fallback_approx_top_k_lowering` is set to `true`.
  - `fallback_approx_top_k_lowering`: Whether to lower `Ops.approx_top_k` to
    `stablehlo.top_k` if the XLA backend doesn't support `ApproxTopK`. Defaults to `true`.

### DotGeneral

  - `dot_general_algorithm`: Algorithm preset for `stablehlo.dot_general`. Can be `nothing`,
    [`DotGeneralAlgorithm`](@ref) or [`DotGeneralAlgorithmPreset`](@ref). Defaults to
    `DotGeneralAlgorithmPreset.DEFAULT`.
  - `dot_general_precision`: Precision for `stablehlo.dot_general`. Can be `nothing`,
    or [`PrecisionConfig`](@ref). Defaults to `PrecisionConfig.DEFAULT`.
  - `convolution_precision`: Precision for `stablehlo.convolution`. Can be `nothing`,
    or [`PrecisionConfig`](@ref). Defaults to `PrecisionConfig.DEFAULT`.

### Zygote Overlay

  - `overlay_zygote_calls`: Whether to overlay `Zygote.gradient` calls with
    `Enzyme.autodiff` calls. Defaults to `true`.
"""
function with_config(
    f;
    dot_general_algorithm=missing,
    dot_general_precision=missing,
    convolution_precision=missing,
    lower_partialsort_to_approx_top_k=missing,
    fallback_approx_top_k_lowering=missing,
    overlay_zygote_calls=missing,
)
    config_vars = ()
    dot_general_algorithm !== missing &&
        (config_vars = (config_vars..., DOT_GENERAL_ALGORITHM => dot_general_algorithm))
    dot_general_precision !== missing &&
        (config_vars = (config_vars..., DOT_GENERAL_PRECISION => dot_general_precision))
    convolution_precision !== missing &&
        (config_vars = (config_vars..., CONVOLUTION_PRECISION => convolution_precision))
    lower_partialsort_to_approx_top_k !== missing && (
        config_vars = (
            config_vars...,
            LOWER_PARTIALSORT_TO_APPROX_TOP_K => lower_partialsort_to_approx_top_k,
        )
    )
    fallback_approx_top_k_lowering !== missing && (
        config_vars = (
            config_vars...,
            FALLBACK_APPROX_TOP_K_LOWERING => fallback_approx_top_k_lowering,
        )
    )
    overlay_zygote_calls !== missing &&
        (config_vars = (config_vars..., OVERLAY_ZYGOTE_CALLS => overlay_zygote_calls))

    return ScopedValues.with(f, config_vars...)
end

# Lower to ApproxTopK
const LOWER_PARTIALSORT_TO_APPROX_TOP_K = ScopedValue(false)
const FALLBACK_APPROX_TOP_K_LOWERING = ScopedValue(true)

# DotGeneral Attributes Configuration
"""
    PrecisionConfig

Controls the `precision_config` for `stablehlo.dot_general`. Valid values are:

  - `DEFAULT`
  - `HIGH`
  - `HIGHEST`

The following functions are available:

  `MLIR.IR.Attribute(precision::PrecisionConfig.T)`
"""
@enumx PrecisionConfig begin
    DEFAULT
    HIGH
    HIGHEST
end

Base.@deprecate_binding DotGeneralPrecision PrecisionConfig

const DOT_GENERAL_PRECISION = ScopedValue{
    Union{PrecisionConfig.T,Nothing,Tuple{PrecisionConfig.T,PrecisionConfig.T}}
}(
    PrecisionConfig.DEFAULT
)

const CONVOLUTION_PRECISION = ScopedValue{
    Union{PrecisionConfig.T,Nothing,Tuple{PrecisionConfig.T,PrecisionConfig.T}}
}(
    PrecisionConfig.DEFAULT
)

function MLIR.IR.Attribute(precision::PrecisionConfig.T)
    precision_str = if precision == PrecisionConfig.DEFAULT
        "DEFAULT"
    elseif precision == PrecisionConfig.HIGH
        "HIGH"
    elseif precision == PrecisionConfig.HIGHEST
        "HIGHEST"
    end
    return MLIR.IR.Attribute(
        MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), precision_str)
    )
end

"""
    DotGeneralAlgorithm(
        ::Type{lhsT}, ::Type{rhsT}, ::Type{accumT},
        rhs_component_count::Int, lhs_component_count::Int, num_primitive_operations::Int,
        allow_imprecise_accumulation::Bool
    )
    DotGeneralAlgorithm{lhsT,rhsT,accumT}(
        lhs_component_count::Int, rhs_component_count::Int, num_primitive_operations::Int,
        allow_imprecise_accumulation::Bool
    )

Represents the configuration of the `stablehlo.dot_general` operation.

# Arguments

- `lhsT`: The type of the left-hand side operand.
- `rhsT`: The type of the right-hand side operand.
- `accumT`: The type of the accumulation operand.
- `lhs_component_count`: The number of components in the left-hand side operand.
- `rhs_component_count`: The number of components in the right-hand side operand.
- `num_primitive_operations`: The number of primitive operations in the
  `stablehlo.dot_general` operation.
"""
struct DotGeneralAlgorithm{lhsT<:ReactantFloat,rhsT<:ReactantFloat,accumT<:ReactantFloat}
    rhs_component_count::Int
    lhs_component_count::Int
    num_primitive_operations::Int
    allow_imprecise_accumulation::Bool
end

function DotGeneralAlgorithm(
    ::Type{lhsT},
    ::Type{rhsT},
    ::Type{accumT},
    rhs_component_count::Int,
    lhs_component_count::Int,
    num_primitive_operations::Int,
    allow_imprecise_accumulation::Bool,
) where {lhsT,rhsT,accumT}
    return DotGeneralAlgorithm{lhsT,rhsT,accumT}(
        rhs_component_count,
        lhs_component_count,
        num_primitive_operations,
        allow_imprecise_accumulation,
    )
end

function MLIR.IR.Attribute(
    algo::DotGeneralAlgorithm{lhsT,rhsT,accumT}
) where {lhsT,rhsT,accumT}
    return MLIR.IR.Attribute(
        MLIR.API.stablehloDotAlgorithmGet(
            MLIR.IR.context(),
            MLIR.IR.Type(lhsT),
            MLIR.IR.Type(rhsT),
            MLIR.IR.Type(accumT),
            algo.lhs_component_count,
            algo.rhs_component_count,
            algo.num_primitive_operations,
            algo.allow_imprecise_accumulation,
        ),
    )
end

"""
    DotGeneralAlgorithmPreset

Controls the `precision_config` for `stablehlo.dot_general`. Valid values are:

  - `DEFAULT`
  - `ANY_F8_ANY_F8_F32`
  - `ANY_F8_ANY_F8_F32_FAST_ACCUM`
  - `ANY_F8_ANY_F8_ANY`
  - `ANY_F8_ANY_F8_ANY_FAST_ACCUM`
  - `F16_F16_F16`
  - `F16_F16_F32`
  - `BF16_BF16_BF16`
  - `BF16_BF16_F32`
  - `BF16_BF16_F32_X3`
  - `BF16_BF16_F32_X6`
  - `BF16_BF16_F32_X9`
  - `F32_F32_F32`
  - `F64_F64_F64`

The following functions are available:

  `supported_lhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`
  `supported_rhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`
  `accumulation_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`
  `supported_output_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T, T1, T2)`
  `MLIR.IR.Attribute(dot_algorithm_preset::DotGeneralAlgorithmPreset.T, T1, T2)`
"""
@enumx DotGeneralAlgorithmPreset begin
    DEFAULT
    ANY_F8_ANY_F8_F32
    ANY_F8_ANY_F8_F32_FAST_ACCUM
    ANY_F8_ANY_F8_ANY
    ANY_F8_ANY_F8_ANY_FAST_ACCUM
    F16_F16_F16
    F16_F16_F32
    BF16_BF16_BF16
    BF16_BF16_F32
    BF16_BF16_F32_X3
    BF16_BF16_F32_X6
    BF16_BF16_F32_X9
    F32_F32_F32
    F64_F64_F64
    TF32_TF32_F32
    TF32_TF32_F32_X3
end

const DOT_GENERAL_ALGORITHM = ScopedValue{
    Union{DotGeneralAlgorithmPreset.T,Nothing,DotGeneralAlgorithm}
}(
    DotGeneralAlgorithmPreset.DEFAULT
)

function supported_lhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)
    if dot_algorithm_preset == DotGeneralAlgorithmPreset.DEFAULT ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
        return Number # Any Eltype
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F16_F16_F16 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.F16_F16_F32
        return Float16
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_BF16 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32_X3 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32_X6 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32_X9
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return Core.BFloat16
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F64_F64_F64
        return Float64
    end

    return Float32
end

function supported_rhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)
    return supported_lhs_eltype(dot_algorithm_preset)
end

function accumulation_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)
    if dot_algorithm_preset == DotGeneralAlgorithmPreset.DEFAULT ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM
        return nothing
    end

    dot_algorithm_preset == DotGeneralAlgorithmPreset.F16_F16_F16 && return Float16

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_BF16
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return Core.BFloat16
    end

    dot_algorithm_preset == DotGeneralAlgorithmPreset.F64_F64_F64 && return Float64

    return Float32
end

function supported_output_type(
    dot_algorithm_preset::DotGeneralAlgorithmPreset.T, ::Type{T1}, ::Type{T2}
) where {T1,T2}
    if dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
        return ReactantFloat
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F16_F16_F32
        return promote_type(T1, T2) == Float16 ? Union{Float16,Float32} : Float32
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return if promote_type(T1, T2) == Core.BFloat16
            Union{Core.BFloat16,Float32}
        else
            Float32
        end
    end

    accum_type = accumulation_eltype(dot_algorithm_preset)
    return accum_type === nothing ? nothing : accum_type
end

function DotGeneralAlgorithm(
    dot_algorithm_preset::DotGeneralAlgorithmPreset.T, ::Type{T1}, ::Type{T2}
) where {T1,T2}
    if dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
        if !(T1 <: ReactantFloat8 && T2 <: ReactantFloat8)
            error("Unsupported combination of types $T1 and $T2")
        end
        return DotGeneralAlgorithm{T1,T2,Float32}(
            1,
            1,
            1,
            dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM,
        )
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F16_F16_F16
        return DotGeneralAlgorithm{Float16,Float16,Float16}(1, 1, 1, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F16_F16_F32
        return DotGeneralAlgorithm{Float16,Float16,Float16}(1, 1, 1, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_BF16
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return DotGeneralAlgorithm{Core.BFloat16,Core.BFloat16,Core.BFloat16}(
            1, 1, 1, false
        )
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return DotGeneralAlgorithm{Core.BFloat16,Core.BFloat16,Float32}(1, 1, 1, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32_X3
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return DotGeneralAlgorithm{Core.BFloat16,Core.BFloat16,Float32}(1, 1, 3, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32_X6
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return DotGeneralAlgorithm{Core.BFloat16,Core.BFloat16,Float32}(1, 1, 6, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.BF16_BF16_F32_X9
        isdefined(Core, :BFloat16) || error("BFloat16 is not defined!")
        return DotGeneralAlgorithm{Core.BFloat16,Core.BFloat16,Float32}(1, 1, 9, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F32_F32_F32
        return DotGeneralAlgorithm{Float32,Float32,Float32}(1, 1, 1, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.F64_F64_F64
        return DotGeneralAlgorithm{Float64,Float64,Float64}(1, 1, 1, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.TF32_TF32_F32
        return DotGeneralAlgorithm{TF32,TF32,Float32}(1, 1, 1, false)
    end

    if dot_algorithm_preset == DotGeneralAlgorithmPreset.TF32_TF32_F32_X3
        return DotGeneralAlgorithm{TF32,TF32,Float32}(1, 1, 3, false)
    end

    return nothing
end

# Overlay Zygote.jl
const OVERLAY_ZYGOTE_CALLS = ScopedValue(true)
