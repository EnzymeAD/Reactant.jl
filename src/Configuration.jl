using ScopedValues: ScopedValues, ScopedValue

export DotGeneralAlgorithmPreset, DotGeneralPrecision, DotGeneralAlgorithm

# """
#     @with_config(f; kwargs...)

# Run the function `f` with the given configuration. The configuration is controlled using
# ScopedValues. Refer to the
# [official documentation for more details](https://docs.julialang.org/en/v1/base/scopedvalues/).
# """
# function with_config(f; dot_general_algorithm=missing, dot_general_precision=missing)
#     config_vars = (;)
#     dot_general_algorithm !== missing &&
#         (config_vars = (; config_vars..., dot_general_algorithm = dot_general_algorithm))
#     dot_general_precision !== missing &&
#         (config_vars = (; config_vars..., dot_general_precision = dot_general_precision))

#     @show config_vars

#     length(config_vars) == 0 && return f()
#     return ScopedValues.with(f, pairs(config_vars)...)
# end

# DotGeneral Attributes Configuration
"""
    DotGeneralPrecision

Controls the `precision_config` for `stablehlo.dot_general`. Valid values are:

  - `DEFAULT`
  - `HIGH`
  - `HIGHEST`

!!! note "ScopedValue"

    This is controlled using the `dot_general_precision` ScopedValue.

The following functions are available:

  `MLIR.IR.Attribute(precision::DotGeneralPrecision.T)`
"""
@enumx DotGeneralPrecision begin
    DEFAULT
    HIGH
    HIGHEST
end

const dot_general_precision = ScopedValue(DotGeneralPrecision.DEFAULT)

function MLIR.IR.Attribute(precision::DotGeneralPrecision.T)
    precision_str = if precision == DotGeneralPrecision.DEFAULT
        "DEFAULT"
    elseif precision == DotGeneralPrecision.HIGH
        "HIGH"
    elseif precision == DotGeneralPrecision.HIGHEST
        "HIGHEST"
    end
    return MLIR.IR.Attribute(
        MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), precision_str)
    )
end

"""
    DotGeneralAlgorithm{lhsT,rhsT,accumT}(
        lhs_component_count, rhs_component_count, num_primitive_operations,
        allow_imprecise_accumulation
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

supported_lhs_eltype(::DotGeneralAlgorithm{lhsT}) where {lhsT} = lhsT
supported_rhs_eltype(::DotGeneralAlgorithm{lhsT,rhsT}) where {lhsT,rhsT} = rhsT
function accumulation_eltype(
    ::DotGeneralAlgorithm{lhsT,rhsT,accumT}
) where {lhsT,rhsT,accumT}
    return accumT
end

function MLIR.IR.Attribute(algo::DotGeneralAlgorithm, ::Type, ::Type)
    return MLIR.IR.Attribute(
        MLIR.API.stablehloDotGeneralAlgorithmGet(
            MLIR.IR.context(),
            MLIR.IR.Type(supported_lhs_eltype(algo)),
            MLIR.IR.Type(supported_rhs_eltype(algo)),
            MLIR.IR.Type(accumulation_eltype(algo)),
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

!!! note "ScopedValue"

    This is controlled using the `dot_general_algorithm` ScopedValue.

The following functions are available:

  `supported_lhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`
  `supported_rhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`
  `accumulation_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`
  `supported_output_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T, T1, T2)`
  `MLIR.IR.Attribute(dot_algorithm_preset::DotGeneralAlgorithmPreset.T, T1, T2)`
"""
@enumx DotGeneralAlgorithmPreset begin
    "An algorithm will be selected based on input and output types."
    DEFAULT
    "Accepts any float8 input types and accumulates into float32."
    ANY_F8_ANY_F8_F32
    "Like `ANY_F8_ANY_F8_F32`, but using faster accumulation with the cost of lower \
     accuracy."
    ANY_F8_ANY_F8_F32_FAST_ACCUM
    "Like `ANY_F8_ANY_F8_F32`, but the accumulation type is controlled by \
     `preferred_element_type`."
    ANY_F8_ANY_F8_ANY
    "Like `ANY_F8_ANY_F8_F32_FAST_ACCUM`, but the accumulation type is controlled by \
     `preferred_element_type`."
    ANY_F8_ANY_F8_ANY_FAST_ACCUM
    F16_F16_F16
    F16_F16_F32
    BF16_BF16_BF16
    BF16_BF16_F32
    "The `_X3` suffix indicates that the algorithm uses 3 operations to emulate higher \
     precision."
    BF16_BF16_F32_X3
    "Like `BF16_BF16_F32_X3`, but using 6 operations instead of 3."
    BF16_BF16_F32_X6
    "Like `BF16_BF16_F32_X3`, but using 9 operations instead of 3."
    BF16_BF16_F32_X9
    F32_F32_F32
    F64_F64_F64
    TF32_TF32_F32
    TF32_TF32_F32_X3
end

const dot_general_algorithm = ScopedValue(DotGeneralAlgorithmPreset.DEFAULT)

function supported_lhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)
    if dot_algorithm_preset == DotGeneralAlgorithmPreset.DEFAULT ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32 ||
        dot_algorithm_preset == DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
        return Union{} # Any Eltype
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
        return nothing
        DotGeneralAlgorithm{T1,T2,Float32}(
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

function MLIR.IR.Attribute(
    dot_algorithm_preset::DotGeneralAlgorithmPreset.T, ::Type{T1}, ::Type{T2}
) where {T1,T2}
    algo = DotGeneralAlgorithm(dot_algorithm_preset, T1, T2)
    algo === nothing && return nothing
    return MLIR.IR.Attribute(algo, T1, T2)
end
