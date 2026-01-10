# This module reflects the HLO ops defined in the openxla/stablehlo repo (plus some extras).
# If you want to add some check or test, the StableHLO spec should be taken as the source of truth, not the Julia or Reactant semantics.
# Julia and Reactant semantics should be considered on the higher abstractions that use these
module Ops
using ..MLIR: MLIR
using ..MLIR.Dialects: stablehlo, chlo, enzyme, enzymexla
using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    RArray,
    RNumber,
    MissingTracedValue,
    unwrapped_eltype
using ReactantCore: ReactantCore

function _function_macro_error()
    throw(ArgumentError("`caller_function` is not available in this context"))
end

macro caller_function()
    return esc(
        quote
            $(Expr(:isdefined, :var"#self#")) || $(_function_macro_error)()
            var"#self#"
        end,
    )
end

"""
    @opcall fn(args...; kwargs...)

This call is expanded to `Reactant.Ops.fn(args...; kwargs..., location)` with the location
of the callsite. This enables better debug info to be propagated about the source location
of different  It is recommended to use this macro for calling into any function in
`Reactant.Ops.<function name>`.
"""
macro opcall(expr)
    if !isa(expr, Expr) || expr.head != :call
        error("@opcall expects a function call")
    end

    # Extract function name and arguments
    func = expr.args[1]
    args = expr.args[2:end]

    # Generate location info at the callsite
    location_expr = :($(mlir_stacktrace)(
        if @isdefined(var"#self#")
            joinpath(string(var"#self#"), $(string(func)))
        else
            $(string(func))
        end,
        $(string(__source__.file)),
        $(__source__.line),
    ))

    # Separate positional and keyword arguments
    pos_args = []
    kw_args = []

    for arg in args
        if isa(arg, Expr) && arg.head == :kw
            push!(kw_args, arg)
        elseif isa(arg, Expr) && arg.head == :parameters
            append!(kw_args, arg.args)
        else
            push!(pos_args, arg)
        end
    end

    # Add location as a keyword argument
    push!(kw_args, Expr(:kw, :location, location_expr))

    # Reconstruct the call expression
    func_full = getproperty(Ops, func)

    if isempty(kw_args)
        new_expr = Expr(:call, func_full, pos_args...)
    else
        new_expr = Expr(:call, func_full, Expr(:parameters, kw_args...), pos_args...)
    end

    return esc(new_expr)
end

function mlir_type(x::Union{RNumber,RArray})::MLIR.IR.Type
    return MLIR.IR.TensorType(collect(Int, size(x)), MLIR.IR.Type(unwrapped_eltype(x)))
end

mlir_type(::MissingTracedValue) = MLIR.IR.TensorType((), MLIR.IR.Type(Bool))

function mlir_type(RT::Type{<:RArray{T,N}}, shape) where {T,N}
    @assert length(shape) == N
    return MLIR.IR.TensorType(collect(Int, shape), MLIR.IR.Type(unwrapped_eltype(RT)))
end

function mlir_type(RT::Type{<:RNumber})::MLIR.IR.Type
    return MLIR.IR.TensorType(Int[], MLIR.IR.Type(unwrapped_eltype(RT)))
end

function mlir_type(::Type{MissingTracedValue})::MLIR.IR.Type
    return MLIR.IR.TensorType(Int[], MLIR.IR.Type(Bool))
end

const DEBUG_MODE::Ref{Bool} = Ref(false)
const LARGE_CONSTANT_THRESHOLD = Ref(100 << 20) # 100 MiB
const LARGE_CONSTANT_RAISE_ERROR = Ref(true)
const GATHER_GETINDEX_DISABLED = Ref(false)

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

function activate_constant_context!(blk::MLIR.IR.Block)
    stack = get!(task_local_storage(), :entry_block) do
        return Tuple{MLIR.IR.Block,Dict{MLIR.IR.Attribute,TracedRArray}}[]
    end
    Base.push!(stack, (blk, Dict{MLIR.IR.Attribute,TracedRArray}()))
    return nothing
end

function constant_context(; throw_error::Core.Bool=true)
    return last(task_local_storage(:entry_block))
end

function deactivate_constant_context!(blk::MLIR.IR.Block)
    constant_context()[1] == blk || error("Deactivating wrong block")
    return Base.pop!(task_local_storage(:entry_block))
end

# constant ops
@noinline function constant(
    x::DenseArray{T,N}; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T,N}
    if sizeof(x) > LARGE_CONSTANT_THRESHOLD[]
        if LARGE_CONSTANT_RAISE_ERROR[]
            error(
                "Generating a constant of $(sizeof(x)) bytes, which larger than the $(LARGE_CONSTANT_THRESHOLD[]) bytes threshold",
            )
        else
            location = with_debug() do
                mlir_stacktrace("constant", @__FILE__, @__LINE__)
            end
        end
    end

    value = MLIR.IR.DenseElementsAttribute(x)
    constant_blk, constants = constant_context()
    parent = MLIR.IR.parent_op(constant_blk)
    @assert MLIR.IR.name(parent) != "builtin.module"
    if haskey(constants, value)
        return constants[value]
    else
        output = mlir_type(TracedRArray{T,N}, size(x))

        op_ty_results = MLIR.IR.Type[output]
        operands = MLIR.IR.Value[]
        owned_regions = MLIR.IR.Region[]
        successors = MLIR.IR.Block[]
        attributes = MLIR.IR.NamedAttribute[MLIR.Dialects.namedattribute("value", value),]

        cstop = MLIR.IR.create_operation(
            "stablehlo.constant",
            location;
            operands,
            owned_regions,
            successors,
            attributes,
            results=op_ty_results,
            result_inference=false,
        )

        res = MLIR.IR.result(cstop)
        tres = TracedRArray{T,N}((), res, size(x))
        constants[value] = tres
        return tres
    end
end

@noinline function constant(
    x::AbstractArray{T,N}; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T,N}
    @assert !(x isa TracedRArray)
    return constant(collect(x); location)
end

@noinline function constant(x::Reactant.AbstractConcreteArray; kwargs...)
    return constant(Base.convert(Array, x); kwargs...)
end

@noinline function constant(
    x::T; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
) where {T<:Number}
    x isa TracedRNumber && return x
    res = fill(x; location)
    return TracedRNumber{T}((), res.mlir_data)
end

@noinline function constant(x::Reactant.AbstractConcreteNumber{T}; kwargs...) where {T}
    return constant(Base.convert(T, x); kwargs...)
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
    return fill(v, collect(Int64, dims); location)
end
function fill(v, ::Tuple{}; location=mlir_stacktrace("fill", @__FILE__, @__LINE__))
    return fill(v, Int[]; location)
end

function fill(
    number::TracedRNumber{T},
    shape::Vector{Int};
    location=mlir_stacktrace("fill", @__FILE__, @__LINE__),
) where {T}
    return broadcast_in_dim(
        TracedRArray{T,0}((), number.mlir_data, ()), Int64[], shape; location
    )
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

@noinline function concatenate(
    inputs::Vector{TracedRArray{T,N}},
    dimension::Int;
    location=mlir_stacktrace("fill", @__FILE__, @__LINE__),
) where {T,N}
    concat_inputs = Vector{MLIR.IR.Value}(undef, length(inputs))
    for (i, inp) in enumerate(inputs)
        @inbounds concat_inputs[i] = inp.mlir_data
    end
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.concatenate(concat_inputs; dimension=(dimension - 1)), 1
    )
    return TracedRArray{T,N}((), res, size(MLIR.IR.type(res)))
end

@noinline function fill(
    element::T, shape::Vector{Int}; location=mlir_stacktrace("fill", @__FILE__, @__LINE__)
) where {T<:AbstractIrrational}
    return fill(float(element), shape; location)
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

@noinline function conj(
    x::TracedRArray{T,N}; location=mlir_stacktrace("conj", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        chlo.conj(x.mlir_data; result=mlir_type(TracedRArray{T,N}, size(x)), location)
    )
    return TracedRArray{T,N}((), res, size(x))
end

@noinline function conj(
    x::TracedRNumber{T}; location=mlir_stacktrace("conj", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        chlo.conj(x.mlir_data; result=mlir_type(TracedRArray{T,0}, ()), location)
    )
    return TracedRNumber{T}((), res)
end

@noinline function conj(
    x::TracedRArray{T,N}; location=mlir_stacktrace("conj", @__FILE__, @__LINE__)
) where {T<:Real,N}
    return TracedRArray{T,N}((), x.mlir_data, size(x))
end

@noinline function conj(
    x::TracedRNumber{T}; location=mlir_stacktrace("conj", @__FILE__, @__LINE__)
) where {T<:Real}
    return TracedRNumber{T}((), x.mlir_data)
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
@noinline function reshape(x::TracedRArray, dims::Integer...; kwargs...)
    return reshape(x, collect(Int64, dims); kwargs...)
end

@noinline function reshape(
    x::TracedRArray{T,N},
    dims::Vector{Int};
    location=mlir_stacktrace("reshape", @__FILE__, @__LINE__),
) where {T,N}
    # HLO reshape semantics collapse the opposite way
    res1 = transpose(x, Int64[N:-1:1...])
    restype = mlir_type(TracedRArray{T,length(dims)}, collect(Int64, Base.reverse(dims)))
    res = MLIR.IR.result(stablehlo.reshape(res1.mlir_data; result_0=restype, location))
    result = TracedRArray{T,length(dims)}((), res, collect(Int64, Base.reverse(dims)))
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
    @assert length(permutation) == ndims(x)
    rsize = permute!(collect(Int64, size(x)), permutation)
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
    start_indices::Vector{<:Integer},
    limit_indices::Vector{<:Integer};
    strides::Union{Nothing,Vector{<:Integer}}=nothing,
    location=mlir_stacktrace("slice", @__FILE__, @__LINE__),
) where {T,N}
    start_indices = start_indices .- 1
    limit_indices = limit_indices
    @assert all(Base.Fix2(≥, 0), start_indices) "Invalid start indices: $(start_indices)"
    @assert all(s < l for (s, l) in zip(start_indices, limit_indices)) "Invalid slice indices: $(start_indices), $(limit_indices)"

    strides = isnothing(strides) ? ones(Int64, N) : strides
    @assert all(s > 0 for s in strides) "Invalid strides: $(strides)"
    rsize = [
        length((start + 1):st:stop) for
        (start, stop, st) in zip(start_indices, limit_indices, strides)
    ]
    @assert all(rsize .> 0) "Invalid slice dimensions"

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
    x::TracedRArray{T,N}; location=mlir_stacktrace("real", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.real(
            x.mlir_data; result=mlir_type(TracedRArray{Base.real(T),N}, size(x)), location
        ),
    )
    return TracedRArray{Base.real(T),N}((), res, size(x))
end

@noinline function real(
    x::TracedRNumber{T}; location=mlir_stacktrace("real", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.real(
            x.mlir_data; result=mlir_type(TracedRArray{Base.real(T),0}, ()), location
        ),
    )
    return TracedRNumber{Base.real(T)}((), res)
end

@noinline function imag(
    x::TracedRArray{T,N}; location=mlir_stacktrace("imag", @__FILE__, @__LINE__)
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.imag(
            x.mlir_data; result=mlir_type(TracedRArray{Base.real(T),N}, size(x)), location
        ),
    )
    return TracedRArray{Base.real(T),N}((), res, size(x))
end

@noinline function imag(
    x::TracedRNumber{T}; location=mlir_stacktrace("imag", @__FILE__, @__LINE__)
) where {T}
    res = MLIR.IR.result(
        stablehlo.imag(
            x.mlir_data; result=mlir_type(TracedRArray{Base.real(T),0}, ()), location
        ),
    )
    return TracedRNumber{Base.real(T)}((), res)
end

function bitcast_convert(
    ::Type{TracedRArray{U,N}},
    x::TracedRArray{T,N};
    location=mlir_stacktrace("bitcast_convert", @__FILE__, @__LINE__),
) where {T,U,N}
    res = MLIR.IR.result(
        stablehlo.bitcast_convert(
            x.mlir_data; result_0=mlir_type(TracedRArray{U,N}, size(x)), location
        ),
    )
    return TracedRArray{U,N}((), res, size(x))
end

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

# TODO: See https://github.com/jax-ml/jax/blob/6c18aa8a468e35b8c11b101dceaa43d05b497177/jax/_src/numpy/fft.py#L106
@noinline function fft(
    x::TracedRArray{T,N};
    type::String,
    length,
    location=mlir_stacktrace("fft", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 <= Base.length(length) <= 3 "fft only supports up to rank 3"

    if type ∈ ("FFT", "IFFT")
        if !(T <: Complex)
            x = complex(x, fill(T(0), size(x); location); location)
        end
        Tout = Base.complex(T)
        rsize = size(x)
    elseif type == "RFFT"
        @assert T <: Real
        Tout = Complex{T}
        rsize = let rsize = collect(Int64, size(x))
            rsize[end] = rsize[end] == 0 ? 0 : rsize[end] ÷ 2 + 1
            Tuple(rsize)
        end
    elseif type == "IRFFT"
        if !(T <: Complex)
            x = complex(x, fill(T(0), size(x); location); location)
        end
        Tout = Base.real(T)
        rsize = let rsize = collect(Int64, size(x))
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
    min::T, x::Union{TracedRArray{T},TracedRNumber{T}}, max::T; kwargs...
) where {T}
    return clamp(constant(min), x, constant(max); kwargs...)
end

@noinline function convolution(
    result_size::Vector{Int64},
    lhs::TracedRArray{T,N},
    rhs::TracedRArray{T,N};
    input_batch_dim::Int64,
    input_feature_dim::Int64,
    input_spatial_dims::Vector{Int64},
    kernel_input_dim::Int64,
    kernel_output_dim::Int64,
    kernel_spatial_dims::Vector{Int64},
    output_batch_dim::Int64,
    output_feature_dim::Int64,
    output_spatial_dims::Vector{Int64},
    padding::Matrix{Int64},
    feature_group_count::Int64,
    batch_group_count::Int64,
    window_strides::Union{Vector{Int64},Nothing}=nothing,
    lhs_dilation::Union{Vector{Int64},Nothing}=nothing,
    rhs_dilation::Union{Vector{Int64},Nothing}=nothing,
    precision_config=Reactant.CONVOLUTION_PRECISION[],
    location=mlir_stacktrace("convolution", @__FILE__, @__LINE__),
) where {T,N}
    num_spatial_dims = N - 2
    @assert length(input_spatial_dims) == num_spatial_dims
    @assert length(kernel_spatial_dims) == num_spatial_dims
    @assert length(output_spatial_dims) == num_spatial_dims
    @assert size(padding, 1) == 2
    @assert size(padding, 2) == num_spatial_dims

    dimension_numbers = MLIR.API.stablehloConvDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(input_batch_dim - 1),
        Int64(input_feature_dim - 1),
        length(input_spatial_dims),
        Int64[i - 1 for i in input_spatial_dims],
        Int64(kernel_input_dim - 1),
        Int64(kernel_output_dim - 1),
        length(kernel_spatial_dims),
        Int64[i - 1 for i in kernel_spatial_dims],
        Int64(output_batch_dim - 1),
        Int64(output_feature_dim - 1),
        length(output_spatial_dims),
        Int64[i - 1 for i in output_spatial_dims],
    )

    if precision_config !== nothing
        if precision_config isa Reactant.PrecisionConfig.T
            precision_config = (precision_config, precision_config)
        end

        @assert precision_config isa Union{Tuple,Vector}
        @assert length(precision_config) == 2
        @assert all(Base.Fix2(isa, Reactant.PrecisionConfig.T), precision_config)
    end

    conv = MLIR.Dialects.stablehlo.convolution(
        lhs.mlir_data,
        rhs.mlir_data;
        result_0=MLIR.IR.TensorType(result_size, MLIR.IR.Type(T)),
        window_strides,
        padding=MLIR.IR.DenseElementsAttribute(padding'),
        dimension_numbers,
        lhs_dilation,
        rhs_dilation,
        feature_group_count,
        batch_group_count,
        precision_config=MLIR.IR.Attribute([
            MLIR.IR.Attribute(precision_config[1]), MLIR.IR.Attribute(precision_config[2])
        ]),
        location,
    )

    return TracedRArray{T,N}((), MLIR.IR.result(conv), result_size)
end

Base.@nospecializeinfer @noinline function dot_general(
    @nospecialize(lhs::TracedRArray{T1}),
    @nospecialize(rhs::TracedRArray{T2});
    contracting_dimensions,
    batching_dimensions=(Int[], Int[]),
    precision_config=Reactant.DOT_GENERAL_PRECISION[],
    algorithm=Reactant.DOT_GENERAL_ALGORITHM[],
    location=mlir_stacktrace("dot_general", @__FILE__, @__LINE__),
) where {T1,T2}
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
    if !isnothing(precision_config)
        if precision_config isa Reactant.PrecisionConfig.T
            precision_config = (precision_config, precision_config)
        end

        @assert precision_config isa Union{Tuple,Vector}
        @assert length(precision_config) == 2
        @assert all(Base.Fix2(isa, Reactant.PrecisionConfig.T), precision_config)
    end

    resT = promote_type(T1, T2)

    if algorithm isa Reactant.DotGeneralAlgorithmPreset.T
        lhs_eltype = Reactant.supported_lhs_eltype(algorithm)
        @assert T1 <: lhs_eltype "$(T1) is not a subtype of $(lhs_eltype)"
        @assert T2 <: lhs_eltype "$(T2) is not a subtype of $(lhs_eltype)"
        rhs_eltype = Reactant.supported_rhs_eltype(algorithm)
        @assert resT <: rhs_eltype "$(resT) is not a subtype of $(rhs_eltype)"

        algorithm = Reactant.DotGeneralAlgorithm(algorithm, T1, T2)
    end

    @assert algorithm isa Reactant.DotGeneralAlgorithm || algorithm === nothing

    if !isnothing(algorithm)
        # C22 + C23
        @assert algorithm.rhs_component_count ≥ 0
        @assert algorithm.lhs_component_count ≥ 0

        # C24
        @assert algorithm.num_primitive_operations > 0
    end

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
    ctx = MLIR.IR.context()

    dot_dimension_numbers = GC.@preserve ctx lhs_contracting_dimensions rhs_contracting_dimensions lhs_batching_dimensions rhs_batching_dimensions begin
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
            MLIR.IR.Attribute(precision_config[1]), MLIR.IR.Attribute(precision_config[2])
        ])
    end

    algorithm = algorithm !== nothing ? MLIR.IR.Attribute(algorithm) : nothing

    res = MLIR.IR.result(
        stablehlo.dot_general(
            lhs.mlir_data,
            rhs.mlir_data;
            result_0=mlir_type(TracedRArray{resT,length(ressize)}, ressize),
            dot_dimension_numbers,
            precision_config,
            algorithm,
            location,
        ),
    )
    return TracedRArray{resT,length(ressize)}((), res, ressize)
end

# parallel ops
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
    @assert length(result_size) ≥ N

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

    sample_inputs = Vector{TracedRNumber}(undef, length(xs) * 2)
    for i in eachindex(xs)
        T = Reactant.unwrapped_eltype(xs[i])
        sample_inputs[2i - 1] = Reactant.promote_to(TracedRNumber{T}, 0)
        sample_inputs[2i] = Reactant.promote_to(TracedRNumber{T}, 0)
    end
    func =
        Reactant.TracedUtils.make_mlir_fn(
            comparator,
            (sample_inputs...,),
            (),
            "comparator",
            false;
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
    @assert MLIR.IR.result(ftype) == MLIR.IR.TensorType(Int[], MLIR.IR.Type(Bool)) error(
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

@noinline function approx_top_k(
    x::TracedRArray{T,N},
    k::Integer;
    comparator,
    init_val::T,
    dimension::Integer=N,
    recall_target::AbstractFloat=0.95f0,
    reduction_input_size_override::Int64=-1,
    aggregate_to_topk::Bool=true,
    fallback::Union{Missing,Bool}=missing,
    location=mlir_stacktrace("approx_top_k", @__FILE__, @__LINE__),
) where {T<:AbstractFloat,N}
    fallback === missing && (fallback = Reactant.FALLBACK_APPROX_TOP_K_LOWERING[])

    func =
        Reactant.TracedUtils.make_mlir_fn(
            comparator,
            (
                Reactant.promote_to(TracedRNumber{T}, 0),
                Reactant.promote_to(TracedRNumber{T}, 0),
                Reactant.promote_to(TracedRNumber{Int32}, 0),
                Reactant.promote_to(TracedRNumber{Int32}, 0),
            ),
            (),
            "comparator",
            false;
            args_in_result=:none,
            return_dialect=:stablehlo,
        ).f
    @assert MLIR.IR.nregions(func) == 1
    fn_name = MLIR.IR.FlatSymbolRefAttribute(
        String(MLIR.IR.attr(func, String(MLIR.API.mlirSymbolTableGetSymbolAttributeName())))
    )

    iota_arg = iota(Int32, collect(Int64, size(x)); iota_dimension=dimension, location)
    init_arg = constant(Int32(-1); location)
    init_val = constant(init_val; location)

    dimension ≤ 0 && (dimension += N)

    backend_config = Dict(
        "reduction_dim" => MLIR.IR.Attribute(dimension - 1),
        "recall_target" => MLIR.IR.Attribute(Float32(recall_target)),
        "reduction_input_size_override" => MLIR.IR.Attribute(reduction_input_size_override),
        "top_k" => MLIR.IR.Attribute(k),
        "aggregate_to_topk" => MLIR.IR.Attribute(aggregate_to_topk),
    )
    fallback && (backend_config["is_fallback"] = MLIR.IR.Attribute(true))

    result_shape = collect(Int64, size(x))
    result_shape[dimension] = k

    out = stablehlo.custom_call(
        [x.mlir_data, iota_arg.mlir_data, init_val.mlir_data, init_arg.mlir_data];
        result_0=[
            mlir_type(TracedRArray{T,N}, result_shape),
            mlir_type(TracedRArray{Int32,N}, result_shape),
        ],
        call_target_name="ApproxTopK",
        called_computations=[fn_name],
        backend_config,
        api_version=Int32(4),
    )

    indices = add(
        TracedRArray{Int32,N}((), MLIR.IR.result(out, 2), result_shape),
        fill(Int32(1), Tuple(result_shape)), # return the 1-indexed index
    ) # stablehlo.approx_top_k returns 0-indexed indices
    indices = convert(TracedRArray{Int64,N}, indices) # julia indexes with Int64 generally

    values = TracedRArray{T,N}((), MLIR.IR.result(out, 1), result_shape)

    return (; values, indices)
end

@noinline function top_k(
    x::TracedRArray{T,N},
    k;
    dimension::Integer=N,
    location=mlir_stacktrace("top_k", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 <= dimension <= N

    # XLA codegen for top.k is extremely sub-optimal. For special cases we can bypass that
    if k isa Integer && k == 1
        values, indices = argmax(x; dimension, location)
        return (;
            values, indices=add(indices, fill(Int64(1), Tuple(size(indices))); location)
        )
    end

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

# Taken from https://github.com/JuliaGPU/GPUArrays.jl/blob/49a339c63a50f1a00ac84844675bcb3a11070cb0/src/host/indexing.jl#L193
@noinline function findfirst(
    x::TracedRArray{Bool,N};
    dimension::Integer=N,
    location=mlir_stacktrace("findfirst", @__FILE__, @__LINE__),
) where {N}
    return reduce(
        TracedRArray[
            x, iota(Int64, collect(Int64, size(x)); iota_dimension=dimension, location)
        ],
        TracedRNumber[
            Reactant.promote_to(TracedRNumber, false),
            Reactant.promote_to(TracedRNumber, typemax(Int64)),
        ],
        [dimension],
        function (x, i, y, j)
            cond_val = x | y
            idx = ifelse(x, ifelse(i < j, i, j), ifelse(y, j, typemax(Int64)))
            return cond_val, idx
        end;
        location,
    )[2] .+ 1
end

@noinline function argmax(
    x::TracedRArray{T,N};
    dimension::Integer=N,
    location=mlir_stacktrace("argmax", @__FILE__, @__LINE__),
) where {T,N}
    values, indices = reduce(
        TracedRArray[
            x, iota(Int64, collect(Int64, size(x)); iota_dimension=dimension, location)
        ],
        TracedRNumber[
            Reactant.promote_to(TracedRNumber{T}, typemin(T)),
            Reactant.promote_to(TracedRNumber{Int64}, -1),
        ],
        [dimension],
        function (a₁, i₁, a₂, i₂)
            cond = a₁ ≥ a₂
            return ifelse(cond, a₁, a₂), ifelse(cond, i₁, i₂)
        end;
        location,
    )
    new_shape = collect(Int64, size(x))
    new_shape[dimension] = 1
    return (reshape(values, new_shape; location), reshape(indices, new_shape; location))
end

@noinline function iota(
    T::Type,
    shape::Vector{Int};
    iota_dimension,
    location=mlir_stacktrace("iota", @__FILE__, @__LINE__),
)
    N = length(shape)
    @assert 0 < iota_dimension <= N
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
            dimensions=MLIR.IR.DenseArrayAttribute(collect(Int64, dimensions .- 1)),
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
        minval::Union{T,Nothing}=nothing,
        maxval::Union{T,Nothing}=nothing,
        algorithm::String="DEFAULT",
        location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
    )

Generate a random array of type `T` with the given shape and seed from a uniform random
distribution between `[minval, maxval)` (for floating point types). Returns a NamedTuple
with the following fields:

- `output_state`: The state of the random number generator after the operation.
- `output`: The generated array.

# Arguments

- `T`: The type of the generated array.
- `seed`: The seed for the random number generator.
- `shape`: The shape of the generated array.
- `minval`: The minimum value of the generated random numbers. (Only for floating point
  types). Defaults to `0`.
- `maxval`: The maximum value of the generated random numbers. (Only for floating point
  types). Defaults to `1`.
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to
  "DEFAULT". Other options include "PHILOX" and "THREE_FRY".
"""
@noinline function rng_bit_generator(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    minval::Union{T,Nothing}=nothing,
    maxval::Union{T,Nothing}=nothing,
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rng_bit_generator", @__FILE__, @__LINE__),
) where {T<:Integer}
    @assert algorithm in ("DEFAULT", "PHILOX", "THREE_FRY")
    @assert minval === nothing "minval is not supported for integer rng_bit_generator"
    @assert maxval === nothing "maxval is not supported for integer rng_bit_generator"
    if algorithm == "PHILOX"
        @assert length(seed) ∈ (2, 3)
    elseif algorithm == "THREE_FRY"
        @assert length(seed) == 2
    end

    output = MLIR.IR.TensorType(collect(Int, shape), MLIR.IR.Type(T))
    output_state = MLIR.IR.TensorType(collect(Int, size(seed)), MLIR.IR.Type(UInt64))
    rng_algorithm = MLIR.API.stablehloRngAlgorithmAttrGet(MLIR.IR.context(), algorithm)
    op = stablehlo.rng_bit_generator(
        seed.mlir_data; output, output_state, rng_algorithm, location
    )
    return (;
        output_state=TracedRArray{UInt64,1}((), MLIR.IR.result(op, 1), size(seed)),
        output=TracedRArray{T,length(shape)}((), MLIR.IR.result(op, 2), Tuple(shape)),
    )
end

function _get_uint_from_bitwidth(width::Int)
    @assert width ∈ (8, 16, 32, 64) "Unsupported bitwidth: $width"
    return width == 8 ? UInt8 : (width == 16 ? UInt16 : (width == 32 ? UInt32 : UInt64))
end

# https://github.com/jax-ml/jax/blob/474dcd409d6fa4c048014851922460f9d4fc199e/jax/_src/random.py#L444-L464
@noinline function rng_bit_generator(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    minval::T=T(0),
    maxval::T=T(1),
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rng_bit_generator", @__FILE__, @__LINE__),
) where {T<:AbstractFloat}
    nbits = sizeof(T) * 8
    nmantissa = Reactant.nmantissa(T)
    rng_bits = nbits
    nmantissa < 8 && (rng_bits = 8)
    uint_gen_dtype = _get_uint_from_bitwidth(rng_bits)
    (; output_state, output) = rng_bit_generator(
        uint_gen_dtype, seed, shape; algorithm, location
    )
    uint_dtype = _get_uint_from_bitwidth(nbits)
    bits = output
    if rng_bits != nbits
        bits = convert(TracedRArray{uint_dtype,length(shape)}, bits)
    end

    float_bits = or(
        shift_right_logical(
            bits, fill(uint_dtype(rng_bits - nmantissa), size(bits); location); location
        ),
        fill(reinterpret(uint_dtype, T(1)), size(bits); location);
        location,
    )
    floats = subtract(
        bitcast_convert(TracedRArray{T,length(shape)}, float_bits; location),
        fill(T(1), size(output); location);
        location,
    )

    maxval = prevfloat(maxval) # make maxval exclusive
    minval_ = fill(minval, size(floats); location)
    maxval_ = fill(maxval, size(floats); location)
    output = clamp(
        minval_,
        add(
            multiply(floats, subtract(maxval_, minval_; location); location),
            minval_;
            location,
        ),
        maxval_;
        location,
    )
    return (; output_state, output)
end

@noinline function rng_bit_generator(
    ::Type{TracedRNumber{T}}, seed::TracedRArray{UInt64,1}, shape; kwargs...
) where {T}
    return rng_bit_generator(T, seed, shape; kwargs...)
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
    location=mlir_stacktrace("randn", @__FILE__, @__LINE__),
) where {T<:AbstractFloat}
    res = rng_bit_generator(
        T, seed, shape; algorithm, location, minval=nextfloat(T(-1)), maxval=T(1)
    )
    rand_uniform = res.output
    seed = res.output_state
    probit = erf_inv(rand_uniform)
    rand_normal = multiply(probit, fill(Base.sqrt(T(2)), size(rand_uniform)))
    return (; output_state=seed, output=rand_normal)
end

@noinline function randn(
    ::Type{Complex{T}},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("randn", @__FILE__, @__LINE__),
) where {T<:AbstractFloat}
    real_result = randn(T, seed, shape; algorithm, location)
    imag_result = randn(T, real_result.output_state, shape; algorithm, location)
    output = complex.(real_result.output, imag_result.output)
    return (; output_state=imag_result.output_state, output)
end

@noinline function randn(
    ::Type{TracedRNumber{T}}, seed::TracedRArray{UInt64,1}, shape; kwargs...
) where {T}
    return randn(T, seed, shape; kwargs...)
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
) where {T<:AbstractFloat}
    res = rng_bit_generator(T, seed, shape; algorithm, location)
    rand_uniform = res.output
    seed = res.output_state
    rand_exp = negate(log_plus_one(negate(rand_uniform)))
    return (; output_state=seed, output=rand_exp)
end

@noinline function randexp(
    ::Type{TracedRNumber{T}}, seed::TracedRArray{UInt64,1}, shape; kwargs...
) where {T}
    return randexp(T, seed, shape; kwargs...)
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
    on_false::TracedRArray{T,N};
    location=mlir_stacktrace("select", @__FILE__, @__LINE__),
) where {T,N}
    @assert size(on_true) == size(on_false) "`on_true` and `on_false` must have the same size"
    @assert size(pred) == size(on_true) || size(pred) == () "`pred` must have the same size as `on_true`/`on_false` or be a scalar"

    res = MLIR.IR.result(
        stablehlo.select(
            pred.mlir_data,
            on_true.mlir_data,
            on_false.mlir_data;
            result=mlir_type(TracedRArray{T,N}, size(on_true)),
            location,
        ),
    )
    return TracedRArray{T,N}((), res, size(on_true))
end

@noinline function select(
    pred::TracedRNumber{Bool},
    on_true::TracedRNumber{T},
    on_false::TracedRNumber{T};
    location=mlir_stacktrace("select", @__FILE__, @__LINE__),
) where {T}
    res = MLIR.IR.result(
        stablehlo.select(
            pred.mlir_data,
            on_true.mlir_data,
            on_false.mlir_data;
            result=mlir_type(TracedRArray{T,0}, ()),
            location,
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
    hlo_call(mlir_code::String, args::Vararg{AnyTracedRArray}...; func_name::String="main") -> NTuple{N, AnyTracedRArray}

Given a MLIR module given as a string, calls the function identified by the `func_name` keyword parameter (default "main")
with the provided arguments and return a tuple for each result of the call.

```julia-repl
julia> Reactant.@jit(
          hlo_call(
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

    @assert all(Base.Fix2(isa, Union{TracedRArray,TracedRNumber}), args) "hlo_call: all inputs to hlo_call should be reactant arrays or numbers"
    @assert MLIR.IR.ninputs(ftype) == length(args) "hlo_call: invalid number of arguments for function $func_name"

    for (i, arg) in enumerate(args)
        expected_type = MLIR.IR.input(ftype, i)
        arg_type = MLIR.IR.type(Reactant.TracedUtils.get_mlir_data(arg))
        @assert expected_type == arg_type "hlo_call: argument #$i has the wrong type (expected $expected_type, got $arg_type)"
    end

    operands = MLIR.IR.Value[Reactant.TracedUtils.get_mlir_data(a) for a in args]
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
    scatter_indices::TracedRArray{T1,2},
    updates::TracedRArray{T2,1};
    location=mlir_stacktrace("scatter_setindex", @__FILE__, @__LINE__),
) where {T,N,T1,T2}
    @assert length(updates) == size(scatter_indices, 1)
    @assert size(scatter_indices, 2) == N

    return scatter(
        (a, b) -> b,
        [dest],
        scatter_indices,
        [convert(TracedRArray{T,1}, updates)];
        update_window_dims=Int64[],
        inserted_window_dims=collect(Int64, 1:N),
        input_batching_dims=Int64[],
        scatter_indices_batching_dims=Int64[],
        scatter_dims_to_operand_dims=collect(Int64, 1:N),
        index_vector_dim=Int64(2),
        location,
    )[1]
end

@noinline function scatter(
    f::F,
    dest::Vector{TracedRArray{T,N}},
    scatter_indices::TracedRArray{Int64},
    updates::Vector{<:TracedRArray{T}};
    location=mlir_stacktrace("scatter", @__FILE__, @__LINE__),
    kwargs...,
) where {F,T,N}
    sample_inputs = (
        Reactant.promote_to(TracedRNumber, zero(T)),
        Reactant.promote_to(TracedRNumber, zero(T)),
    )

    compiled_fn =
        Reactant.TracedUtils.make_mlir_fn(
            f,
            sample_inputs,
            (),
            "update_computation",
            false;
            args_in_result=:result,
            return_dialect=:stablehlo,
        ).f
    update_computation = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(update_computation, MLIR.IR.region(compiled_fn, 1))
    MLIR.IR.rmfromparent!(compiled_fn)

    return scatter(dest, scatter_indices, updates; update_computation, location, kwargs...)
end

@noinline function scatter(
    dest::Vector{TracedRArray{T,N}},
    scatter_indices::TracedRArray{TI},
    updates::Vector{<:TracedRArray{T}};
    update_computation::MLIR.IR.Region,
    update_window_dims::Vector{Int64},
    inserted_window_dims::Vector{Int64},
    input_batching_dims::Vector{Int64},
    scatter_indices_batching_dims::Vector{Int64},
    scatter_dims_to_operand_dims::Vector{Int64},
    index_vector_dim::Int64,
    unique_indices::Union{Bool,Nothing}=nothing,
    indices_are_sorted::Union{Bool,Nothing}=nothing,
    location=mlir_stacktrace("scatter", @__FILE__, @__LINE__),
) where {T,TI,N}
    scatter_indices = subtract(
        scatter_indices, fill(TI(1), size(scatter_indices)); location
    )

    update_window_dims = update_window_dims .- 1
    inserted_window_dims = inserted_window_dims .- 1
    input_batching_dims = input_batching_dims .- 1
    scatter_indices_batching_dims = scatter_indices_batching_dims .- 1
    scatter_dims_to_operand_dims = scatter_dims_to_operand_dims .- 1
    index_vector_dim -= 1

    #! format: off
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

    dest_values = [d.mlir_data for d in dest]
    update_values = [u.mlir_data for u in updates]
    scatter_op = stablehlo.scatter(
        dest_values,
        scatter_indices.mlir_data,
        update_values;
        update_computation,
        scatter_dimension_numbers,
        result_0=[mlir_type(TracedRArray{T,N}, size(d)) for d in dest],
        unique_indices,
        indices_are_sorted,
        location,
    )

    return [
        TracedRArray{T,N}((), MLIR.IR.result(scatter_op, i), size(dest[i])) for
        i in eachindex(dest)
    ]
end

"""
    gather_getindex(src, gather_indices)

Uses [`MLIR.Dialects.stablehlo.gather`](@ref) to get the values of `src` at the indices
specified by `gather_indices`. If the indices are contiguous it is recommended to directly
use [`MLIR.Dialects.stablehlo.dynamic_slice`](@ref) instead.
"""
@noinline function gather_getindex(
    src::TracedRArray{T,N},
    gather_indices::TracedRArray{TI,2};
    location=mlir_stacktrace("gather_getindex", @__FILE__, @__LINE__),
) where {T,TI,N}
    @assert size(gather_indices, 2) == N

    if GATHER_GETINDEX_DISABLED[]
        GPUArraysCore.assertscalar(
            "gather_getindex(::TracedRArray, ::TracedRArray{Int64,2}"
        )
    end

    return reshape(
        gather(
            src,
            gather_indices;
            offset_dims=Int64[1],
            collapsed_slice_dims=collect(Int64, 1:(N - 1)),
            operand_batching_dims=Int64[],
            start_indices_batching_dims=Int64[],
            start_index_map=collect(Int64, 1:N),
            index_vector_dim=Int64(2),
            slice_sizes=ones(Int64, N),
            indices_are_sorted=false,
            location,
        ),
        [size(gather_indices, 1)],
    )
end

@noinline function gather(
    src::TracedRArray{T,N},
    gather_indices::TracedRArray{TI};
    offset_dims::Vector{Int64},
    collapsed_slice_dims::Vector{Int64},
    operand_batching_dims::Vector{Int64},
    start_indices_batching_dims::Vector{Int64},
    start_index_map::Vector{Int64},
    index_vector_dim::Int64,
    slice_sizes::Vector{Int64},
    indices_are_sorted::Bool=false,
    location=mlir_stacktrace("gather", @__FILE__, @__LINE__),
) where {T,TI,N}
    gather_indices = subtract(gather_indices, fill(TI(1), size(gather_indices)); location)

    offset_dims = offset_dims .- 1
    start_indices_batching_dims = start_indices_batching_dims .- 1
    start_index_map = start_index_map .- 1
    operand_batching_dims = operand_batching_dims .- 1
    collapsed_slice_dims = collapsed_slice_dims .- 1
    index_vector_dim -= 1

    #! format: off
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

    return TracedRArray{T}(
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.gather(
                src.mlir_data,
                gather_indices.mlir_data;
                dimension_numbers,
                slice_sizes,
                indices_are_sorted,
                location,
            ),
            1,
        ),
    )
end

@noinline function while_loop(
    cond_fn::CFn,
    body_fn::BFn,
    args...;
    track_numbers,
    verify_arg_names=nothing,
    checkpointing=false,
    mincut=false,
    location=mlir_stacktrace("while_loop", @__FILE__, @__LINE__),
) where {CFn,BFn}
    # TODO: detect and prevent mutation within the condition

    # Make all the args traced or concrete
    N = length(args)
    seen_args = Reactant.OrderedIdDict()
    traced_args = Vector{Any}(undef, N)

    for (i, prev) in enumerate(args)
        @inbounds traced_args[i] = Reactant.make_tracer(
            seen_args, prev, (), Reactant.NoStopTracedTrack; track_numbers
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
            args_in_result=:result,
            do_transpose=false,
            argprefix=gensym("loop_condarg"),
            resprefix=gensym("loop_condres"),
            resargprefix=gensym("loop_condresarg"),
        ).f

    body_fn_compiled =
        Reactant.TracedUtils.make_mlir_fn(
            body_fn,
            traced_args,
            (),
            string(gensym("body_fn")),
            false;
            return_dialect=:stablehlo,
            args_in_result=:all,
            do_transpose=false,
            verify_arg_names,
            argprefix=gensym("loop_bodyarg"),
            resprefix=gensym("loop_bodyres"),
            resargprefix=gensym("loop_bodyresarg"),
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
        location,
    )

    if !mincut
        MLIR.IR.attr!(while_op, "enzyme.disable_mincut", MLIR.IR.UnitAttribute())
    end

    if checkpointing
        MLIR.IR.attr!(while_op, "enzymexla.enable_checkpointing", MLIR.IR.Attribute(true))
    end

    return map(enumerate(linear_args)) do (i, arg)
        Reactant.TracedUtils.set_mlir_data!(arg, MLIR.IR.result(while_op, i))
    end
end

@noinline function if_condition(
    cond::TracedRNumber{Bool},
    true_fn::TFn,
    false_fn::FFn,
    args...;
    track_numbers,
    location=mlir_stacktrace("if_condition", @__FILE__, @__LINE__),
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
            track_numbers,
        )
        @inbounds fb_traced_args[i] = Reactant.make_tracer(
            fb_seen_args,
            args[i],
            (false_fn_names[1], i),
            Reactant.TracedSetPath;
            track_numbers,
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
    activate_constant_context!(true_fn_body)
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
        deactivate_constant_context!(true_fn_body)
        MLIR.IR.deactivate!(true_fn_body)
    end

    seen_true_results = Reactant.OrderedIdDict()
    traced_true_results = Reactant.make_tracer(
        seen_true_results,
        tb_result,
        (true_fn_names[2],),
        Reactant.NoStopTracedTrack;
        track_numbers,
    )
    for (i, arg) in enumerate(tb_traced_args)
        Reactant.make_tracer(
            seen_true_results,
            arg,
            (true_fn_names[3], i),
            Reactant.NoStopTracedTrack;
            track_numbers,
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
    activate_constant_context!(false_fn_body)
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
        deactivate_constant_context!(false_fn_body)
        MLIR.IR.deactivate!(false_fn_body)
    end

    seen_false_results = Reactant.OrderedIdDict()
    traced_false_results = Reactant.make_tracer(
        seen_false_results,
        fb_result,
        (false_fn_names[2],),
        Reactant.NoStopTracedTrack;
        track_numbers,
    )
    for (i, arg) in enumerate(fb_traced_args)
        Reactant.make_tracer(
            seen_false_results,
            arg,
            (false_fn_names[3], i),
            Reactant.NoStopTracedTrack;
            track_numbers,
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

    @assert length(tb_paths) == length(all_paths)
    @assert length(fb_paths) == length(all_paths)

    # finalize the true branch by adding the missing values
    MLIR.IR.activate!(true_fn_body)
    activate_constant_context!(true_fn_body)
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
        deactivate_constant_context!(true_fn_body)
    end

    # finalize the false branch by adding the missing values
    MLIR.IR.activate!(false_fn_body)
    activate_constant_context!(false_fn_body)
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
        deactivate_constant_context!(false_fn_body)
    end

    # All MissingTracedValues must be replaced with zeroes
    @assert length(tb_corrected_linear_results) == length(fb_corrected_linear_results)

    @assert length(all_paths) == length(tb_corrected_linear_results)
    @assert length(all_paths) == length(fb_corrected_linear_results)

    result_types = MLIR.IR.Type[]
    both_missing = Set{Int}()
    for (i, (tr, fr)) in
        enumerate(zip(tb_corrected_linear_results, fb_corrected_linear_results))
        res = if tr isa MissingTracedValue && fr isa MissingTracedValue
            z = zero(TracedRNumber{Int})
            tb_corrected_linear_results[i] = z
            fb_corrected_linear_results[i] = z
            z
        elseif tr isa MissingTracedValue
            @assert !(fr isa MissingTracedValue)
            MLIR.IR.activate!(true_fn_body)
            activate_constant_context!(true_fn_body)
            try
                tb_corrected_linear_results[i] = zero(fr)
            finally
                MLIR.IR.deactivate!(true_fn_body)
                deactivate_constant_context!(true_fn_body)
            end
            fr
        elseif fr isa MissingTracedValue
            @assert !(tr isa MissingTracedValue)
            MLIR.IR.activate!(false_fn_body)
            activate_constant_context!(false_fn_body)
            try
                fb_corrected_linear_results[i] = zero(tr)
            finally
                MLIR.IR.deactivate!(false_fn_body)
                deactivate_constant_context!(false_fn_body)
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

    @assert length(all_paths) == length(result_types) + length(both_missing)

    MLIR.IR.activate!(true_fn_body)
    activate_constant_context!(true_fn_body)
    try
        vals = MLIR.IR.Value[
            Reactant.TracedUtils.get_mlir_data(res) for
            res in tb_corrected_linear_results if !(res isa MissingTracedValue)
        ]
        MLIR.Dialects.stablehlo.return_(vals; location)
    finally
        MLIR.IR.deactivate!(true_fn_body)
        deactivate_constant_context!(true_fn_body)
    end

    MLIR.IR.activate!(false_fn_body)
    activate_constant_context!(false_fn_body)
    try
        vals = MLIR.IR.Value[
            Reactant.TracedUtils.get_mlir_data(res) for
            res in fb_corrected_linear_results if !(res isa MissingTracedValue)
        ]
        MLIR.Dialects.stablehlo.return_(vals; location)
    finally
        MLIR.IR.deactivate!(false_fn_body)
        deactivate_constant_context!(false_fn_body)
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

    @assert length(all_paths) == length(result_types) + length(both_missing)
    if_compiled = MLIR.Dialects.stablehlo.if_(
        cond.mlir_data;
        true_branch=tb_region,
        false_branch=fb_region,
        result_0=result_types,
        location,
    )

    corrected_traced_results =
        map(zip(traced_false_results, traced_true_results)) do (fr, tr)
            if fr isa MissingTracedValue && tr isa MissingTracedValue
                return fr
            elseif fr isa MissingTracedValue
                return tr
            else
                return fr
            end
        end

    @assert length(all_paths) == length(result_types)

    residx = 0
    for (i, path) in enumerate(all_paths)
        if path[1] == :result
            residx += 1
            Reactant.TracedUtils.set!(
                corrected_traced_results, path[2:end], MLIR.IR.result(if_compiled, residx)
            )
        elseif path[1] == :resarg
            residx += 1
            Reactant.TracedUtils.set!(
                args, path[2:end], MLIR.IR.result(if_compiled, residx)
            )
        end
    end

    return corrected_traced_results
end

@noinline function call(f, args...; location=mlir_stacktrace("call", @__FILE__, @__LINE__))
    seen = Reactant.OrderedIdDict()
    cache_key = []
    Reactant.make_tracer(seen, (f, args...), cache_key, Reactant.TracedToTypes)
    cache = Reactant.Compiler.callcache()
    if haskey(cache, cache_key)
        # cache lookup:
        (; f_name, mlir_result_types, traced_result, mutated_args, linear_results, fnwrapped, argprefix, resprefix, resargprefix) = cache[cache_key]
    else
        f_name = String(gensym(Symbol(f)))

        argprefix::Symbol = gensym("callarg")
        resprefix::Symbol = gensym("calllresult")
        resargprefix::Symbol = gensym("callresarg")

        temp = Reactant.TracedUtils.make_mlir_fn(
            f,
            args,
            (),
            f_name,
            false;
            args_in_result=:all,
            do_transpose=false,
            argprefix,
            resprefix,
            resargprefix,
        )
        (; traced_result, ret, mutated_args, linear_results, fnwrapped) = temp
        mlir_result_types = [
            MLIR.IR.type(MLIR.IR.operand(ret, i)) for i in 1:MLIR.IR.noperands(ret)
        ]
        cache[cache_key] = (;
            f_name,
            mlir_result_types,
            traced_result,
            mutated_args,
            linear_results,
            fnwrapped,
            argprefix,
            resprefix,
            resargprefix,
        )
    end

    seen_cache = Reactant.OrderedIdDict()
    Reactant.make_tracer(
        seen_cache,
        fnwrapped ? (f, args) : args,
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

    call_op = MLIR.Dialects.func.call(
        mlir_caller_args;
        result_0=mlir_result_types,
        callee=MLIR.IR.FlatSymbolRefAttribute(f_name),
        location,
    )

    seen_results = Reactant.OrderedIdDict()
    traced_result = Reactant.make_tracer(
        seen_results,
        traced_result,
        (), # we have to insert something here, but we remove it immediately below.
        Reactant.TracedSetPath;
        toscalar=false,
    )

    for r in seen_results
        if r isa TracedRNumber || r isa TracedRArray
            r.paths = ()
        end
    end

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(call_op, i)
        for path in res.paths
            if length(path) == 0
                continue
            end
            if path[1] == resprefix
                Reactant.TracedUtils.set!(traced_result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if idx == 1 && fnwrapped
                    Reactant.TracedUtils.set!(f, path[3:end], resv)
                else
                    if fnwrapped
                        idx -= 1
                    end
                    Reactant.TracedUtils.set!(args[idx], path[3:end], resv)
                end
            end
        end
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
        logical_device_ids::Vector{Int64};
        sym_name::String="mesh",
        mod::MLIR.IR.Module=MLIR.IR.mmodule(),
        location=mlir_stacktrace("mesh", @__FILE__, @__LINE__)
    )

Produces a [`Reactant.MLIR.Dialects.sdy.mesh`](@ref) operation with the given `mesh` and
`logical_device_ids`.

Based on the provided `sym_name``, we generate a unique name for the mesh in the module's
`SymbolTable`. Note that users shouldn't use this sym_name directly, instead they should
use the returned `sym_name` to refer to the mesh in the module.

!!! warning

    The `logical_device_ids` argument are the logical device ids, not the physical device
    ids. For example, if the physical device ids are `[2, 4, 123, 293]`, the corresponding
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
    key = (m.logical_device_ids, m.axis_names, size(m))
    cache !== nothing && haskey(cache, key) && return cache[key]
    result = mesh(
        [k => Int64(v) for (k, v) in zip(m.axis_names, size(m))],
        m.logical_device_ids;
        mod,
        sym_name,
        location,
    )
    cache !== nothing && (cache[key] = merge(result, (; mesh=m)))
    return result
end

@noinline function mesh(
    mesh_axes::Vector{<:Pair{<:Union{String,Symbol},Int64}},
    logical_device_ids::AbstractVector{Int64};
    mod::MLIR.IR.Module=MLIR.IR.mmodule(),
    sym_name::String="mesh",
    location=mlir_stacktrace("mesh", @__FILE__, @__LINE__),
)
    # See https://github.com/openxla/shardy/blob/f9d83e779a58b811b848c4edfaf68e88b636787d/shardy/dialect/sdy/ir/verifiers.cc#L647-L699 for the checks
    ndevices = prod(last, mesh_axes)

    @assert allunique(first, mesh_axes) "mesh_axes must be unique"
    @assert ndevices == length(logical_device_ids) "length(logical_device_ids) should be \
                                                    same as prod(last, mesh_axes)"
    @assert all(Base.Fix2(≥, 0), logical_device_ids) "logical_device_ids must be \
                                                      non-negative"

    sorted_logical_device_ids = Base.sort(logical_device_ids)
    @assert sorted_logical_device_ids == 0:(ndevices - 1) "sorted logical_device_ids \
                                                           must be the same \
                                                           as iota(product(axes)), got \
                                                           $(sorted_logical_device_ids)"

    # error: if the ordered device ids are the same as iota(product(axes)), no need to
    # specify them for simplicity
    logical_device_ids == sorted_logical_device_ids && (logical_device_ids = Int64[])

    ctx = MLIR.IR.context()
    mesh_axis_attrs = [
        MLIR.API.sdyMeshAxisAttrGet(ctx, String(name), size) for (name, size) in mesh_axes
    ]
    mesh_attr = MLIR.API.sdyMeshAttrGet(
        ctx,
        Int64(length(mesh_axis_attrs)),
        mesh_axis_attrs,
        Int64(length(logical_device_ids)),
        collect(Int64, logical_device_ids),
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
    key = (sharding.mesh.logical_device_ids, sharding.mesh.axis_names, size(sharding.mesh))
    haskey(cache, key) || mesh(sharding.mesh; location)
    (; sym_name, mesh_attr) = cache[key]

    tensor_sharding_attr, dialect = Reactant.Sharding.get_tensor_sharding_attribute(
        sharding, MLIR.IR.context(), sym_name, mesh_attr, size(input); do_transpose=false
    )
    @assert dialect == :sdy "Expected dialect to be `sdy`, got $(dialect)"

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

function _construct_reduce_function(f::F, Ts::Type...) where {F}
    inputs_1 = [Reactant.promote_to(TracedRNumber{T}, 0) for T in Ts]
    inputs_2 = [Reactant.promote_to(TracedRNumber{T}, 0) for T in Ts]
    func =
        Reactant.TracedUtils.make_mlir_fn(
            f,
            (inputs_1..., inputs_2...),
            (),
            "reduce_fn" * string(f),
            false;
            args_in_result=:none,
            return_dialect=:stablehlo,
        ).f

    @assert MLIR.IR.nregions(func) == 1
    ftype_attr = MLIR.IR.attr(func, "function_type")
    ftype = MLIR.IR.Type(ftype_attr)

    @assert MLIR.IR.nresults(ftype) == length(Ts)
    for i in 1:MLIR.IR.nresults(ftype)
        tType = MLIR.IR.TensorType(Int[], MLIR.IR.Type(Ts[i]))
        @assert MLIR.IR.result(ftype, i) == tType "$(f) return type $(i) is not of \
                                                   tensor<$(Ts[i])>"
    end

    fn = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(fn, MLIR.IR.region(func, 1))
    MLIR.IR.rmfromparent!(func)

    return fn
end

"""
    reduce(
        x::TracedRArray{T},
        init_values::TracedRNumber{T},
        dimensions::Vector{Int},
        fn::Function,
        location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
    )

Applies a reduction function `fn` along the specified `dimensions` of input `x`, starting from `init_values`.

# Arguments

- `x`: The input array.
- `init_values`: The initial value.
- `dimensions`: The dimensions to reduce along.
- `fn`: A binary operator.

!!! warning
    This reduction operation follows StableHLO semantics. The key difference between this operation and Julia's built-in `reduce` is explained below:

    - The function `fn` and the initial value `init_values` must form a **monoid**, meaning:
      - `fn` must be an **associative** binary operation.
      - `init_values` must be the **identity element** associated with `fn`.
    - This constraint ensures consistent results across all implementations.

    If `init_values` is not the identity element of `fn`, the results may vary between CPU and GPU executions. For example:

    ```julia
    A = [1 3; 2 4;;; 5 7; 6 8;;; 9 11; 10 12]
    init_values = 2
    dimensions = [1, 3]
    ```

    - **CPU version & Julia's `reduce`**:
      - Reduce along dimension 1 → `[(15) (21); (18) (24)]`
      - Reduce along dimension 3 → `[(33 + 2)  (45 + 2)]` → `[35 47]`

    - **GPU version**:
      - Reduce along dimension 1 → `[(15 + 2) (21 + 2); (18 + 2) (24 + 2)]`
      - Reduce along dimension 3 → `[37 49]`
"""
@noinline function reduce(
    x::TracedRArray{T},
    init_values::TracedRNumber{T},
    dimensions::Vector{Int},
    fn::F;
    location=mlir_stacktrace("reduce", @__FILE__, @__LINE__),
) where {T,F}
    return only(reduce([x], [init_values], dimensions, fn; location))
end

@noinline function reduce(
    xs::Vector{<:TracedRArray},
    init_values::Vector{<:TracedRNumber},
    dimensions::Vector{Int},
    fn::F;
    location=mlir_stacktrace("reduce", @__FILE__, @__LINE__),
) where {F}
    @assert allequal(size.(xs)) "All input arrays must have the same size."

    reduced_shape = Tuple(deleteat!(collect(Int64, size(xs[1])), dimensions))

    op = stablehlo.reduce(
        [x.mlir_data for x in xs],
        [init_value.mlir_data for init_value in init_values];
        result_0=[
            mlir_type(
                TracedRArray{unwrapped_eltype(x),length(reduced_shape)}, reduced_shape
            ) for x in xs
        ],
        dimensions=MLIR.IR.Attribute(dimensions .- 1),
        body=_construct_reduce_function(fn, [unwrapped_eltype(x) for x in xs]...),
        location,
    )

    return [
        TracedRArray{unwrapped_eltype(xs[i]),length(reduced_shape)}(
            (), MLIR.IR.result(op, i), reduced_shape
        ) for i in 1:MLIR.IR.nresults(op)
    ]
end

function standardize_start_index(
    sz::Int,
    update_sz::Union{Int,Nothing},
    start_index::Union{Integer,TracedRNumber{<:Integer}},
    idx::Integer,
)
    if (start_index isa Integer && start_index ≤ typemax(Int32)) || sz ≤ typemax(Int32)
        if start_index isa Integer && update_sz !== nothing
            @assert start_index + update_sz - 1 ≤ sz "Index $(idx) out of bounds: \
                                                      start_index=$(start_index), \
                                                      update_sz=$(update_sz), sz=$(sz)"
        end
        start_index = Reactant.promote_to(TracedRNumber{Int32}, start_index)
    elseif start_index isa Integer && update_sz !== nothing
        @assert start_index + update_sz - 1 ≤ sz "Index $(idx) out of bounds: \
                                                  start_index=$(start_index), \
                                                  update_sz=$(update_sz), sz=$(sz)"
        start_index = Reactant.promote_to(TracedRNumber, start_index)
    end

    start_index = start_index - Reactant.unwrapped_eltype(start_index)(1)
    return start_index
end

function standardize_start_indices(
    operand::TracedRArray{T,N}, update, start_indices::Vector
) where {T,N}
    @assert length(start_indices) == N
    return [
        standardize_start_index(
            size(operand, i),
            update === nothing ? nothing : size(update, i),
            start_indices[i],
            i,
        ).mlir_data for i in 1:N
    ]
end

@noinline function dynamic_update_slice(
    operand::TracedRArray{T,N},
    update::TracedRArray{T},
    start_indices::Vector;
    location=mlir_stacktrace("dynamic_update_slice", @__FILE__, @__LINE__),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.dynamic_update_slice(
            operand.mlir_data,
            update.mlir_data,
            standardize_start_indices(operand, update, start_indices);
            location,
        ),
        1,
    )
    return TracedRArray{T,N}((), res, size(res))
end

@noinline function dynamic_slice(
    operand::TracedRArray{T,N},
    start_indices::Vector,
    slice_sizes::Vector;
    location=mlir_stacktrace("dynamic_slice", @__FILE__, @__LINE__),
) where {T,N}
    res = MLIR.IR.result(
        stablehlo.dynamic_slice(
            operand.mlir_data,
            standardize_start_indices(operand, nothing, start_indices);
            slice_sizes=collect(Int64, slice_sizes),
            location,
        ),
        1,
    )
    return TracedRArray{T,ndims(res)}((), res, size(res))
end

# Currently this is very simplistic and doesn't linearize/delinearize and supports only
# a single argument (similar to how Julia's mapslices works)
@noinline function batch(
    f::F,
    A::TracedRArray{T,N},
    dims::Vector{Int};
    location=mlir_stacktrace("batch", @__FILE__, @__LINE__),
) where {F,T,N}
    sort!(dims)

    # First we permute and make sure the batch dims are at the beginning
    batch_dims = Int64[i for i in 1:N if i ∉ dims]
    batch_shape = [size(A, i) for i in batch_dims]
    permutation = zeros(Int64, N)
    for (i, d) in enumerate(batch_dims)
        permutation[i] = d
    end
    for (i, d) in enumerate(dims)
        permutation[i + length(batch_dims)] = d
    end

    res = only(batch(f, [transpose(A, permutation; location)], batch_shape; location))
    if ndims(res) != length(permutation)
        res = reshape(
            res,
            vcat(collect(Int64, size(res)), ones(Int64, length(permutation) - ndims(res))),
        )
    end
    return transpose(res, invperm(permutation); location)
end

@noinline function batch(
    f::F,
    inputs::Vector{<:TracedRArray},
    batch_shape::Vector{Int64};
    location=mlir_stacktrace("batch", @__FILE__, @__LINE__),
) where {F}
    sample_inputs = [
        fill(
            unwrapped_eltype(input)(0),
            [size(input, i) for i in (length(batch_shape) + 1):ndims(input)]...,
        ) for input in inputs
    ]
    argprefix = gensym("batcharg")
    mlir_fn_res = Reactant.TracedUtils.make_mlir_fn(
        f,
        (sample_inputs...,),
        (),
        "unbatched_" * string(f),
        false;
        args_in_result=:result,
        do_transpose=false,
        argprefix,
    )

    func = mlir_fn_res.f
    @assert MLIR.IR.nregions(func) == 1

    if mlir_fn_res.fnwrapped
        # In the long-term we should be able to do per-argument batching.
        # Rn we simply broadcast_in_dim the arguments to the correct shape.
        final_inputs = TracedRArray[]
        seenargs = Reactant.OrderedIdDict()
        Reactant.make_tracer(
            seenargs, f, (argprefix, 1), Reactant.TracedSetPath; toscalar=false
        )
        for (k, v) in seenargs
            v isa Reactant.TracedType || continue
            bcasted_arg = broadcast_in_dim(
                v,
                collect(Int64, (length(batch_shape) + 1):(ndims(v) + length(batch_shape))),
                vcat(batch_shape, collect(Int64, size(v)));
                location,
            )
            push!(final_inputs, bcasted_arg)
        end
        append!(final_inputs, inputs)
    else
        final_inputs = inputs
    end

    output_types = MLIR.IR.Type[]
    for result in mlir_fn_res.linear_results
        push!(
            output_types,
            MLIR.IR.TensorType(
                vcat(batch_shape, collect(Int64, size(result))),
                MLIR.IR.Type(unwrapped_eltype(result)),
            ),
        )
    end

    return batch(final_inputs, output_types, batch_shape; fn=func, location)
end

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

function triangular_solve(
    a::TracedRArray{T,N},
    b::TracedRArray{T,M};
    left_side::Bool,
    location=mlir_stacktrace("triangular_solve", @__FILE__, @__LINE__),
    kwargs...,
) where {T,N,M}
    @assert M == N - 1

    if left_side
        b = reshape(b, size(b)..., 1)
    else
        b = reshape(b, size(b)[1:(M - 1)]..., 1, size(b, M))
    end

    return dropdims(
        triangular_solve(a, b; location, left_side, kwargs...); dims=(N - 1 + left_side)
    )
end

function triangular_solve(
    a::TracedRArray{T,N},
    b::TracedRArray{T,N};
    left_side::Bool,
    lower::Bool,
    transpose_a::Char,
    unit_diagonal::Bool,
    location=mlir_stacktrace("triangular_solve", @__FILE__, @__LINE__),
) where {T,N}
    @assert N >= 2
    @assert size(a, N - 1) == size(a, N) == size(b, N - left_side)
    @assert size(a)[1:(N - 2)] == size(b)[1:(N - 2)] "a and b must have the same leading \
                                                      dimensions"

    @assert transpose_a in ('N', 'T', 'C') "transpose_a must be one of 'N', 'T', or 'C'"
    transpose_attr = MLIR.API.stablehloTransposeAttrGet(
        MLIR.IR.context(),
        if transpose_a == 'N'
            "NO_TRANSPOSE"
        elseif transpose_a == 'T'
            "TRANSPOSE"
        else
            "ADJOINT"
        end,
    )

    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.triangular_solve(
            a.mlir_data,
            b.mlir_data;
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_attr,
            unit_diagonal=unit_diagonal,
            location,
        ),
        1,
    )

    return TracedRArray{T,N}((), res, size(res))
end

"""
    lu(
        x::TracedRArray{T},
        ::Type{pT}=Int32;
        location=mlir_stacktrace("lu", @__FILE__, @__LINE__)
    ) where {T,pT}

Compute the row maximum pivoted LU factorization of `x` and return the factors `LU`,
`ipiv`, `permutation` tensor, and `info`.
"""
@noinline function lu(
    x::TracedRArray{T},
    ::Type{pT}=Int32;
    location=mlir_stacktrace("lu", @__FILE__, @__LINE__),
) where {T,pT}
    @assert ndims(x) >= 2

    output_shape = collect(Int64, size(x))
    batch_shape = output_shape[1:(end - 2)]
    pivots_shape = vcat(batch_shape, min(size(x, ndims(x) - 1), size(x, ndims(x))))
    permutation_shape = vcat(batch_shape, size(x, ndims(x) - 1))
    info_shape = batch_shape

    op = enzymexla.linalg_lu(
        x.mlir_data;
        output=MLIR.IR.TensorType(output_shape, MLIR.IR.Type(unwrapped_eltype(T))),
        pivots=MLIR.IR.TensorType(pivots_shape, MLIR.IR.Type(pT)),
        permutation=MLIR.IR.TensorType(permutation_shape, MLIR.IR.Type(pT)),
        info=MLIR.IR.TensorType(info_shape, MLIR.IR.Type(pT)),
        location,
    )

    res = TracedRArray{T,ndims(x)}((), MLIR.IR.result(op, 1), size(x))
    ipiv = TracedRArray{pT,ndims(x) - 1}((), MLIR.IR.result(op, 2), pivots_shape)
    perm = TracedRArray{pT,ndims(x) - 1}((), MLIR.IR.result(op, 3), permutation_shape)

    if ndims(x) == 2
        info = TracedRNumber{pT}((), MLIR.IR.result(op, 4))
    else
        info = TracedRArray{pT,ndims(x) - 2}((), MLIR.IR.result(op, 4), info_shape)
    end
    return (res, ipiv, perm, info)
end

@noinline function svd(
    x::TracedRArray{T,N},
    ::Type{iT}=Int32;
    full::Bool=false,
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("svd", @__FILE__, @__LINE__),
) where {T,iT,N}
    @assert N >= 2

    batch_sizes = size(x)[1:(end - 2)]
    m, n = size(x)[(end - 1):end]
    r = min(m, n)

    U_size = (batch_sizes..., m, full ? m : r)
    S_size = (batch_sizes..., r)
    Vt_size = (batch_sizes..., full ? n : r, n)
    info_size = batch_sizes

    if algorithm == "DEFAULT"
        algint = 0
    elseif algorithm == "QRIteration"
        algint = 1
    elseif algorithm == "DivideAndConquer"
        algint = 2
    elseif algorithm == "Jacobi"
        algint = 3
    else
        error("Unsupported SVD algorithm: $algorithm")
    end

    svd_op = enzymexla.linalg_svd(
        x.mlir_data;
        U=mlir_type(TracedRArray{T,N}, U_size),
        S=mlir_type(TracedRArray{Base.real(T),N - 1}, S_size),
        Vt=mlir_type(TracedRArray{T,N}, Vt_size),
        info=mlir_type(TracedRArray{iT,N - 2}, info_size),
        full=full,
        algorithm=MLIR.API.enzymexlaSVDAlgorithmAttrGet(MLIR.IR.context(), algint),
        location,
    )

    U = TracedRArray{T,N}((), MLIR.IR.result(svd_op, 1), U_size)
    S = TracedRArray{Base.real(T),N - 1}((), MLIR.IR.result(svd_op, 2), S_size)
    Vt = TracedRArray{T,N}((), MLIR.IR.result(svd_op, 3), Vt_size)

    if N == 2
        info = TracedRNumber{iT}((), MLIR.IR.result(svd_op, 4))
    else
        info = TracedRArray{iT,N - 2}((), MLIR.IR.result(svd_op, 4), info_size)
    end

    return U, S, Vt, info
end

@noinline function reduce_window(
    f::F,
    inputs::Vector{TracedRArray{T,N}},
    init_values::Vector{TracedRNumber{T}};
    window_dimensions::Vector{Int},
    window_strides::Vector{Int},
    base_dilations::Vector{Int},
    window_dilations::Vector{Int},
    padding_low::Vector{Int},
    padding_high::Vector{Int},
    output_shape::Vector{Int},
    location=mlir_stacktrace("reduce_window", @__FILE__, @__LINE__),
) where {F,T,N}
    @assert length(inputs) == length(init_values)
    @assert length(window_dimensions) ==
        length(window_strides) ==
        length(base_dilations) ==
        length(window_dilations) ==
        length(padding_low) ==
        length(padding_high) ==
        N

    reduction = stablehlo.reduce_window(
        [inp.mlir_data for inp in inputs],
        [init.mlir_data for init in init_values];
        result_0=[
            mlir_type(TracedRArray{T,length(output_shape)}, output_shape) for
            _ in 1:length(inputs)
        ],
        window_dimensions,
        window_strides,
        base_dilations,
        window_dilations,
        padding=MLIR.IR.DenseElementsAttribute(hcat(padding_low, padding_high)),
        body=_construct_reduce_function(f, T),
        location,
    )

    return [
        TracedRArray{T,length(output_shape)}(
            (), MLIR.IR.result(reduction, i), output_shape
        ) for i in 1:length(inputs)
    ]
end

@noinline function batch_norm_inference(
    operand::TracedRArray{T,N},
    scale::Union{TracedRArray{T,1},Nothing},
    offset::Union{TracedRArray{T,1},Nothing},
    mean::TracedRArray{T,1},
    variance::TracedRArray{T,1};
    epsilon,
    feature_index::Int64,
    location=mlir_stacktrace("batch_norm_inference", @__FILE__, @__LINE__),
) where {T,N}
    len = size(operand, feature_index)
    @assert length(mean) == length(variance) == len

    if scale === nothing
        scale = fill(T(1), len; location)
    else
        @assert size(scale) == (len,)
    end

    if offset === nothing
        offset = fill(T(0), len; location)
    else
        @assert size(offset) == (len,)
    end

    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            stablehlo.batch_norm_inference(
                operand.mlir_data,
                scale.mlir_data,
                offset.mlir_data,
                mean.mlir_data,
                variance.mlir_data;
                epsilon=Float32(epsilon),
                feature_index=feature_index - 1,
                location,
            ),
            1,
        ),
        size(operand),
    )
end

@noinline function batch_norm_training(
    operand::TracedRArray{T,N},
    scale::Union{TracedRArray{T,1},Nothing},
    offset::Union{TracedRArray{T,1},Nothing};
    epsilon,
    feature_index::Int64,
    location=mlir_stacktrace("batch_norm_training", @__FILE__, @__LINE__),
) where {T,N}
    len = size(operand, feature_index)

    if scale === nothing
        scale = fill(T(1), len; location)
    else
        @assert size(scale) == (len,)
    end

    if offset === nothing
        offset = fill(T(0), len; location)
    else
        @assert size(offset) == (len,)
    end

    batch_norm_train_op = stablehlo.batch_norm_training(
        operand.mlir_data,
        scale.mlir_data,
        offset.mlir_data;
        epsilon=Float32(epsilon),
        feature_index=feature_index - 1,
        location,
    )

    return (
        TracedRArray{T,N}((), MLIR.IR.result(batch_norm_train_op, 1), size(operand)),
        TracedRArray{T,1}((), MLIR.IR.result(batch_norm_train_op, 2), (len,)),
        TracedRArray{T,1}((), MLIR.IR.result(batch_norm_train_op, 3), (len,)),
    )
end

@noinline function batch_norm_grad(
    operand::TracedRArray{T,N},
    scale::Union{TracedRArray{T,1},Nothing},
    mean::TracedRArray{T,1},
    variance::TracedRArray{T,1},
    grad_output::TracedRArray{T,N};
    epsilon,
    feature_index::Int64,
    location=mlir_stacktrace("batch_norm_grad", @__FILE__, @__LINE__),
) where {T,N}
    len = size(operand, feature_index)
    @assert length(mean) == length(variance) == len
    @assert size(grad_output) == size(operand)

    has_affine = scale !== nothing

    if !has_affine
        scale = fill(T(1), len; location)
    else
        @assert size(scale) == (len,)
    end

    batch_norm_grad_op = stablehlo.batch_norm_grad(
        operand.mlir_data,
        scale.mlir_data,
        mean.mlir_data,
        variance.mlir_data,
        grad_output.mlir_data;
        epsilon=Float32(epsilon),
        feature_index=feature_index - 1,
        location,
    )

    grad_operand = TracedRArray{T,N}(
        (), MLIR.IR.result(batch_norm_grad_op, 1), size(operand)
    )
    grad_scale = TracedRArray{T,1}((), MLIR.IR.result(batch_norm_grad_op, 2), (len,))
    grad_offset = TracedRArray{T,1}((), MLIR.IR.result(batch_norm_grad_op, 3), (len,))

    return (
        grad_operand, has_affine ? grad_scale : nothing, has_affine ? grad_offset : nothing
    )
end

@noinline function ignore_derivatives(
    input::Union{TracedRArray,TracedRNumber};
    location=mlir_stacktrace("ignore_derivatives", @__FILE__, @__LINE__),
)
    res = MLIR.IR.result(
        enzyme.ignore_derivatives(input.mlir_data; output=mlir_type(input), location), 1
    )

    if input isa TracedRArray
        return TracedRArray{unwrapped_eltype(input),ndims(input)}((), res, size(res))
    else
        return TracedRNumber{unwrapped_eltype(input)}((), res)
    end
end

@noinline function wrap(
    input::TracedRArray{T,N},
    lhs::Integer,
    rhs::Integer;
    dimension::Int,
    location=mlir_stacktrace("wrap", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 ≤ dimension ≤ N "dimension must be between 1 and $(N) (got $(dimension))"
    @assert 0 ≤ lhs ≤ size(input, dimension) "lhs must be between 0 and \
                                              $(size(input, dimension)) (got $(lhs))"
    @assert 0 ≤ rhs ≤ size(input, dimension) "rhs must be between 0 and \
                                              $(size(input, dimension)) (got $(rhs))"

    sz = collect(Int64, size(input))
    sz[dimension] = sz[dimension] + lhs + rhs

    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            enzymexla.wrap(input.mlir_data; lhs, rhs, dimension=dimension - 1, location), 1
        ),
        sz,
    )
end

@noinline function extend(
    input::TracedRArray{T,N},
    lhs::Integer,
    rhs::Integer;
    dimension::Int,
    location=mlir_stacktrace("extend", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 ≤ dimension ≤ N "dimension must be between 1 and $(N) (got $(dimension))"
    @assert 0 ≤ lhs ≤ size(input, dimension) "lhs must be between 0 and \
                                              $(size(input, dimension)) (got $(lhs))"
    @assert 0 ≤ rhs ≤ size(input, dimension) "rhs must be between 0 and \
                                              $(size(input, dimension)) (got $(rhs))"
    sz = collect(Int64, size(input))
    sz[dimension] = sz[dimension] + lhs + rhs
    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            enzymexla.extend(input.mlir_data; lhs, rhs, dimension=dimension - 1, location),
            1,
        ),
        sz,
    )
end

@noinline function rotate(
    input::TracedRArray{T,N},
    amount::Integer;
    dimension::Int,
    location=mlir_stacktrace("rotate", @__FILE__, @__LINE__),
) where {T,N}
    @assert 1 ≤ dimension ≤ N "dimension must be between 1 and $(N) (got $(dimension))"
    @assert 0 ≤ amount ≤ size(input, dimension) "amount must be between 0 and \
                                                 $(size(input, dimension)) (got $(amount))"
    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            enzymexla.rotate(
                input.mlir_data;
                amount=Int32(amount),
                dimension=Int32(dimension - 1),
                location,
            ),
            1,
        ),
        size(input),
    )
end

@noinline function sharding_group(
    inputs::Union{TracedRArray,TracedRNumber}...;
    group_id::Union{Integer,Nothing}=nothing,
    location=mlir_stacktrace("sharding_group", @__FILE__, @__LINE__),
)
    @assert length(inputs) > 1 "At least two inputs are required to form a sharding group, \
                                got $(length(inputs))"

    counter, cache = Reactant.Compiler.sdygroupidcache()

    group_ids = unique([cache[input] for input in inputs if haskey(cache, input)])
    if length(group_ids) > 1
        error("All inputs must belong to the same sharding group. Found multiple group \
               ids: $(group_ids)")
    end

    if length(group_ids) == 0
        if group_id === nothing
            group_id = @atomic counter.group_id
            @atomic counter.group_id += 1
        end
    else
        found_group_id = only(group_ids)
        if group_id !== nothing && found_group_id != group_id
            error("Provided group_id $(group_id) does not match the existing group_id \
                   $(found_group_id) for the inputs. All inputs must belong to the same \
                   sharding group.")
        end
        group_id = found_group_id
    end

    for input in inputs
        if !haskey(cache, input)
            cache[input] = group_id
            MLIR.Dialects.sdy.sharding_group(input.mlir_data; group_id=group_id, location)
        end
    end

    return nothing
end

end # module Ops
