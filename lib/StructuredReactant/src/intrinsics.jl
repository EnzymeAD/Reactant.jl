# Intrinsics reach the emitter on traced operands only through values that became
# traced after inference (loop carries, traced branches); map them to `Base`.

function emit_intrinsic(fr::Frame, f::Core.IntrinsicFunction, args::Vector{Any})
    # A traced enum, typed as a host one by inference, is its base integer.
    for (i, a) in enumerate(args)
        a isa TracedEnum && (args[i] = getfield(a, :value))
    end
    f === Core.Intrinsics.bitcast && args[2] isa TracedRNumber{args[1]} && return args[2]
    any(a -> a isa TracedRNumber, args) || return f(args...)
    op = get(TRACED_INTRINSICS, f, nothing)
    op === nothing && unsupported(fr, "the `$(f)` intrinsic on traced operands")
    return op(args...)
end

traced(op) = (args...) -> Reactant.call_with_reactant(op, args...)
convert_traced(::Type{T}, x) where {T} = Ops.convert(TracedRNumber{T}, x)
bitwise_not(x) = Reactant.unwrapped_eltype(x) === Bool ? traced(!)(x) : traced(~)(x)

function unsigned_only(op)
    return function (args...)
        tuple_all(x -> Reactant.unwrapped_eltype(x) <: Unsigned, args) || throw(
            FrontendError("unsigned `$(op)` on signed traced integers is not supported")
        )
        return Reactant.call_with_reactant(op, args...)
    end
end

const TRACED_INTRINSICS = Dict{Core.IntrinsicFunction,Any}()

let I = Core.Intrinsics
    for (name, op) in (
        :add_int => traced(+),
        :sub_int => traced(-),
        :mul_int => traced(*),
        :neg_int => traced(-),
        :sdiv_int => traced(div),
        :srem_int => traced(rem),
        :udiv_int => unsigned_only(div),
        :urem_int => unsigned_only(rem),
        :add_float => traced(+),
        :sub_float => traced(-),
        :mul_float => traced(*),
        :div_float => traced(/),
        :neg_float => traced(-),
        :add_float_fast => traced(+),
        :sub_float_fast => traced(-),
        :mul_float_fast => traced(*),
        :div_float_fast => traced(/),
        :neg_float_fast => traced(-),
        :fma_float => traced(fma),
        :muladd_float => traced(muladd),
        :eq_int => traced(==),
        :ne_int => traced(!=),
        :slt_int => traced(<),
        :sle_int => traced(<=),
        :ult_int => unsigned_only(<),
        :ule_int => unsigned_only(<=),
        :eq_float => traced(==),
        :ne_float => traced(!=),
        :lt_float => traced(<),
        :le_float => traced(<=),
        :eq_float_fast => traced(==),
        :ne_float_fast => traced(!=),
        :lt_float_fast => traced(<),
        :le_float_fast => traced(<=),
        :and_int => traced(&),
        :or_int => traced(|),
        :xor_int => traced(xor),
        :not_int => bitwise_not,
        :shl_int => traced(<<),
        :lshr_int => traced(>>>),
        :ashr_int => traced(>>),
        :abs_float => traced(abs),
        :copysign_float => traced(copysign),
        :flipsign_int => traced(flipsign),
        :sqrt_llvm => traced(sqrt),
        :sqrt_llvm_fast => traced(sqrt),
        :floor_llvm => traced(floor),
        :ceil_llvm => traced(ceil),
        :trunc_llvm => traced(trunc),
        :rint_llvm => traced(round),
        :sitofp => convert_traced,
        :uitofp => convert_traced,
        :fptosi => convert_traced,
        :fptoui => convert_traced,
        :fpext => convert_traced,
        :fptrunc => convert_traced,
        :sext_int => convert_traced,
        :zext_int => convert_traced,
        :trunc_int => convert_traced,
    )
        isdefined(I, name) || continue
        TRACED_INTRINSICS[getfield(I, name)] = op
    end
end
