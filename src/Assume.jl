"""
    Reactant.assume(
        x;
        no_nan::Union{Bool,Nothing}=nothing,
        finite::Union{Bool,Nothing}=nothing,
        non_negative::Union{Bool,Nothing}=nothing,
        symmetric::Union{Bool,Nothing}=nothing,
        purely_real::Union{Bool,Nothing}=nothing,
        purely_imaginary::Union{Bool,Nothing}=nothing,
    )

Provide compiler optimization hints about properties guaranteed to hold for `x`, returning
a value with the same contents and those assumptions attached during Reactant compilation.
Outside of Reactant compilation, validates the requested properties and returns `x`
unchanged.

Supported keyword arguments:
- `no_nan`: whether elements of `x` are guaranteed not to be `NaN` (`enzymexla.no_nan`).
- `finite`: whether elements of `x` are guaranteed to be finite (`enzymexla.finite`;
  `finite=true` also implies `no_nan=true`).
- `non_negative`: whether elements of `x` are guaranteed to be `>= 0`
  (`enzymexla.non_negative`).
- `symmetric`: whether a square 2D matrix `x` is symmetric (`x == transpose(x)`,
  `enzymexla.symmetric_matrix`).
- `purely_real`: whether complex elements of `x` have zero imaginary part
  (`enzymexla.complex_is_purely_real`).
- `purely_imaginary`: whether complex elements of `x` have zero real part
  (`enzymexla.complex_is_purely_imaginary`).
"""
function assume(
    x::TracedRNumber;
    no_nan::Union{Bool,Nothing}=nothing,
    finite::Union{Bool,Nothing}=nothing,
    non_negative::Union{Bool,Nothing}=nothing,
    symmetric::Union{Bool,Nothing}=nothing,
    purely_real::Union{Bool,Nothing}=nothing,
    purely_imaginary::Union{Bool,Nothing}=nothing,
)
    return Ops.assume(
        x; no_nan, finite, non_negative, symmetric, purely_real, purely_imaginary
    )
end

function assume(
    x::TracedRArray;
    no_nan::Union{Bool,Nothing}=nothing,
    finite::Union{Bool,Nothing}=nothing,
    non_negative::Union{Bool,Nothing}=nothing,
    symmetric::Union{Bool,Nothing}=nothing,
    purely_real::Union{Bool,Nothing}=nothing,
    purely_imaginary::Union{Bool,Nothing}=nothing,
)
    return Ops.assume(
        x; no_nan, finite, non_negative, symmetric, purely_real, purely_imaginary
    )
end

function assume(
    x::AnyTracedRArray;
    no_nan::Union{Bool,Nothing}=nothing,
    finite::Union{Bool,Nothing}=nothing,
    non_negative::Union{Bool,Nothing}=nothing,
    symmetric::Union{Bool,Nothing}=nothing,
    purely_real::Union{Bool,Nothing}=nothing,
    purely_imaginary::Union{Bool,Nothing}=nothing,
)
    return Ops.assume(
        materialize_traced_array(x);
        no_nan,
        finite,
        non_negative,
        symmetric,
        purely_real,
        purely_imaginary,
    )
end

function assume(
    x::Union{Number,AbstractArray};
    no_nan::Union{Bool,Nothing}=nothing,
    finite::Union{Bool,Nothing}=nothing,
    non_negative::Union{Bool,Nothing}=nothing,
    symmetric::Union{Bool,Nothing}=nothing,
    purely_real::Union{Bool,Nothing}=nothing,
    purely_imaginary::Union{Bool,Nothing}=nothing,
)
    Ops._validate_and_collect_assume_attrs(
        ndims(x),
        size(x),
        no_nan,
        finite,
        non_negative,
        symmetric,
        purely_real,
        purely_imaginary,
    )
    return x
end

const _ENZYMEXLA_ANALYSIS_ATTR_NAMES = (
    "enzymexla.symmetric_matrix",
    "enzymexla.non_negative",
    "enzymexla.finite",
    "enzymexla.bounds",
    "enzymexla.no_nan",
    "enzymexla.complex_is_purely_real",
    "enzymexla.complex_is_purely_imaginary",
)

function _strip_enzymexla_func_arg_attrs!(mod::MLIR.IR.Module)
    ctx = MLIR.IR.context(mod)
    for op in MLIR.IR.body(mod)
        arg_attrs = MLIR.IR.getattr(op, "arg_attrs")
        (arg_attrs === nothing || !MLIR.IR.isarray(arg_attrs)) && continue
        nargs = MLIR.API.mlirArrayAttrGetNumElements(arg_attrs)
        changed = false
        all_empty = true
        new_dicts = Vector{MLIR.API.MlirAttribute}(undef, nargs)
        for i in 1:nargs
            raw_dict = MLIR.API.mlirArrayAttrGetElement(arg_attrs, i - 1)
            dict = MLIR.IR.Attribute(raw_dict)
            if !MLIR.IR.isdict(dict)
                new_dicts[i] = raw_dict
                all_empty = false
                continue
            end
            nd = MLIR.API.mlirDictionaryAttrGetNumElements(dict)
            kept = MLIR.API.MlirNamedAttribute[]
            for j in 1:nd
                na = MLIR.API.mlirDictionaryAttrGetElement(dict, j - 1)
                attr_name = String(MLIR.API.mlirIdentifierStr(na.name))
                if attr_name in _ENZYMEXLA_ANALYSIS_ATTR_NAMES
                    changed = true
                else
                    push!(kept, na)
                end
            end
            if !isempty(kept)
                all_empty = false
            end
            new_dicts[i] = MLIR.API.mlirDictionaryAttrGet(ctx, length(kept), kept)
        end
        if changed
            if all_empty
                MLIR.IR.rmattr!(op, "arg_attrs")
            else
                MLIR.IR.setattr!(
                    op,
                    "arg_attrs",
                    MLIR.IR.Attribute(MLIR.API.mlirArrayAttrGet(ctx, nargs, new_dicts)),
                )
            end
        end
    end
    return nothing
end

@static if VERSION ≥ v"1.11"
    @eval $(Expr(:public, :assume))
end
