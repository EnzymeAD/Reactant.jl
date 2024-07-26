module ReactantNNlibExt

using NNlib
using Reactant

for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval begin
        function Reactant.elem_apply(
            ::typeof($jlop), lhs::Reactant.TracedRArray{ElType,Shape,N}
        ) where {ElType,Shape,N}
            return Reactant.TracedRArray{ElType,Shape,N}(
                (),
                Reactant.MLIR.IR.result(
                    Reactant.MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1
                ),
            )
        end
    end
end

function Reactant.elem_apply(
    ::typeof(NNlib.relu), lhs::Reactant.TracedRArray{ElType,Shape,N}
) where {ElType,Shape,N}
    return ifelse.((lhs .> zero(ElType)), lhs, zero(ElType))
end

function Reactant.elem_apply(
    ::typeof(NNlib.gelu), lhs::Reactant.TracedRArray{ElType,Shape,N}
) where {ElType,Shape,N}
    # See https://arxiv.org/pdf/1606.08415v5 Section 2
    return lhs .* sigmoid.(ElType(1.702) .* lhs)
end

# TODO handle non finite cases
function NNlib.softmax!(
    out::Reactant.TracedRArray{T,Shape,N}, x::AbstractArray; dims=1
) where {T,Shape,N}
    max_ = NNlib.fast_maximum(x; dims)
    #if all(isfinite, max_)
    @fastmath out .= exp.(x .- max_)
    #else
    #    _zero, _one, _inf = T(0), T(1), T(Inf)
    #    @fastmath @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _one, _zero), exp(x - max_))
    #end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    return out ./= tmp
end

end
