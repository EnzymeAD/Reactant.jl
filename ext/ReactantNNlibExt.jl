module ReactantNNlibExt

using NNlib
using Reactant

for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval function $(jlop)(x::Reactant.TracedRArray{T,0}) where {T}
        return Reactant.TracedRArray{T,0}(
            (),
            Reactant.MLIR.IR.result(
                Reactant.MLIR.Dialects.stablehlo.$(hloop)(x.mlir_data), 1
            ),
        )
    end
end

NNlib.relu(x::Reactant.TracedRArray{T,(),0}) where {T} = max(x, zero(T))

NNlib.gelu(x::Reactant.TracedRArray{T,(),0}) where {T} = x * sigmoid(T(1.702) * x)

# TODO handle non finite cases
function NNlib.softmax!(
    out::Reactant.TracedRArray{T,N}, x::AbstractArray; dims=1
) where {T,N}
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
