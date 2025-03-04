@inline Base.vcat(a::Number, b::Union{AnyConcretePJRTArray,AnyTracedRArray}) =
    @allowscalar(vcat(fill!(similar(b, typeof(a), (1, size(b)[2:end]...)), a), b))
@inline Base.hcat(a::Number, b::Union{AnyConcretePJRTArray,AnyTracedRArray}) =
    @allowscalar(hcat(fill!(similar(b, typeof(a), (size(b)[1:(end - 1)]..., 1)), a), b))
