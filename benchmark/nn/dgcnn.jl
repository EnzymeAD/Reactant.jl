using Lux, Reactant, NNlib
using Reactant: AnyTracedRArray, @opcall
using ConcreteStructs: @concrete

include("common.jl")

function create_single_knn_graph(x::AnyTracedRArray{T,3}, k::Int) where {T}
    x_permuted = permutedims(x, (3, 1, 2))
    res = @opcall batch(
        Base.Fix2(create_single_knn_graph, k), [x_permuted], [size(x_permuted, 1)]
    )
    return permutedims(res[1], (2, 3, 4, 1))
end

function create_single_knn_graph(x::AbstractMatrix, k::Int)
    sqx = sum(abs2, x; dims=1)
    inner = 2 .* (x' * x)
    pairwise_distances = sqx .- inner .+ sqx'
    return NNlib.gather(
        Lux.Utils.contiguous(x),
        mapslices(Base.Fix2(partialsortperm, 1:k), pairwise_distances; dims=1),
    )
end

@concrete struct EdgeConv <: AbstractLuxContainerLayer{(:model, :skip)}
    model
    skip
    K::Int
    residual_connection::Bool
end

function EdgeConv(layers::AbstractArray{Int}, K::Int; residual_connection::Bool=true)
    mlp_layers = map(1:(length(layers) - 1)) do i
        in_dims = i == 1 ? 2 * layers[i] : layers[i]
        out_dims = layers[i + 1]
        return Chain(Dense(in_dims => out_dims), BatchNorm(out_dims, relu))
    end

    skip = residual_connection ? Dense(layers[1] => layers[end], relu) : NoOpLayer()

    return EdgeConv(Chain(mlp_layers...), skip, K, residual_connection)
end

function (edge_conv::EdgeConv)(x::AbstractArray{T,3}, ps, st) where {T}
    F, N, B = size(x)

    knn_graph = create_single_knn_graph(x, edge_conv.K) # F, K, N, B

    x′ = reshape(x, F, 1, N, B)
    x_repeated = repeat(x′, 1, edge_conv.K, 1, 1)

    input = vcat(x_repeated, knn_graph .- x′) # 2F, K, N, B
    input = reshape(input, 2F, :)             # 2F, K * N * B

    output, stₙ_model = edge_conv.model(input, ps.model, st.model)
    result = reshape(output, :, edge_conv.K, N, B) # O, K, N, B

    pooled_result = dropdims(maximum(result; dims=2); dims=2) # O, N, B

    if edge_conv.residual_connection
        skip_connection, stₙ_skip = edge_conv.skip(x, ps.skip, st.skip)
        result = pooled_result .+ skip_connection
        return result, merge(st, (; skip=stₙ_skip, model=stₙ_model))
    end

    return pooled_result, merge(st, (; model=stₙ_model))
end

@concrete struct DGCNN <: AbstractLuxWrapperLayer{:model}
    model
end

function DGCNN(
    edge_conv_layers,
    fc_layers,
    K::Int,
    embed_dim::Int;
    dropout_rate::AbstractFloat=0.0,
    residual_connection::Bool=true,
)
    edge_conv = Chain(map(l -> EdgeConv(l, K; residual_connection), edge_conv_layers)...)
    conv_block = Chain(
        Conv((1,), last(last(edge_conv_layers)) => first(fc_layers)),
        BatchNorm(first(fc_layers), relu),
    )
    embed_block = Chain(
        [
            Chain(Dense(fc_layers[i] => fc_layers[i + 1], relu), Dropout(dropout_rate)) for
            i in 1:(length(fc_layers) - 1)
        ]...,
        Dense(fc_layers[end] => embed_dim),
    )
    return DGCNN(
        Chain(;
            edge_conv,
            pdims=WrappedFunction(Base.Fix2(permutedims, (2, 1, 3))),
            conv_block,
            pool=GlobalMaxPool(),
            flatten=FlattenLayer(),
            embed_block,
        ),
    )
end

function run_dgcnn_benchmark!(results, backend)
    model = DGCNN([[3, 32, 64], [64, 128]], [512, 128, 64], 4, 32)

    N, S, B = 3, 128, 256

    run_lux_benchmark!(
        results,
        "DGCNN [$(N), $(S), $(B)]",
        backend,
        model,
        (N, S, B);
        disable_transpose_bench=false,
        disable_bwd_transpose_bench=false,
    )

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_dgcnn_benchmark!(results, backend)
end
