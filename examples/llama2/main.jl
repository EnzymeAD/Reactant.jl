# Code is adapted from the Pytorch Version https://github.com/hkproj/pytorch-llama/blob/main/model.py
using Lux, Random, Reactant, Enzyme, Statistics
using PythonCall, CondaPkg # For the tokenizer
using Downloads: Downloads
using BenchmarkTools, InteractiveUtils

@info sprint(versioninfo)

sentencepiece = pyimport("sentencepiece")

# TODO: Test CUDA support once yggdrasil build is happy

@kwdef mutable struct ModelArgs
    dim::Int = 4096
    n_layers::Int = 32
    n_heads::Int = 32
    n_kv_heads::Union{Nothing,Int} = nothing
    vocab_size::Int = -1 # Later set in the build method
    multiple_of::Int = 256
    ffn_dim_multiplier::Union{Nothing,Float64} = nothing
    norm_eps::Float64 = 1e-5

    # Needed for KV cache
    max_batch_size::Int = 32
    max_seq_len::Int = 2048
end

function RMSNorm(dim::Int, ϵ=1.0f-6)
    return @compact(; weight=ones32(dim), ϵ) do x::AbstractArray{<:Real,3}
        xmean = mean(abs2, x; dims=1)
        tmp = @. sqrt(xmean + ϵ)
        @return @. weight * x / tmp
    end
end

function precompute_theta_pos_frequencies(
    head_dim::Int, seq_len::Int, theta::Float32=10000.0f0
)
    @assert head_dim % 2 == 0 "Dimension must be divisible by 2"
    theta_numerator = Float32.(0:2:(head_dim - 1))          # (head_dim / 2)
    theta = @. 1.0f0 / (theta^(theta_numerator / head_dim)) # (Dim / 2)
    m = Float32.(0:(seq_len - 1))                           # (Seq_Len)
    freqs = theta * m'                                      # (head_dim / 2, Seq_Len)
    cf = reshape(cos.(freqs), size(freqs)..., 1)
    sf = reshape(sin.(freqs), size(freqs)..., 1)
    return cat(cf, sf; dims=Val(3))                         # (head_dim / 2, Seq_Len, 2)
end

function apply_rotary_embeddings(
    x::AbstractArray{T1,4}, freqs_complex::AbstractArray{T2,3}
) where {T1,T2}
    head_dim, H, seq_len, B = size(x)
    x_complex = reshape(x, 2, :, H, seq_len, B)                 # (2, head_dim / 2, H, Seq_Len, B)
    freqs_complex = reshape(freqs_complex, 2, :, 1, seq_len, 1) # (2, head_dim / 2, 1, Seq_Len, 1)
    x_rotated = x_complex .* freqs_complex                      # (2, head_dim / 2, H, Seq_Len, B)
    return reshape(x_rotated, head_dim, H, seq_len, B)          # (head_dim, H, Seq_Len, B)
end

function repeat_kv(x::AbstractArray{T,4}, n_rep::Int) where {T}
    n_rep == 1 && return x
    return repeat(x, 1, 1, n_rep, 1)
end

@views function SelfAttention(args::ModelArgs)
    n_kv_heads = args.n_kv_heads === nothing ? args.n_heads : args.n_kv_heads
    n_heads_q = args.n_heads
    n_rep = n_heads_q ÷ n_kv_heads
    head_dim = args.dim ÷ args.n_heads

    wq = Dense(args.dim => args.n_heads * head_dim; use_bias=false)
    wk = Dense(args.dim => n_kv_heads * head_dim; use_bias=false)
    wv = Dense(args.dim => n_kv_heads * head_dim; use_bias=false)
    wo = Dense(args.n_heads * head_dim => args.dim; use_bias=false)

    cache_k = zeros32(head_dim, n_kv_heads, args.max_seq_len, args.max_batch_size)
    cache_v = zeros32(head_dim, n_kv_heads, args.max_seq_len, args.max_batch_size)

    return @compact(;
        wq,
        wk,
        wv,
        wo,
        n_kv_heads,
        n_heads_q,
        n_rep,
        head_dim,
        cache_k=@non_trainable(cache_k),
        cache_v=@non_trainable(cache_v)
    ) do (x, start_pos, freqs_complex)
        _, seq_len, B = size(x)

        xq = reshape(wq(x), head_dim, n_heads_q, seq_len, B)     # (head_dim, H_Q, SL, B)
        xk = reshape(wk(x), head_dim, n_kv_heads, seq_len, B)    # (head_dim, H_KV, SL, B)
        xv = reshape(wv(x), head_dim, n_kv_heads, seq_len, B)    # (head_dim, H_KV, SL, B)

        xq = apply_rotary_embeddings(xq, freqs_complex)          # (head_dim, H_Q, SL, B)
        xk = apply_rotary_embeddings(xk, freqs_complex)          # (head_dim, H_KV, SL, B)

        # Replace the entry in the cache
        cache_k[:, :, start_pos:(start_pos + seq_len - 1), 1:B] .= xk
        cache_v[:, :, start_pos:(start_pos + seq_len - 1), 1:B] .= xv

        keys = cache_k[:, :, 1:(start_pos + seq_len - 1), 1:B]   # (head_dim, H_KV, SL_KV, B)
        values = cache_v[:, :, 1:(start_pos + seq_len - 1), 1:B] # (head_dim, H_KV, SL_KV, B)

        keys = repeat_kv(keys, n_rep)                            # (head_dim, H_Q, SL_KV, B)
        values = repeat_kv(values, n_rep)                        # (head_dim, H_Q, SL_KV, B)

        # TODO: Lazy Permutedims?
        xq2 = permutedims(xq, (1, 3, 2, 4))                      # (head_dim, SL, H_Q, B)
        keys2 = permutedims(keys, (1, 3, 2, 4))                  # (head_dim, SL_KV, H_Q, B)
        values2 = permutedims(values, (1, 3, 2, 4))              # (head_dim, SL_KV, H_Q, B)

        xq2_flat = reshape(xq2, size(xq2, 1), size(xq2, 2), :)           # (head_dim, SL, H_Q * B)
        keys2_flat = reshape(keys2, size(keys2, 1), size(keys2, 2), :)   # (head_dim, SL_KV, H_Q * B)
        scores = batched_matmul(batched_transpose(keys2_flat), xq2_flat) # (SL_KV, SL, H_Q * B)
        scores = softmax(scores ./ sqrt(head_dim); dims=1)               # (SL_KV, SL, H_Q * B)

        values2_flat = reshape(values2, size(values2, 1), size(values2, 2), :) # (head_dim, SL_KV, H_Q * B)
        output = batched_matmul(values2_flat, scores)             # (head_dim, SL, H_Q * B)
        output = reshape(output, head_dim, seq_len, n_heads_q, B) # (head_dim, SL, H_Q, B)
        output = permutedims(output, (1, 3, 2, 4))                # (head_dim, H_Q, SL, B)
        @return wo(reshape(output, :, seq_len, B))                # (head_dim, H_Q, B)
    end
end

function FeedForward(args::ModelArgs)
    hidden_dim = 2 * ((4 * args.dim) ÷ 3)
    if args.ffn_dim_multiplier !== nothing
        hidden_dim *= args.ffn_dim_multiplier
    end
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) ÷ args.multiple_of)

    w1 = Dense(args.dim => hidden_dim, swish; use_bias=false)
    w2 = Dense(hidden_dim => args.dim; use_bias=false)
    w3 = Dense(args.dim => hidden_dim; use_bias=false)

    return Chain(Parallel(.*, w1, w3), w2)
end

function EncoderBlock(args::ModelArgs)
    attention = SelfAttention(args)
    feed_forward = FeedForward(args)

    attention_norm = RMSNorm(args.dim, args.norm_eps)
    ffn_norm = RMSNorm(args.dim, args.norm_eps)

    return @compact(;
        attention, feed_forward, attention_norm, ffn_norm
    ) do (x, start_pos, freqs_complex)
        h = x .+ attention((attention_norm(x), start_pos, freqs_complex))
        @return h .+ feed_forward(ffn_norm(h))
    end
end

function Transformer(args::ModelArgs)
    @assert args.vocab_size != -1 "Vocab size must be set"

    token_embeddings = Embedding(args.vocab_size => args.dim)
    layers = [EncoderBlock(args) for _ in 1:(args.n_layers)]
    norm = RMSNorm(args.dim, args.norm_eps)
    output = Dense(args.dim, args.vocab_size; use_bias=false)

    freqs_complex = precompute_theta_pos_frequencies(
        args.dim ÷ args.n_heads, args.max_seq_len * 2
    )

    return @compact(;
        token_embeddings, layers, norm, output, freqs_complex=@non_trainable(freqs_complex)
    ) do (tokens, start_pos)
        seq_len, _ = size(tokens)

        @assert seq_len == 1 "Only one token at a time can be processed"

        h = token_embeddings(tokens)
        freqs_complex_part = @view freqs_complex[:, start_pos:(start_pos + seq_len - 1), :]
        for layer in layers
            h = layer((h, start_pos, freqs_complex_part))
        end
        h = norm(h)
        @return output(h)
    end
end

# Main Model
struct Llama2{M,T,A}
    model::M
    tokenizer::T
    args::A
end

function Llama2(;
    tokenizer_path=joinpath(@__DIR__, "tokenizer.model"),
    max_seq_len::Int,
    max_batch_size::Int,
)
    model_args = ModelArgs(; max_seq_len, max_batch_size)

    if !isfile(tokenizer_path)
        @info "Downloading `tokenizer.model` to $(tokenizer_path)"
        Downloads.download(
            "https://github.com/juvi21/llama2.jl/raw/master/tokenizer.model", tokenizer_path
        )
    end
    tokenizer = sentencepiece.SentencePieceProcessor(;
        model_file=joinpath(@__DIR__, "tokenizer.model")
    )

    model_args.vocab_size = pyconvert(Int, tokenizer.get_piece_size())

    model = Transformer(model_args)
    ps, st = Lux.setup(Random.default_rng(), model)

    # TODO: Load a pretrained model

    return Llama2(StatefulLuxLayer{true}(model, ps, st), tokenizer, model_args)
end

function text_completion(
    llama2::Llama2,
    prompts::Vector{<:AbstractString},
    temperature=0.6f0,
    top_p=0.9f0,
    max_gen_len=nothing,
)
    max_gen_len = max_gen_len === nothing ? llama2.args.max_seq_len : max_gen_len

    prompt_tokens = [
        pyconvert(Vector, llama2.tokenizer.encode(prompt; add_bos=true, add_eos=true)) for
        prompt in prompts
    ]
    batch_size = length(prompt_tokens)
    @assert batch_size < args.max_batch_size "Batch size exceeds the maximum batch size"

    max_prompt_len = maximum(length, prompt_tokens)
    @assert max_prompt_len < args.max_seq_len "Prompt length exceeds the maximum sequence length"

    total_len = min(args.max_seq_len, max_prompt_len + max_gen_len)

    pad_id = pyconvert(Int64, llama2.tokenizer.pad_id())
    tokens = fill(pad_id, total_len, batch_size)
    for (k, t) in enumerate(prompt_tokens)
        tokens[1:length(t), k] .= t
    end

    eos_reached = fill(false, batch_size)
    prompt_tokens_mask = tokens .!= pad_id

    # FIXME: something is getting promoted to Float64??

    for cur_pos in 2:total_len
        logits = llama2.model((view(tokens, cur_pos:cur_pos, :), cur_pos))

        # TODO: use temperature
        # Greedy selection
        next_token = vec(argmax(logits; dims=1))

        # Replace padding tokens
        tokens[cur_pos, :] .=
            ifelse.(prompt_tokens_mask[cur_pos, :], tokens[cur_pos, :], next_token)

        eos_reached .|=
            .!prompt_tokens_mask[cur_pos, :] .& (next_token == llama2.tokenizer.eos_id)
        all(eos_reached) && break
    end

    # TODO: Process output

    return nothing
end

prompts = [
    "Simply put, the theory of relativity states that ",
    "If Google was an Italian company founded in Milan, it would",

    # Few shot promt
    """Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>""",

    # Zero shot prompt
    """Tell me if the following person is actually Doraemon disguised as human:
    Name: Umar Jamil
    Decision: 
    """,
]

llama2 = Llama2(; max_seq_len=1024, max_batch_size=length(prompts));
nothing

# First we benchmark the forward pass of vanilla Lux and then Reactant compiled Lux version

# Now we benchmark the following reverse passes
# 1. Lux + Enzyme
# 2. Reactant compiled Lux + Enzyme
