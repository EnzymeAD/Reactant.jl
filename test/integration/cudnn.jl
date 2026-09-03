using Reactant, Test
using Reactant: MLIR
using BFloat16s: BFloat16

# cuDNN's runtime-compiled engines (fused attention / SDPA, runtime fusion) JIT their
# kernels with NVRTC when an execution plan is built. Through the NVRTC that is linked
# statically into Reactant_jll that compile fails
# (`CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED ... compilationResult != NVRTC_SUCCESS`,
# surfaced by XLA as "[cudnn_frontend] Error: No valid execution plans built."), while the
# same kernels compile through a shared libnvrtc of the same major version. ReactantCUDAExt
# therefore points cuDNN at CUDA_Runtime_jll's libnvrtc via `CUDNN_NVRTC_OVERRIDE_PATH`.
# This exercises the whole chain with XLA:GPU's own fused-attention custom call (the op
# JAX emits for `jax.nn.dot_product_attention(implementation="cudnn")`).

const stablehlo = MLIR.Dialects.stablehlo

function fmha_backend_config(B, H, T, S, scale)
    dims = join(repr.(string.([B, H, T, S])), ",")
    function dotdims(lc, rc)
        return """{"lhs_contracting_dimensions": ["$lc"], "rhs_contracting_dimensions": ["$rc"],
 "lhs_batch_dimensions": ["0","1"], "rhs_batch_dimensions": ["0","1"]}"""
    end
    return """{"operation_queue_id": "0", "wait_on_operation_queues": [],
     "cudnn_fmha_backend_config": {
      "algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"},
                    "is_cudnn_frontend": true, "workspace_size": "0"},
      "fmha_scale": $(scale), "dropout_rate": 0.0,
      "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": [$(dims)],
         "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [],
         "minor_to_major": ["3","2","1","0"], "tiles": [], "element_size_in_bits": "0",
         "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
         "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"},
         "is_dynamic_dimension": [false, false, false, false]},
      "seed": "42", "is_flash_attention": true, "mask_type": "CAUSAL",
      "bmm1_dot_dimension_numbers": $(dotdims(3, 3)),
      "bmm2_dot_dimension_numbers": $(dotdims(3, 2))
     }}"""
end

# q, k, v: traced (D, H, T, B) bf16 (feature-first Julia layout). A Julia array shows up
# in the MLIR tensor type with its dims reversed, so `permutedims(x, (4, 2, 3, 1))` hands
# the custom call JAX's BNTH layout (`tensor<BxHxTxD>`, embedding dim minor) that cuDNN's
# fused attention expects. Returns (D, H, T, B).
function cudnn_fmha_causal(q, k, v; scale)
    mat = Reactant.TracedUtils.materialize_traced_array
    D, H, T, B = size(q)
    S = size(k, 3)
    tojax(x) = mat(permutedims(x, (4, 2, 3, 1)))
    out = stablehlo.custom_call(
        [tojax(q).mlir_data, tojax(k).mlir_data, tojax(v).mlir_data];
        result_0=[
            Reactant.Ops.mlir_type(Reactant.TracedRArray{BFloat16,4}, [B, H, T, D]),
            Reactant.Ops.mlir_type(Reactant.TracedRArray{Float32,3}, [B, H, T]),
        ],
        call_target_name="__cudnn\$fmhaSoftmax",
        backend_config=MLIR.IR.Attribute(fmha_backend_config(B, H, T, S, scale)),
        api_version=Int32(1),
    )
    y = Reactant.TracedRArray{BFloat16,4}((), MLIR.IR.result(out, 1), (B, H, T, D))
    return permutedims(y, (4, 2, 3, 1))
end

function reference_attention_causal(q, k, v; scale)
    D, H, T, B = size(q)
    o = zeros(Float32, D, H, T, B)
    for b in 1:B, h in 1:H
        Q = Float32.(q[:, h, :, b])'   # (T, D)
        K = Float32.(k[:, h, :, b])'
        V = Float32.(v[:, h, :, b])'
        s = (Q * K') .* Float32(scale)
        for i in 1:T, j in (i + 1):T
            s[i, j] = -Inf32
        end
        s .-= maximum(s; dims=2)
        p = exp.(s)
        p ./= sum(p; dims=2)
        o[:, h, :, b] = (p * V)'
    end
    return o
end

@testset "cuDNN runtime-compiled fused attention" begin
    if Reactant.XLA.platform_name(Reactant.XLA.default_backend()) != "cuda"
        @info "cuDNN fused attention needs the CUDA backend, skipping"
    elseif Reactant.XLA.reactant_jll_libnvrtc() === nothing
        # cuDNN cannot build a runtime-compiled plan without a shared libnvrtc to dlopen,
        # so there is nothing to exercise until the CUDA bundles ship one.
        @info "JLL carries no shared libnvrtc, skipping" maxlog = 1
    else
        # Reactant installs the override at load time, from the JLL's own libnvrtc
        @test haskey(ENV, Reactant.XLA.CUDNN_NVRTC_OVERRIDE_ENV)
        @test isfile(ENV[Reactant.XLA.CUDNN_NVRTC_OVERRIDE_ENV])

        D, H, T, B = 64, 4, 256, 2
        scale = 1 / sqrt(D)
        q = BFloat16.(0.5f0 .* randn(Float32, D, H, T, B))
        k = BFloat16.(0.5f0 .* randn(Float32, D, H, T, B))
        v = BFloat16.(0.5f0 .* randn(Float32, D, H, T, B))
        q_ra, k_ra, v_ra = Reactant.to_rarray.((q, k, v))

        f = @compile cudnn_fmha_causal(q_ra, k_ra, v_ra; scale=scale)
        y = Array(f(q_ra, k_ra, v_ra))
        @test size(y) == (D, H, T, B)
        yref = reference_attention_causal(q, k, v; scale)
        @test maximum(abs, Float32.(y) .- yref) < 2e-2   # bf16 output
    end
end
