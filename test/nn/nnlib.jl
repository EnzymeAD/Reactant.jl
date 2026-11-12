using NNlib, Reactant, Enzyme, Statistics, Test, LuxLib

@testset "Batched Matrix Multiplication" begin
    Reactant.with_config(;
        convolution_precision=PrecisionConfig.HIGH,
        dot_general_precision=PrecisionConfig.HIGH,
    ) do
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 5)
        y = Reactant.TestUtils.construct_test_array(Float32, 3, 2, 5)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)

        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 1)
        y = Reactant.TestUtils.construct_test_array(Float32, 3, 2, 5)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)

        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 5)
        y = Reactant.TestUtils.construct_test_array(Float32, 3, 2, 1)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)
    end
end

@testset "Constant Padding: NNlib.pad_constant" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 4, 4)
    x_ra = Reactant.to_rarray(x)

    # Symmetric Padding
    @test @jit(NNlib.pad_constant(x_ra, (1, 1))) ≈ NNlib.pad_constant(x, (1, 1))
    @test @jit(NNlib.pad_constant(x_ra, (1, 1, 1, 1))) ≈ NNlib.pad_constant(x, (1, 1, 1, 1))

    # Asymmetric Padding
    @test @jit(NNlib.pad_constant(x_ra, (1, 3, 2, 1))) ≈ NNlib.pad_constant(x, (1, 3, 2, 1))
    @test @jit(NNlib.pad_constant(x_ra, (1, 0))) ≈ NNlib.pad_constant(x, (1, 0))

    # Symmetric Padding with value (test type-casting)
    @test @jit(NNlib.pad_constant(x_ra, (1, 1), 2)) ≈ NNlib.pad_constant(x, (1, 1), 2)
    @test @jit(NNlib.pad_constant(x_ra, (1, 1, 1, 1), 2)) ≈
        NNlib.pad_constant(x, (1, 1, 1, 1), 2)

    # Asymmetric Padding with value (test type-casting)
    @test @jit(NNlib.pad_constant(x_ra, (1, 3, 2, 1), 2)) ≈
        NNlib.pad_constant(x, (1, 3, 2, 1), 2)
    @test @jit(NNlib.pad_constant(x_ra, (1, 0), 2)) ≈ NNlib.pad_constant(x, (1, 0), 2)

    # pad_zeros just forward to pad_constant
    @test @jit(NNlib.pad_zeros(x_ra, (1, 1))) ≈ NNlib.pad_zeros(x, (1, 1))
    @test @jit(NNlib.pad_zeros(x_ra, (1, 1, 1, 1))) ≈ NNlib.pad_zeros(x, (1, 1, 1, 1))

    sumabs2(f, x) = sum(abs2, f(x))
    ∇sumabs2(f, x) = Enzyme.gradient(Reverse, sumabs2, Const(f), x)[2]

    pad_fn = Base.Fix2(NNlib.pad_constant, (1, 1, 1, 1))
    @test @jit(∇sumabs2(pad_fn, x_ra)) ≈ 2 .* x

    pad_fn2 = Base.Fix2(NNlib.pad_constant, (1, 0, 1, 3))
    @test @jit(∇sumabs2(pad_fn2, x_ra)) ≈ 2 .* x

    x = Reactant.TestUtils.construct_test_array(ComplexF32, 4, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(NNlib.pad_constant(x_ra, (1, 1))) ≈ NNlib.pad_constant(x, (1, 1))
end

@testset "make_causal_mask" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(NNlib.make_causal_mask(x_ra)) ≈ NNlib.make_causal_mask(x)

    causal_mask2(x) = NNlib.make_causal_mask(x; dims=1)
    @test @jit(causal_mask2(x_ra)) ≈ causal_mask2(x)
end

@testset "softmax/logsoftmax reshaped input" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 3, 4, 5)
    x_ra = reshape(Reactant.to_rarray(x), 12, 5)
    x = reshape(x, 12, 5)

    @test @jit(NNlib.softmax(x_ra)) ≈ NNlib.softmax(x)
    @test @jit(NNlib.logsoftmax(x_ra)) ≈ NNlib.logsoftmax(x)
end

@testset "logsumexp #1593" begin
    x = collect(Float32, 1:16)
    x_ra = Reactant.to_rarray(x)

    y = logsumexp(x)
    y_ra = @jit(logsumexp(x_ra))
    @test Float32(y_ra) ≈ y
end

@testset "Fused Dense" begin
    sumabs2fuseddense(act, weight, x, bias) =
        sum(abs2, fused_dense_bias_activation(act, weight, x, bias))

    function ∇fuseddense(act, weight, x, bias)
        return Enzyme.gradient(Reverse, sumabs2fuseddense, Const(act), weight, x, bias)[2:end]
    end

    function ∇fuseddense(actgradfn, act, weight, x, bias)
        z = weight * x
        if bias !== nothing
            z .+= bias
        end
        Ω = act.(z)

        δ = 2 .* actgradfn.(Ω, z) .* Ω

        ∂weight = δ * x'
        if bias !== nothing
            ∂bias = vec(sum(δ; dims=2))
        else
            ∂bias = nothing
        end
        ∂x = weight' * δ
        return ∂weight, ∂x, ∂bias
    end

    @testset "Activation: $act | bias=$has_bias" for (act, gradfn) in (
            (identity, (Ω, x) -> one(Ω)),
            (relu, (Ω, x) -> (Ω > 0)),
            (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
            (tanh, (Ω, x) -> conj(1 - Ω^2)),
            (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
        ),
        has_bias in (true, false)

        weight = Reactant.TestUtils.construct_test_array(Float32, 9, 10)
        x = Reactant.TestUtils.construct_test_array(Float32, 10, 12)
        bias = has_bias ? Reactant.TestUtils.construct_test_array(Float32, 9) : nothing

        weight_ra = Reactant.to_rarray(weight)
        x_ra = Reactant.to_rarray(x)
        bias_ra = Reactant.to_rarray(bias)

        y_compile = @jit fused_dense_bias_activation(act, weight_ra, x_ra, bias_ra)
        y_res = fused_dense_bias_activation(act, weight, x, bias)
        @test y_res ≈ y_compile atol = 1e-5 rtol = 1e-2

        @testset "Enzyme: fused_dense_bias_activation" begin
            dw_compile, dx_compile, db_compile = @jit ∇fuseddense(
                act, weight_ra, x_ra, bias_ra
            )

            dw_gt, dx_gt, db_gt = @jit ∇fuseddense(gradfn, act, weight_ra, x_ra, bias_ra)

            @test dw_gt ≈ dw_compile atol = 1e-5 rtol = 1e-2
            @test dx_gt ≈ dx_compile atol = 1e-5 rtol = 1e-2
            has_bias && @test db_gt ≈ db_compile atol = 1e-5 rtol = 1e-2
        end
    end
end

@testset "Scaled Dot-Product Attention" begin
    platform = lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    # Attention lowering emits DEFAULT-precision dot_general ops:
    # - CPU uses IEEE FP32 (24-bit mantissa, eps ≈ 1.2e-7)
    # - CUDA/ROCm GPUs use TF32 Tensor Cores (10-bit mantissa, eps ≈ 9.8e-4)
    # - TPUs execute DEFAULT Float32 matmuls in 1-pass BFloat16 on the MXU
    #   (7-bit mantissa, eps ≈ 7.8e-3), while reference ops run under HIGH precision.
    fwd_atol, fwd_rtol = if platform == "cpu"
        (1e-5, 1e-4)
    elseif platform == "tpu"
        (1e-2, 2e-2)
    else
        (1e-3, 1e-2)
    end
    ad_atol, ad_rtol = if platform == "cpu"
        (1e-4, 1e-3)
    elseif platform == "tpu"
        (3e-2, 5e-2)
    else
        (1e-3, 1e-2)
    end
    ad2_atol, ad2_rtol = if platform == "cpu"
        (1e-4, 1e-3)
    elseif platform == "tpu"
        (5e-2, 8e-2)
    else
        (2e-3, 2e-2)
    end

    # Deterministic zero-centered, non-collinear test inputs in [-1, 1] so head
    # vectors are not nearly constant along D (avoiding ~98% DC cancellation in
    # softmax backward dO' * V - dO' * O on BFloat16 / TF32 matmul backends).
    function make_attn_array(::Type{T}, dims::Int...; seed::Int=1) where {T}
        n = prod(dims)
        vals = T[
            sin(Float64(i) * (0.73 + 0.11 * seed) + cos(Float64(i + 17 * seed) * 0.31)) for
            i in 1:n
        ]
        return reshape(vals, dims...)
    end

    Reactant.with_config(;
        convolution_precision=PrecisionConfig.HIGH,
        dot_general_precision=PrecisionConfig.HIGH,
    ) do
        @testset "IR emission and lowering" begin
            q = Reactant.to_rarray(make_attn_array(Float32, 8, 4, 6, 2; seed=1))
            k = Reactant.to_rarray(make_attn_array(Float32, 8, 4, 6, 2; seed=2))
            v = Reactant.to_rarray(make_attn_array(Float32, 8, 4, 6, 2; seed=3))

            hlo_pre = repr(
                @code_hlo optimize = false NNlib.scaled_dot_product_attention(q, k, v)
            )
            @test occursin("enzymexla.ml.attention", hlo_pre)

            hlo_post = repr(@code_hlo NNlib.scaled_dot_product_attention(q, k, v))
            @test !occursin("enzymexla.ml.attention", hlo_post)

            if isdefined(NNlib, :is_fused_attention)
                @test @jit(NNlib.is_fused_attention(q, k, v))
                @test !@jit(NNlib.is_fused_attention(q, k, v; fdrop=x -> 2x))
            end

            # Verify CUDA cuDNN fMHA lowering & execution for bf16 with head dim multiple of 8.
            # Convert Float32 <-> BFloat16 inside @jit (XLA) rather than broadcasting BFloat16
            # on host Arrays to avoid Julia's LLVM X86ISD::VFPROUND crash on avx512_bf16 hosts
            # (JuliaLang/julia#62666).
            q_bf16_src = Reactant.to_rarray(
                make_attn_array(Float32, 64, 4, 64, 2; seed=101)
            )
            k_bf16_src = Reactant.to_rarray(
                make_attn_array(Float32, 64, 4, 64, 2; seed=102) .* 0.7f0
            )
            v_bf16_src = Reactant.to_rarray(
                make_attn_array(Float32, 64, 4, 64, 2; seed=103) .* -0.5f0
            )
            to_bf16_fn(x) = Reactant.BFloat16.(x)
            to_f32_fn(x) = Float32.(x)
            q_bf16 = @jit to_bf16_fn(q_bf16_src)
            k_bf16 = @jit to_bf16_fn(k_bf16_src)
            v_bf16 = @jit to_bf16_fn(v_bf16_src)
            hlo_cuda = repr(
                @code_hlo optimize = " lower-enzymexla-math{backend=cuda}" NNlib.scaled_dot_product_attention(
                    q_bf16, k_bf16, v_bf16; is_causal=true
                )
            )
            @test occursin("__cudnn\$fmhaSoftmax", hlo_cuda)

            hlo_sm75 = repr(
                @code_hlo optimize = " lower-enzymexla-math{backend=cuda_sm75}" NNlib.scaled_dot_product_attention(
                    q_bf16, k_bf16, v_bf16; is_causal=true
                )
            )
            @test !occursin("__cudnn\$fmhaSoftmax", hlo_sm75)
            @test !occursin("enzymexla.ml.attention", hlo_sm75)

            q_f32_ra = @jit to_f32_fn(q_bf16)
            k_f32_ra = @jit to_f32_fn(k_bf16)
            v_f32_ra = @jit to_f32_fn(v_bf16)
            bf16_fwd_f32(q, k, v) =
                Float32.(NNlib.scaled_dot_product_attention(q, k, v; is_causal=true))
            y_bf16_f32 = @jit bf16_fwd_f32(q_bf16, k_bf16, v_bf16)
            @test Array(y_bf16_f32) ≈ NNlib.scaled_dot_product_attention(
                Array(q_f32_ra), Array(k_f32_ra), Array(v_f32_ra); is_causal=true
            ) atol = 5e-2 rtol = 5e-2

            bf16_loss(q, k, v) = sum(
                abs2,
                Float32.(NNlib.scaled_dot_product_attention(q, k, v; is_causal=true)),
            )
            bf16_loss_ref(q, k, v) = sum(
                abs2,
                first(
                    NNlib.scaled_dot_product_attention_with_weights(
                        q, k, v; is_causal=true
                    ),
                ),
            )
            bf16_grad_f32(q, k, v) =
                map(g -> Float32.(g), Enzyme.gradient(Reverse, bf16_loss, q, k, v))
            g_bf16_f32 = @jit bf16_grad_f32(q_bf16, k_bf16, v_bf16)
            g_f32_ref = @jit Enzyme.gradient(
                Reverse, bf16_loss_ref, q_f32_ra, k_f32_ra, v_f32_ra
            )
            for idx in 1:3
                @test Array(g_bf16_f32[idx]) ≈ Array(g_f32_ref[idx]) atol = 5e-2 rtol = 5e-2
            end
        end

        @testset "Forward and AD (MHA & GQA, causal=$is_causal, has_bias=$has_bias, has_mask=$has_mask)" for nkvheads in
                                                                                                             (
                4, 2
            ),
            is_causal in (false, true),
            has_bias in (false, true),
            has_mask in (false, true)

            q = make_attn_array(Float32, 8, 4, 6, 2; seed=1)
            k = make_attn_array(Float32, 8, nkvheads, 5, 2; seed=2)
            v = make_attn_array(Float32, 6, nkvheads, 5, 2; seed=3)
            bias = if has_bias
                make_attn_array(Float32, 5, 6, 4, 1; seed=4)
            else
                nothing
            end
            mask = if has_mask
                # Ensure every query attends to at least one key so CPU NNlib doesn't produce 0/0 NaN
                m = trues(5, 6, 1, 2)
                m[end, :, 1, 1] .= false
                m[1, 2:end, 1, 2] .= false
                m
            else
                nothing
            end

            q_ra = Reactant.to_rarray(q)
            k_ra = Reactant.to_rarray(k)
            v_ra = Reactant.to_rarray(v)
            bias_ra = Reactant.to_rarray(bias)
            mask_ra = Reactant.to_rarray(mask)

            y_cpu = NNlib.scaled_dot_product_attention(q, k, v, bias; is_causal, mask)
            y_ra = @jit NNlib.scaled_dot_product_attention(
                q_ra, k_ra, v_ra, bias_ra; is_causal, mask=mask_ra
            )
            @test Array(y_ra) ≈ y_cpu atol = fwd_atol rtol = fwd_rtol

            function loss_fn(q, k, v, bias, mask)
                return sum(
                    abs2, NNlib.scaled_dot_product_attention(q, k, v, bias; is_causal, mask)
                )
            end
            function ref_loss_fn(q, k, v, bias, mask)
                return sum(
                    abs2,
                    first(
                        NNlib.scaled_dot_product_attention_with_weights(
                            q, k, v, bias; is_causal, mask
                        ),
                    ),
                )
            end
            grads_ref = @jit Enzyme.gradient(
                Reverse,
                ref_loss_fn,
                q_ra,
                k_ra,
                v_ra,
                has_bias ? bias_ra : Const(nothing),
                Const(mask_ra),
            )
            grads_ra = @jit Enzyme.gradient(
                Reverse,
                loss_fn,
                q_ra,
                k_ra,
                v_ra,
                has_bias ? bias_ra : Const(nothing),
                Const(mask_ra),
            )
            @test Array(grads_ra[1]) ≈ Array(grads_ref[1]) atol = ad_atol rtol = ad_rtol
            @test Array(grads_ra[2]) ≈ Array(grads_ref[2]) atol = ad_atol rtol = ad_rtol
            @test Array(grads_ra[3]) ≈ Array(grads_ref[3]) atol = ad_atol rtol = ad_rtol
            if has_bias
                @test Array(grads_ra[4]) ≈ Array(grads_ref[4]) atol = ad_atol rtol = ad_rtol
            end

            # Forward-mode AD tangent array check
            dq = make_attn_array(Float32, size(q)...; seed=21)
            dk = make_attn_array(Float32, size(k)...; seed=22)
            dv = make_attn_array(Float32, size(v)...; seed=23)
            dq_ra = Reactant.to_rarray(dq)
            dk_ra = Reactant.to_rarray(dk)
            dv_ra = Reactant.to_rarray(dv)

            function fwd_fn(q, k, v, bias, mask)
                return NNlib.scaled_dot_product_attention(q, k, v, bias; is_causal, mask)
            end
            function ref_fwd_fn(q, k, v, bias, mask)
                return first(
                    NNlib.scaled_dot_product_attention_with_weights(
                        q, k, v, bias; is_causal, mask
                    ),
                )
            end
            dy_fwd = @jit Enzyme.autodiff(
                Forward,
                Const(fwd_fn),
                Duplicated,
                Duplicated(q_ra, dq_ra),
                Duplicated(k_ra, dk_ra),
                Duplicated(v_ra, dv_ra),
                Const(bias_ra),
                Const(mask_ra),
            )
            dy_ref = @jit Enzyme.autodiff(
                Forward,
                Const(ref_fwd_fn),
                Duplicated,
                Duplicated(q_ra, dq_ra),
                Duplicated(k_ra, dk_ra),
                Duplicated(v_ra, dv_ra),
                Const(bias_ra),
                Const(mask_ra),
            )
            @test Array(dy_fwd[1]) ≈ Array(dy_ref[1]) atol = ad_atol rtol = ad_rtol
        end

        @testset "Padding lengths, custom scale, 5D batching, and fdrop" begin
            q = make_attn_array(Float32, 4, 2, 5, 3; seed=31)
            k = make_attn_array(Float32, 4, 2, 6, 3; seed=32)
            v = make_attn_array(Float32, 4, 2, 6, 3; seed=33)
            q_ra = Reactant.to_rarray(q)
            k_ra = Reactant.to_rarray(k)
            v_ra = Reactant.to_rarray(v)

            if hasmethod(
                NNlib.scaled_dot_product_attention,
                Tuple{typeof(q),typeof(k),typeof(v)},
                (:q_lengths, :kv_lengths),
            )
                q_lengths = Int32[5, 3, 4]
                kv_lengths = Int32[6, 4, 2]
                q_len_ra = Reactant.to_rarray(q_lengths)
                kv_len_ra = Reactant.to_rarray(kv_lengths)

                y_cpu = NNlib.scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths)
                y_ra = @jit NNlib.scaled_dot_product_attention(
                    q_ra, k_ra, v_ra; q_lengths=q_len_ra, kv_lengths=kv_len_ra
                )
                @test Array(y_ra) ≈ y_cpu atol = fwd_atol rtol = fwd_rtol
            end

            # Traced scale differentiation
            s = Float32(0.25)
            s_ra = ConcreteRNumber(s)
            scale_loss(q, k, v, s) =
                sum(abs2, NNlib.scaled_dot_product_attention(q, k, v; scale=s))
            ref_scale_loss(q, k, v, s) = sum(
                abs2,
                first(NNlib.scaled_dot_product_attention_with_weights(q, k, v; scale=s)),
            )
            gs_ref = @jit Enzyme.gradient(Reverse, ref_scale_loss, q_ra, k_ra, v_ra, s_ra)
            gs_ra = @jit Enzyme.gradient(Reverse, scale_loss, q_ra, k_ra, v_ra, s_ra)
            @test Float32(gs_ra[4]) ≈ Float32(gs_ref[4]) atol = ad_atol rtol = ad_rtol

            # 5D multiple batch dimensions
            q5 = make_attn_array(Float32, 4, 2, 5, 3, 2; seed=34)
            k5 = make_attn_array(Float32, 4, 2, 6, 3, 2; seed=35)
            v5 = make_attn_array(Float32, 4, 2, 6, 3, 2; seed=36)
            q5_ra = Reactant.to_rarray(q5)
            k5_ra = Reactant.to_rarray(k5)
            v5_ra = Reactant.to_rarray(v5)
            @test Array(@jit(NNlib.scaled_dot_product_attention(q5_ra, k5_ra, v5_ra))) ≈
                NNlib.scaled_dot_product_attention(q5, k5, v5) atol = fwd_atol rtol =
                fwd_rtol

            # Custom fdrop fallback
            fdrop_half(x) = 0.5f0 .* x
            @test Array(
                @jit(NNlib.scaled_dot_product_attention(q_ra, k_ra, v_ra; fdrop=fdrop_half))
            ) ≈ NNlib.scaled_dot_product_attention(q, k, v; fdrop=fdrop_half) atol =
                fwd_atol rtol = fwd_rtol
        end

        @testset "Batching, Batched AD, LSE AD, AttentionGrad AD, and Higher-Order AD" begin
            q = make_attn_array(Float32, 4, 4, 5, 2; seed=41)
            k = make_attn_array(Float32, 4, 2, 6, 2; seed=42)
            v = make_attn_array(Float32, 4, 2, 6, 2; seed=43)
            bias = make_attn_array(Float32, 6, 5, 4, 1; seed=44)

            dq1 = make_attn_array(Float32, size(q)...; seed=45)
            dq2 = make_attn_array(Float32, size(q)...; seed=46) .* 0.7f0
            dk1 = make_attn_array(Float32, size(k)...; seed=47)
            dk2 = make_attn_array(Float32, size(k)...; seed=48) .* -0.5f0
            dv1 = make_attn_array(Float32, size(v)...; seed=49)
            dv2 = make_attn_array(Float32, size(v)...; seed=50) .* 0.3f0
            db1 = make_attn_array(Float32, size(bias)...; seed=51)
            db2 = make_attn_array(Float32, size(bias)...; seed=52) .* -0.4f0
            dout1 = make_attn_array(Float32, size(q)...; seed=53)
            dout2 = make_attn_array(Float32, size(q)...; seed=54) .* -0.6f0

            q_ra = Reactant.to_rarray(q)
            k_ra = Reactant.to_rarray(k)
            v_ra = Reactant.to_rarray(v)
            bias_ra = Reactant.to_rarray(bias)
            dq1_ra = Reactant.to_rarray(dq1)
            dq2_ra = Reactant.to_rarray(dq2)
            dk1_ra = Reactant.to_rarray(dk1)
            dk2_ra = Reactant.to_rarray(dk2)
            dv1_ra = Reactant.to_rarray(dv1)
            dv2_ra = Reactant.to_rarray(dv2)
            db1_ra = Reactant.to_rarray(db1)
            db2_ra = Reactant.to_rarray(db2)
            dout1_ra = Reactant.to_rarray(dout1)
            dout2_ra = Reactant.to_rarray(dout2)

            attn_gqa(q, k, v, b) =
                NNlib.scaled_dot_product_attention(q, k, v, b; is_causal=true)
            attn_gqa_ref(q, k, v, b) = first(
                NNlib.scaled_dot_product_attention_with_weights(q, k, v, b; is_causal=true),
            )

            # 1. Batched Forward-mode AD (width = 2)
            bfwd = @jit Enzyme.autodiff(
                Forward,
                Const(attn_gqa),
                BatchDuplicated,
                BatchDuplicated(q_ra, (dq1_ra, dq2_ra)),
                BatchDuplicated(k_ra, (dk1_ra, dk2_ra)),
                BatchDuplicated(v_ra, (dv1_ra, dv2_ra)),
                BatchDuplicated(bias_ra, (db1_ra, db2_ra)),
            )
            bfwd_ref1 = @jit Enzyme.autodiff(
                Forward,
                Const(attn_gqa_ref),
                Duplicated,
                Duplicated(q_ra, dq1_ra),
                Duplicated(k_ra, dk1_ra),
                Duplicated(v_ra, dv1_ra),
                Duplicated(bias_ra, db1_ra),
            )
            bfwd_ref2 = @jit Enzyme.autodiff(
                Forward,
                Const(attn_gqa_ref),
                Duplicated,
                Duplicated(q_ra, dq2_ra),
                Duplicated(k_ra, dk2_ra),
                Duplicated(v_ra, dv2_ra),
                Duplicated(bias_ra, db2_ra),
            )
            @test Array(bfwd[1][1]) ≈ Array(bfwd_ref1[1]) atol = ad_atol rtol = ad_rtol
            @test Array(bfwd[1][2]) ≈ Array(bfwd_ref2[1]) atol = ad_atol rtol = ad_rtol

            # 2. Batched Reverse-mode AD (width = 2)
            function attn_mut!(out, q, k, v, b)
                copyto!(out, attn_gqa(q, k, v, b))
                return nothing
            end

            out_ra = Reactant.to_rarray(zeros(Float32, size(q)...))
            gq1_ra = Reactant.to_rarray(zeros(Float32, size(q)...))
            gq2_ra = Reactant.to_rarray(zeros(Float32, size(q)...))
            gk1_ra = Reactant.to_rarray(zeros(Float32, size(k)...))
            gk2_ra = Reactant.to_rarray(zeros(Float32, size(k)...))
            gv1_ra = Reactant.to_rarray(zeros(Float32, size(v)...))
            gv2_ra = Reactant.to_rarray(zeros(Float32, size(v)...))
            gb1_ra = Reactant.to_rarray(zeros(Float32, size(bias)...))
            gb2_ra = Reactant.to_rarray(zeros(Float32, size(bias)...))

            d1_copy = copy(dout1_ra)
            d2_copy = copy(dout2_ra)

            @jit Enzyme.autodiff(
                Reverse,
                attn_mut!,
                BatchDuplicated(out_ra, (d1_copy, d2_copy)),
                BatchDuplicated(q_ra, (gq1_ra, gq2_ra)),
                BatchDuplicated(k_ra, (gk1_ra, gk2_ra)),
                BatchDuplicated(v_ra, (gv1_ra, gv2_ra)),
                BatchDuplicated(bias_ra, (gb1_ra, gb2_ra)),
            )
            attn_vjp_ref(qq, kk, vv, bb, dd) = sum(attn_gqa_ref(qq, kk, vv, bb) .* dd)
            brev_ref1 = @jit Enzyme.gradient(
                Reverse, attn_vjp_ref, q_ra, k_ra, v_ra, bias_ra, Const(dout1_ra)
            )
            brev_ref2 = @jit Enzyme.gradient(
                Reverse, attn_vjp_ref, q_ra, k_ra, v_ra, bias_ra, Const(dout2_ra)
            )

            @test Array(gq1_ra) ≈ Array(brev_ref1[1]) atol = ad_atol rtol = ad_rtol
            @test Array(gq2_ra) ≈ Array(brev_ref2[1]) atol = ad_atol rtol = ad_rtol
            @test Array(gk1_ra) ≈ Array(brev_ref1[2]) atol = ad_atol rtol = ad_rtol
            @test Array(gk2_ra) ≈ Array(brev_ref2[2]) atol = ad_atol rtol = ad_rtol
            @test Array(gv1_ra) ≈ Array(brev_ref1[3]) atol = ad_atol rtol = ad_rtol
            @test Array(gv2_ra) ≈ Array(brev_ref2[3]) atol = ad_atol rtol = ad_rtol
            @test Array(gb1_ra) ≈ Array(brev_ref1[4]) atol = ad_atol rtol = ad_rtol
            @test Array(gb2_ra) ≈ Array(brev_ref2[4]) atol = ad_atol rtol = ad_rtol

            # 3. Active LSE forward & reverse AD (unbatched & batched) on Ops.attention(; return_lse=true)
            dlse1 = make_attn_array(Float32, 5, 4, 2; seed=55)
            dlse2 = make_attn_array(Float32, 5, 4, 2; seed=56) .* -0.3f0
            dlse1_ra = Reactant.to_rarray(dlse1)
            dlse2_ra = Reactant.to_rarray(dlse2)

            function lse_loss(q, k, v, b, wo, wl)
                o, lse = Reactant.Ops.attention(q, k, v, b; return_lse=true)
                return sum(o .* wo) + sum(lse .* wl)
            end
            function lse_loss_ref(q, k, v, b, wo, wl)
                D, Nq, T, B = size(q)
                _, Nkv, S, _ = size(k)
                G = Nq ÷ Nkv
                k_exp = reshape(
                    repeat(reshape(k, D, 1, Nkv, S, B), 1, G, 1, 1, 1), D, Nq, S, B
                )
                v_exp = reshape(
                    repeat(reshape(v, D, 1, Nkv, S, B), 1, G, 1, 1, 1), D, Nq, S, B
                )
                k3 = reshape(permutedims(k_exp, (3, 1, 2, 4)), S, D, Nq * B)
                q3 = reshape(permutedims(q, (1, 3, 2, 4)), D, T, Nq * B)
                scores =
                    reshape(NNlib.batched_mul(k3, q3), S, T, Nq, B) .*
                    inv(sqrt(Float32(D))) .+ b
                m = maximum(scores; dims=1)
                se = sum(exp.(scores .- m); dims=1)
                lse = dropdims(m .+ log.(se); dims=1)
                p3 = reshape(exp.(scores .- m) ./ se, S, T, Nq * B)
                v3 = reshape(permutedims(v_exp, (1, 3, 2, 4)), D, S, Nq * B)
                o = permutedims(
                    reshape(NNlib.batched_mul(v3, p3), D, T, Nq, B), (1, 3, 2, 4)
                )
                return sum(o .* wo) + sum(lse .* wl)
            end

            glse = @jit Enzyme.gradient(
                Reverse,
                lse_loss,
                q_ra,
                k_ra,
                v_ra,
                bias_ra,
                Const(dout1_ra),
                Const(dlse1_ra),
            )
            glse_ref = @jit Enzyme.gradient(
                Reverse,
                lse_loss_ref,
                q_ra,
                k_ra,
                v_ra,
                bias_ra,
                Const(dout1_ra),
                Const(dlse1_ra),
            )
            @test Array(glse[1]) ≈ Array(glse_ref[1]) atol = ad_atol rtol = ad_rtol
            @test Array(glse[2]) ≈ Array(glse_ref[2]) atol = ad_atol rtol = ad_rtol
            @test Array(glse[3]) ≈ Array(glse_ref[3]) atol = ad_atol rtol = ad_rtol
            @test Array(glse[4]) ≈ Array(glse_ref[4]) atol = ad_atol rtol = ad_rtol

            flse = @jit Enzyme.autodiff(
                Forward,
                Const(lse_loss),
                BatchDuplicated,
                BatchDuplicated(q_ra, (dq1_ra, dq2_ra)),
                BatchDuplicated(k_ra, (dk1_ra, dk2_ra)),
                BatchDuplicated(v_ra, (dv1_ra, dv2_ra)),
                BatchDuplicated(bias_ra, (db1_ra, db2_ra)),
                Const(dout1_ra),
                Const(dlse1_ra),
            )
            flse_ref1 = @jit Enzyme.autodiff(
                Forward,
                Const(lse_loss_ref),
                Duplicated,
                Duplicated(q_ra, dq1_ra),
                Duplicated(k_ra, dk1_ra),
                Duplicated(v_ra, dv1_ra),
                Duplicated(bias_ra, db1_ra),
                Const(dout1_ra),
                Const(dlse1_ra),
            )
            flse_ref2 = @jit Enzyme.autodiff(
                Forward,
                Const(lse_loss_ref),
                Duplicated,
                Duplicated(q_ra, dq2_ra),
                Duplicated(k_ra, dk2_ra),
                Duplicated(v_ra, dv2_ra),
                Duplicated(bias_ra, db2_ra),
                Const(dout1_ra),
                Const(dlse1_ra),
            )
            @test Float32(flse[1][1]) ≈ Float32(flse_ref1[1]) atol = ad_atol rtol = ad_rtol
            @test Float32(flse[1][2]) ≈ Float32(flse_ref2[1]) atol = ad_atol rtol = ad_rtol

            # 4. Auto-batching (@trace for -> enzyme.batch) on both attention and attention_grad
            function looped_attn_and_grad(qs, ks, vs, douts, bias)
                dq_out = similar(qs)
                @trace for i in 1:size(qs, 5)
                    qi = qs[:, :, :, :, i]
                    ki = ks[:, :, :, :, i]
                    vi = vs[:, :, :, :, i]
                    doi = douts[:, :, :, :, i]
                    oi, lsei = Reactant.Ops.attention(
                        qi, ki, vi, bias; is_causal=true, return_lse=true
                    )
                    dqi, _, _, _ = Reactant.Ops.attention_grad(
                        qi, ki, vi, oi, lsei, doi, bias; is_causal=true
                    )
                    dq_out[:, :, :, :, i] = dqi
                end
                return dq_out
            end

            qs_arr = make_attn_array(Float32, 4, 4, 5, 2, 3; seed=61)
            ks_arr = make_attn_array(Float32, 4, 2, 6, 2, 3; seed=62)
            vs_arr = make_attn_array(Float32, 4, 2, 6, 2, 3; seed=63)
            dos_arr = make_attn_array(Float32, 4, 4, 5, 2, 3; seed=64)
            qs_ra = Reactant.to_rarray(qs_arr)
            ks_ra = Reactant.to_rarray(ks_arr)
            vs_ra = Reactant.to_rarray(vs_arr)
            dos_ra = Reactant.to_rarray(dos_arr)

            hlo_looped = repr(
                @code_hlo looped_attn_and_grad(qs_ra, ks_ra, vs_ra, dos_ra, bias_ra)
            )
            @test !occursin("stablehlo.while", hlo_looped)

            dq_batched = Array(
                @jit looped_attn_and_grad(qs_ra, ks_ra, vs_ra, dos_ra, bias_ra)
            )
            for i in 1:3
                qi_ra = Reactant.to_rarray(qs_arr[:, :, :, :, i])
                ki_ra = Reactant.to_rarray(ks_arr[:, :, :, :, i])
                vi_ra = Reactant.to_rarray(vs_arr[:, :, :, :, i])
                doi_ra = Reactant.to_rarray(dos_arr[:, :, :, :, i])
                single_grad = @jit Enzyme.gradient(
                    Reverse, attn_vjp_ref, qi_ra, ki_ra, vi_ra, bias_ra, Const(doi_ra)
                )
                @test dq_batched[:, :, :, :, i] ≈ Array(single_grad[1]) atol = ad_atol rtol =
                    ad_rtol
            end

            # 5. Direct Forward, Batched Forward, Reverse, and Batched Reverse AD on Ops.attention_grad
            out0_ra, lse0_ra = @jit Reactant.Ops.attention(
                q_ra, k_ra, v_ra, bias_ra; is_causal=true, return_lse=true
            )
            tout1_ra = Reactant.to_rarray(
                make_attn_array(Float32, size(q)...; seed=71) .* 0.25f0
            )
            tout2_ra = Reactant.to_rarray(
                make_attn_array(Float32, size(q)...; seed=72) .* -0.35f0
            )
            tlse1_ra = Reactant.to_rarray(dlse1 .* 0.4f0)
            tlse2_ra = Reactant.to_rarray(dlse2 .* 0.5f0)

            bar_dq1 = Reactant.to_rarray(
                make_attn_array(Float32, size(q)...; seed=73) .* 0.3f0
            )
            bar_dk1 = Reactant.to_rarray(
                make_attn_array(Float32, size(k)...; seed=74) .* -0.2f0
            )
            bar_dv1 = Reactant.to_rarray(
                make_attn_array(Float32, size(v)...; seed=75) .* 0.4f0
            )
            bar_db1 = Reactant.to_rarray(
                make_attn_array(Float32, size(bias)...; seed=76) .* -0.5f0
            )

            function grad_scalar_obj(q, k, v, o, lse, dout, b, wq, wk, wv, wb)
                dq, dk, dv, db = Reactant.Ops.attention_grad(
                    q, k, v, o, lse, dout, b; is_causal=true
                )
                return sum(dq .* wq) + sum(dk .* wk) + sum(dv .* wv) + sum(db .* wb)
            end

            grad_fwd_b = @jit Enzyme.autodiff(
                Forward,
                Const(grad_scalar_obj),
                BatchDuplicated,
                BatchDuplicated(q_ra, (dq1_ra, dq2_ra)),
                BatchDuplicated(k_ra, (dk1_ra, dk2_ra)),
                BatchDuplicated(v_ra, (dv1_ra, dv2_ra)),
                BatchDuplicated(out0_ra, (tout1_ra, tout2_ra)),
                BatchDuplicated(lse0_ra, (tlse1_ra, tlse2_ra)),
                BatchDuplicated(dout1_ra, (dout2_ra, dq1_ra)),
                BatchDuplicated(bias_ra, (db1_ra, db2_ra)),
                Const(bar_dq1),
                Const(bar_dk1),
                Const(bar_dv1),
                Const(bar_db1),
            )

            grad_rev = @jit Enzyme.gradient(
                Reverse,
                grad_scalar_obj,
                q_ra,
                k_ra,
                v_ra,
                out0_ra,
                lse0_ra,
                dout1_ra,
                bias_ra,
                Const(bar_dq1),
                Const(bar_dk1),
                Const(bar_dv1),
                Const(bar_db1),
            )

            # Adjoint consistency check across all 7 active inputs of Ops.attention_grad
            rhs_adj1 =
                sum(Array(grad_rev[1]) .* dq1) +
                sum(Array(grad_rev[2]) .* dk1) +
                sum(Array(grad_rev[3]) .* dv1) +
                sum(Array(grad_rev[4]) .* Array(tout1_ra)) +
                sum(Array(grad_rev[5]) .* Array(tlse1_ra)) +
                sum(Array(grad_rev[6]) .* dout2) +
                sum(Array(grad_rev[7]) .* db1)
            rhs_adj2 =
                sum(Array(grad_rev[1]) .* dq2) +
                sum(Array(grad_rev[2]) .* dk2) +
                sum(Array(grad_rev[3]) .* dv2) +
                sum(Array(grad_rev[4]) .* Array(tout2_ra)) +
                sum(Array(grad_rev[5]) .* Array(tlse2_ra)) +
                sum(Array(grad_rev[6]) .* dq1) +
                sum(Array(grad_rev[7]) .* db2)
            @test Float32(grad_fwd_b[1][1]) ≈ rhs_adj1 atol = ad2_atol rtol = ad2_rtol
            @test Float32(grad_fwd_b[1][2]) ≈ rhs_adj2 atol = ad2_atol rtol = ad2_rtol

            # Batched Reverse AD (width = 2) directly on Ops.attention_grad
            function attn_grad_mut!(dq_o, q, k, v, o, lse, dout, b)
                dq, _, _, _ = Reactant.Ops.attention_grad(
                    q, k, v, o, lse, dout, b; is_causal=true
                )
                copyto!(dq_o, dq)
                return nothing
            end
            dq_out_ra = Reactant.to_rarray(zeros(Float32, size(q)...))
            bar_dq2 = copy(dout2_ra)
            bar_dq1_c = copy(bar_dq1)
            rq1 = Reactant.to_rarray(zeros(Float32, size(q)...))
            rq2 = Reactant.to_rarray(zeros(Float32, size(q)...))
            rk1 = Reactant.to_rarray(zeros(Float32, size(k)...))
            rk2 = Reactant.to_rarray(zeros(Float32, size(k)...))
            rv1 = Reactant.to_rarray(zeros(Float32, size(v)...))
            rv2 = Reactant.to_rarray(zeros(Float32, size(v)...))
            ro1 = Reactant.to_rarray(zeros(Float32, size(q)...))
            ro2 = Reactant.to_rarray(zeros(Float32, size(q)...))
            rl1 = Reactant.to_rarray(zeros(Float32, size(lse0_ra)...))
            rl2 = Reactant.to_rarray(zeros(Float32, size(lse0_ra)...))
            rdo1 = Reactant.to_rarray(zeros(Float32, size(q)...))
            rdo2 = Reactant.to_rarray(zeros(Float32, size(q)...))
            rb1 = Reactant.to_rarray(zeros(Float32, size(bias)...))
            rb2 = Reactant.to_rarray(zeros(Float32, size(bias)...))

            @jit Enzyme.autodiff(
                Reverse,
                attn_grad_mut!,
                BatchDuplicated(dq_out_ra, (bar_dq1_c, bar_dq2)),
                BatchDuplicated(q_ra, (rq1, rq2)),
                BatchDuplicated(k_ra, (rk1, rk2)),
                BatchDuplicated(v_ra, (rv1, rv2)),
                BatchDuplicated(out0_ra, (ro1, ro2)),
                BatchDuplicated(lse0_ra, (rl1, rl2)),
                BatchDuplicated(dout1_ra, (rdo1, rdo2)),
                BatchDuplicated(bias_ra, (rb1, rb2)),
            )
            function attn_grad_dq_vjp(qq, kk, vv, oo, ll, dd, bb, ww)
                dq, _, _, _ = Reactant.Ops.attention_grad(
                    qq, kk, vv, oo, ll, dd, bb; is_causal=true
                )
                return sum(dq .* ww)
            end
            single_rev1 = @jit Enzyme.gradient(
                Reverse,
                attn_grad_dq_vjp,
                q_ra,
                k_ra,
                v_ra,
                out0_ra,
                lse0_ra,
                dout1_ra,
                bias_ra,
                Const(bar_dq1),
            )
            single_rev2 = @jit Enzyme.gradient(
                Reverse,
                attn_grad_dq_vjp,
                q_ra,
                k_ra,
                v_ra,
                out0_ra,
                lse0_ra,
                dout1_ra,
                bias_ra,
                Const(dout2_ra),
            )
            @test Array(rq1) ≈ Array(single_rev1[1]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rk1) ≈ Array(single_rev1[2]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rv1) ≈ Array(single_rev1[3]) atol = ad2_atol rtol = ad2_rtol
            @test Array(ro1) ≈ Array(single_rev1[4]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rl1) ≈ Array(single_rev1[5]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rdo1) ≈ Array(single_rev1[6]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rb1) ≈ Array(single_rev1[7]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rq2) ≈ Array(single_rev2[1]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rk2) ≈ Array(single_rev2[2]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rv2) ≈ Array(single_rev2[3]) atol = ad2_atol rtol = ad2_rtol
            @test Array(ro2) ≈ Array(single_rev2[4]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rl2) ≈ Array(single_rev2[5]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rdo2) ≈ Array(single_rev2[6]) atol = ad2_atol rtol = ad2_rtol
            @test Array(rb2) ≈ Array(single_rev2[7]) atol = ad2_atol rtol = ad2_rtol

            # 6. Second-order AD (Batched Forward-over-Reverse HVP & Reverse-over-Reverse)
            sdpa_loss(q, k, v, b) = sum(abs2, attn_gqa(q, k, v, b))
            sdpa_loss_ref(q, k, v, b) = sum(abs2, attn_gqa_ref(q, k, v, b))

            grad_fn(q, k, v, b) = Enzyme.gradient(Reverse, sdpa_loss, q, k, v, b)
            grad_fn_ref(q, k, v, b) = Enzyme.gradient(Reverse, sdpa_loss_ref, q, k, v, b)

            hvp_b = @jit Enzyme.autodiff(
                Forward,
                Const(grad_fn),
                BatchDuplicated,
                BatchDuplicated(q_ra, (dq1_ra, dq2_ra)),
                BatchDuplicated(k_ra, (dk1_ra, dk2_ra)),
                BatchDuplicated(v_ra, (dv1_ra, dv2_ra)),
                BatchDuplicated(bias_ra, (db1_ra, db2_ra)),
            )
            hvp_ref1 = @jit Enzyme.autodiff(
                Forward,
                Const(grad_fn_ref),
                Duplicated,
                Duplicated(q_ra, dq1_ra),
                Duplicated(k_ra, dk1_ra),
                Duplicated(v_ra, dv1_ra),
                Duplicated(bias_ra, db1_ra),
            )
            hvp_ref2 = @jit Enzyme.autodiff(
                Forward,
                Const(grad_fn_ref),
                Duplicated,
                Duplicated(q_ra, dq2_ra),
                Duplicated(k_ra, dk2_ra),
                Duplicated(v_ra, dv2_ra),
                Duplicated(bias_ra, db2_ra),
            )
            for arg_idx in 1:4
                @test Array(hvp_b[1][1][arg_idx]) ≈ Array(hvp_ref1[1][arg_idx]) atol =
                    ad2_atol rtol = ad2_rtol
                @test Array(hvp_b[1][2][arg_idx]) ≈ Array(hvp_ref2[1][arg_idx]) atol =
                    ad2_atol rtol = ad2_rtol
            end

            rev_over_rev(q, k, v, b, wq, wk, wv, wb) = begin
                gq, gk, gv, gb = Enzyme.gradient(Reverse, sdpa_loss, q, k, v, b)
                sum(gq .* wq) + sum(gk .* wk) + sum(gv .* wv) + sum(gb .* wb)
            end
            rev_over_rev_ref(q, k, v, b, wq, wk, wv, wb) = begin
                gq, gk, gv, gb = Enzyme.gradient(Reverse, sdpa_loss_ref, q, k, v, b)
                sum(gq .* wq) + sum(gk .* wk) + sum(gv .* wv) + sum(gb .* wb)
            end

            ror = @jit Enzyme.gradient(
                Reverse,
                rev_over_rev,
                q_ra,
                k_ra,
                v_ra,
                bias_ra,
                Const(dq1_ra),
                Const(dk1_ra),
                Const(dv1_ra),
                Const(db1_ra),
            )
            ror_ref = @jit Enzyme.gradient(
                Reverse,
                rev_over_rev_ref,
                q_ra,
                k_ra,
                v_ra,
                bias_ra,
                Const(dq1_ra),
                Const(dk1_ra),
                Const(dv1_ra),
                Const(db1_ra),
            )
            for arg_idx in 1:4
                @test Array(ror[arg_idx]) ≈ Array(ror_ref[arg_idx]) atol = ad2_atol rtol =
                    ad2_rtol
            end
        end
    end
end

sum_bcast_fn(f, x) = sum(f.(x))
grad_sum_bcast_fn(f, x) = Enzyme.gradient(Reverse, sum_bcast_fn, Const(f), x)[2]

elu_half(x) = NNlib.elu(x, 0.5f0)
celu_two(x) = NNlib.celu(x, 2.0f0)

delu_one(x) = x > 0 ? one(x) : exp(x)
function dselu(x)
    λ = Float32(1.0507009873554805)
    α = Float32(1.6732632423543772)
    return x > 0 ? λ : λ * α * exp(x)
end
dcelu_one(x) = x > 0 ? one(x) : exp(x)
delu_half(x) = x > 0 ? one(x) : 0.5f0 * exp(x)
dcelu_two(x) = x > 0 ? one(x) : exp(x / 2.0f0)

@testset "elu/selu/celu large positive gradient (#2973)" begin
    x = Float32[-2.0, -1.0, 0.0, 1.0, 89.0, 100.0]
    x_ra = Reactant.to_rarray(x)
    x_cpu = Float32[-2.0, -1.0, 1.0]
    x_cpu_ra = Reactant.to_rarray(x_cpu)
    @testset for (fn, dfn, opname) in (
        (NNlib.elu, delu_one, "enzymexla.math.elu"),
        (NNlib.selu, dselu, "enzymexla.math.selu"),
        (NNlib.celu, dcelu_one, "enzymexla.math.celu"),
        (elu_half, delu_half, "enzymexla.math.elu"),
        (celu_two, dcelu_two, "enzymexla.math.celu"),
    )
        hlo = repr(@code_hlo optimize = false fn.(x_ra))
        @test occursin(opname, hlo)

        y_ra = @jit fn.(x_ra)
        g_ra = @jit grad_sum_bcast_fn(fn, x_ra)
        @test Array(y_ra) ≈ fn.(x)
        @test !any(isnan, Array(g_ra))
        @test Array(g_ra) ≈ dfn.(x)

        g_cpu_ra = @jit grad_sum_bcast_fn(fn, x_cpu_ra)
        g_cpu = grad_sum_bcast_fn(fn, x_cpu)
        @test Array(g_cpu_ra) ≈ g_cpu
    end

    # Integer inputs are promoted to float, as in NNlib.
    xi_ra = Reactant.to_rarray(Int32[-2, 0, 3])
    @test Array(@jit(NNlib.elu.(xi_ra))) ≈ NNlib.elu.(Int32[-2, 0, 3])
    @test Array(@jit(NNlib.selu.(xi_ra))) ≈ NNlib.selu.(Int32[-2, 0, 3])
    @test Array(@jit(NNlib.celu.(xi_ra))) ≈ NNlib.celu.(Int32[-2, 0, 3])
end
