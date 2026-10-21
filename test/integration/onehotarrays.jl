using Reactant, Test, OneHotArrays, Random

@testset "OneHotArrays" begin
    m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
    r_m = Reactant.to_rarray(m)
    a = Reactant.TestUtils.construct_test_array(Float32, 100, 4)
    r_a = Reactant.to_rarray(a)
    # At the default precision a TPU does this Float32 matmul in bfloat16, and
    # the result is off by ~1e-3 relative; the test is about the one-hot
    # lowering, not the accelerator's default precision.
    r_res = Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGHEST) do
        @jit r_a * r_m
    end
    res = a * m
    @test convert(Array, r_res) ≈ res
end

@testset "broadcasting" begin
    m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
    r_m = Reactant.to_rarray(m)
    x = Reactant.TestUtils.construct_test_array(Float32, 4, 5)
    x_ra = Reactant.to_rarray(x)

    @testset "addition" begin
        res_ra = @jit r_m .+ x_ra
        res = m .+ x
        @test res_ra ≈ res

        @test Array(r_m) isa Matrix{Bool}
    end

    @testset "multiplication" begin
        # Broadcasting a multiplication has special passes
        res_ra = @jit r_m .* x_ra
        res = m .* x
        @test res_ra ≈ res
    end
end

@testset "onehotbatch/onecold" begin
    x = Int32[10, 20, 30, 10, 10]
    x_ra = Reactant.to_rarray(x)
    labels = Int32(10):Int32(10):Int32(40)
    res_ra = @jit onehotbatch(x_ra, labels)
    res = onehotbatch(x, labels)
    @test Array(res_ra) ≈ res

    x = rand(10:10:40, 2, 3, 5)
    x_ra = Reactant.to_rarray(x)
    labels = reshape([10, 20, 30, 40], 2, 2)
    res = onehotbatch(x, labels)
    res_ra = @jit onehotbatch(x_ra, labels)
    @test Array(res_ra) ≈ res

    x = Int32[1, 2, 3, 1, 1]
    x_ra = Reactant.to_rarray(x)
    labels = Int32(1):Int32(4)
    res_ra = @jit onehotbatch(x_ra, labels)
    res = onehotbatch(x, labels)
    @test Array(res_ra) ≈ res

    vec_ra = Reactant.to_rarray(Float32[0.3, 0.2, 0.5])
    @test @jit(onecold(vec_ra)) == 3

    dense_ra = Reactant.to_rarray(Array(res))
    oc_res = onecold(res)
    @test @jit(onecold(dense_ra)) == oc_res
end

flat(x) = reshape(x, size(x, 1), :)
flat_mul(W, x) = W * flat(x)
flat_sum(x) = sum(flat(x))
flat_sum_dims(x) = sum(flat(x); dims=1)
flat_vec_add(x, y) = vec(flat(x)) .+ vec(y)
flat_bcast_mul(x, y) = flat(x) .* y

@testset "matmul with a reshaped $(length(dims) + 1)-d array" for dims in ((15,), (3, 5))
    m = onehotbatch(rand(1:10, dims...), 1:10)
    r_m = Reactant.to_rarray(m)
    W = Reactant.TestUtils.construct_test_array(Float32, 4, 10)
    r_W = Reactant.to_rarray(W)
    r_res = Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGHEST) do
        @jit flat_mul(r_W, r_m)
    end
    @test r_res ≈ flat_mul(W, m)
end

@testset "reshaped $(length(dims) + 1)-d array" for dims in ((15,), (3, 5))
    m = onehotbatch(rand(1:10, dims...), 1:10)
    r_m = Reactant.to_rarray(m)
    y = Reactant.TestUtils.construct_test_array(Float32, 10, 15)
    r_y = Reactant.to_rarray(y)

    @test @jit(sum(r_m)) == sum(m)
    @test @jit(flat_sum(r_m)) == flat_sum(m)
    @test @jit(flat_sum_dims(r_m)) == flat_sum_dims(m)
    @test @jit(flat_vec_add(r_m, r_y)) ≈ flat_vec_add(m, y)
    @test @jit(flat_bcast_mul(r_m, r_y)) ≈ flat_bcast_mul(m, y)
end
