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

flat_mul(W, x) = W * reshape(x, size(x, 1), :)

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

@testset "show/print_array without scalar indexing" begin
    ext = Base.get_extension(Reactant, :ReactantOneHotArraysExt)
    oh_cpu = onehotbatch([1, 3, 2, 1], 1:3)
    oh_r = Reactant.to_rarray(oh_cpu)

    @test which(Base.print_array, Tuple{IOBuffer,typeof(oh_r)}).module === ext
    @test which(Base.show, Tuple{IOBuffer,typeof(oh_r)}).module === ext

    task_local_storage(:ScalarIndexing, Reactant.GPUArraysCore.ScalarDisallowed) do
        @test occursin("⋅", sprint(show, MIME"text/plain"(), oh_r))
        @test sprint(show, oh_r) == sprint(show, oh_cpu)
        @test sprint(show, MIME"text/plain"(), oh_r') isa String
        @test sprint(show, MIME"text/plain"(), transpose(oh_r)) isa String
        @test sprint(show, MIME"text/plain"(), reshape(oh_r, 3, 2, 2)) isa String

        oh_view = OneHotArray(@view(oh_r.indices[1:2]), 3)
        @test Array(oh_view) == Array(oh_cpu)[:, 1:2]
        @test sprint(show, MIME"text/plain"(), oh_view) isa String
    end
end
