using Reactant, Test, OneHotArrays, Random

@testset "OneHotArrays" begin
    m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
    r_m = Reactant.to_rarray(m)
    a = Reactant.TestUtils.construct_test_array(Float32, 100, 4)
    r_a = Reactant.to_rarray(a)
    r_res = @jit r_a * r_m
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
