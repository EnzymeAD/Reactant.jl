using Reactant, Test, OneHotArrays, Random

@testset "OneHotArrays" begin
    m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
    r_m = Reactant.to_rarray(m)
    a = rand(100, 4)
    r_a = Reactant.to_rarray(a)
    r_res = @jit r_a * r_m
    res = a * m
    @test convert(Array, r_res) ≈ res
end

@testset "broadcasting" begin
    m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
    r_m = Reactant.to_rarray(m)
    x = rand(Float32, 4, 5)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit r_m .+ x_ra
    res = m .+ x
    @test res_ra ≈ res
end
