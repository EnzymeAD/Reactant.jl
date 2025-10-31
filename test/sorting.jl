using Reactant, Test

@testset "sort & sortperm" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)

    srt_rev(x) = sort(x; rev=true)
    srtperm_rev(x) = sortperm(x; rev=true)
    srt_by(x) = sort(x; by=abs2)
    srtperm_by(x) = sortperm(x; by=abs2)
    srt_lt(x) = sort(x; lt=(a, b) -> a > b)
    srtperm_lt(x) = sortperm(x; lt=(a, b) -> a > b)

    @test @jit(sort(x_ra)) ≈ sort(x)
    @test @jit(srt_rev(x_ra)) ≈ srt_rev(x)
    @test @jit(srt_lt(x_ra)) ≈ srt_lt(x)
    @test @jit(srt_by(x_ra)) ≈ srt_by(x)
    @test @jit(sortperm(x_ra)) ≈ sortperm(x)
    @test @jit(srtperm_rev(x_ra)) ≈ srtperm_rev(x)
    @test @jit(srtperm_lt(x_ra)) ≈ srtperm_lt(x)
    @test @jit(srtperm_by(x_ra)) ≈ srtperm_by(x)

    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)
    @jit sort!(x_ra)
    @test x_ra ≈ sort(x)

    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)
    ix = similar(x_ra, Int)
    @jit sortperm!(ix, x_ra)
    @test ix ≈ sortperm(x)

    x = Reactant.TestUtils.construct_test_array(Float64, 10, 4, 3)
    x_ra = Reactant.to_rarray(x)

    srt(x, d) = sort(x; dims=d)
    srt_rev(x, d) = sort(x; dims=d, rev=true)
    srt_by(x, d) = sort(x; dims=d, by=abs2)
    srt_lt(x, d) = sort(x; dims=d, lt=(a, b) -> a > b)
    srtperm(x, d) = sortperm(x; dims=d)
    srtperm_rev(x, d) = sortperm(x; dims=d, rev=true)
    srtperm_by(x, d) = sortperm(x; dims=d, by=abs2)
    srtperm_lt(x, d) = sortperm(x; dims=d, lt=(a, b) -> a > b)

    @testset for d in 1:ndims(x)
        @test @jit(srt(x_ra, d)) ≈ srt(x, d)
        @test @jit(srtperm(x_ra, d)) ≈ srtperm(x, d)
        @test @jit(srt_rev(x_ra, d)) ≈ srt_rev(x, d)
        @test @jit(srtperm_rev(x_ra, d)) ≈ srtperm_rev(x, d)
        @test @jit(srt_by(x_ra, d)) ≈ srt_by(x, d)
        @test @jit(srtperm_by(x_ra, d)) ≈ srtperm_by(x, d)
        @test @jit(srt_lt(x_ra, d)) ≈ srt_lt(x, d)
        @test @jit(srtperm_lt(x_ra, d)) ≈ srtperm_lt(x, d)
    end
end

@testset "partialsort & partialsortperm" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(partialsort(x_ra, 1:5)) ≈ partialsort(x, 1:5)
    @test @jit(partialsort(x_ra, 1:5; rev=true)) ≈ partialsort(x, 1:5; rev=true)
    @test @jit(partialsortperm(x_ra, 1:5)) ≈ partialsortperm(x, 1:5)
    @test @jit(partialsortperm(x_ra, 1:5; rev=true)) ≈ partialsortperm(x, 1:5; rev=true)
    @test @jit(partialsort(x_ra, 3:6)) ≈ partialsort(x, 3:6)
    @test @jit(partialsort(x_ra, 3:6; rev=true)) ≈ partialsort(x, 3:6; rev=true)
    @test @jit(partialsortperm(x_ra, 3:6)) ≈ partialsortperm(x, 3:6)
    @test @jit(partialsortperm(x_ra, 3:6; rev=true)) ≈ partialsortperm(x, 3:6; rev=true)
    @test @jit(partialsort(x_ra, 4)) ≈ partialsort(x, 4)
    @test @jit(partialsort(x_ra, 4; rev=true)) ≈ partialsort(x, 4; rev=true)
    @test @jit(partialsortperm(x_ra, 4)) ≈ partialsortperm(x, 4)
    @test @jit(partialsortperm(x_ra, 4; rev=true)) ≈ partialsortperm(x, 4; rev=true)

    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)
    @jit partialsort!(x_ra, 1:5)
    partialsort!(x, 1:5)
    @test Array(x_ra)[1:5] ≈ x[1:5]

    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)
    @jit partialsort!(x_ra, 3:5; rev=true)
    partialsort!(x, 3:5; rev=true)
    @test Array(x_ra)[3:5] ≈ x[3:5]

    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)
    @jit partialsort!(x_ra, 3)
    partialsort!(x, 3)
    @test @allowscalar(x_ra[3]) ≈ x[3]

    x = Reactant.TestUtils.construct_test_array(Float64, 10)
    x_ra = Reactant.to_rarray(x)

    ix = similar(x, Int)
    ix_ra = Reactant.to_rarray(ix)
    @jit partialsortperm!(ix_ra, x_ra, 1:5)
    partialsortperm!(ix, x, 1:5)
    @test Array(ix_ra)[1:5] ≈ ix[1:5]

    ix = similar(x, Int)
    ix_ra = Reactant.to_rarray(ix)
    @jit partialsortperm!(ix_ra, x_ra, 3)
    partialsortperm!(ix, x, 3)
    @test @allowscalar(ix_ra[3]) ≈ ix[3]
end

@testset "argmin / argmax" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    x_ra = Reactant.to_rarray(x)

    linargmin(x) = LinearIndices(x)[argmin(x)]
    linargmax(x) = LinearIndices(x)[argmax(x)]

    @test linargmin(x) == @jit(argmin(x_ra))
    @test linargmax(x) == @jit(argmax(x_ra))

    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    linargmin(x, d) = LinearIndices(x)[argmin(x; dims=d)]
    linargmax(x, d) = LinearIndices(x)[argmax(x; dims=d)]
    argmindims(x, d) = argmin(x; dims=d)
    argmaxdims(x, d) = argmax(x; dims=d)

    @test linargmin(x, 1) == @jit(argmindims(x_ra, 1))
    @test linargmax(x, 1) == @jit(argmaxdims(x_ra, 1))
    @test linargmin(x, 2) == @jit(argmindims(x_ra, 2))
    @test linargmax(x, 2) == @jit(argmaxdims(x_ra, 2))
    @test linargmin(x, 3) == @jit(argmindims(x_ra, 3))
    @test linargmax(x, 3) == @jit(argmaxdims(x_ra, 3))

    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test argmin(abs2, x) ≈ @jit(argmin(abs2, x_ra))
    @test argmax(abs2, x) ≈ @jit(argmax(abs2, x_ra))
end

function dual_approx(x, y)
    @test (x[1] ≈ y[1])
    @test (x[2] ≈ y[2])
end

@testset "findmin / findmax" begin
    xvec = Reactant.TestUtils.construct_test_array(Float64, 10)
    xvec_ra = Reactant.to_rarray(xvec)

    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    x_ra = Reactant.to_rarray(x)

    function fwithlinindices(g, f, x; kwargs...)
        values, indices = g(f, x; kwargs...)
        return values, LinearIndices(x)[indices]
    end

    dual_approx(fwithlinindices(findmin, identity, x), @jit(findmin(x_ra)))
    dual_approx(fwithlinindices(findmax, identity, x), @jit(findmax(x_ra)))
    dual_approx(fwithlinindices(findmin, identity, xvec), @jit(findmin(xvec_ra)))
    dual_approx(fwithlinindices(findmax, identity, xvec), @jit(findmax(xvec_ra)))

    fmindims(x, d) = findmin(x; dims=d)
    fmindims(f, x, d) = findmin(f, x; dims=d)
    fmaxdims(x, d) = findmax(x; dims=d)
    fmaxdims(f, x, d) = findmax(f, x; dims=d)

    dual_approx(fwithlinindices(findmin, identity, x; dims=1), @jit(fmindims(x_ra, 1)))
    dual_approx(fwithlinindices(findmax, identity, x; dims=1), @jit(fmaxdims(x_ra, 1)))
    dual_approx(fwithlinindices(findmin, identity, x; dims=2), @jit(fmindims(x_ra, 2)))
    dual_approx(fwithlinindices(findmax, identity, x; dims=2), @jit(fmaxdims(x_ra, 2)))
    dual_approx(fwithlinindices(findmin, abs2, x; dims=1), @jit(fmindims(abs2, x_ra, 1)))
    dual_approx(fwithlinindices(findmax, abs2, x; dims=1), @jit(fmaxdims(abs2, x_ra, 1)))
    dual_approx(fwithlinindices(findmin, abs2, x; dims=2), @jit(fmindims(abs2, x_ra, 2)))
    dual_approx(fwithlinindices(findmax, abs2, x; dims=2), @jit(fmaxdims(abs2, x_ra, 2)))
end

@testset "findfirst / findlast" begin
    x = Bool[
        0 0 0 0
        1 0 1 0
        0 1 0 1
    ]
    x_ra = Reactant.to_rarray(x)

    ffirstlinindices(x) = LinearIndices(x)[findfirst(x)]
    ffirstlinindices(f, x) = LinearIndices(x)[findfirst(f, x)]
    flastlinindices(x) = LinearIndices(x)[findlast(x)]
    flastlinindices(f, x) = LinearIndices(x)[findlast(f, x)]

    @test ffirstlinindices(x) ≈ @jit(findfirst(x_ra))
    @test flastlinindices(x) ≈ @jit(findlast(x_ra))

    x = Int64[
        3 5 7 9
        4 6 7 8
        5 7 8 9
    ]
    x_ra = Reactant.to_rarray(x)

    @test ffirstlinindices(iseven, x) ≈ @jit(findfirst(iseven, x_ra))
    @test flastlinindices(iseven, x) ≈ @jit(findlast(iseven, x_ra))
end

@testset "approx top k lowering" begin
    x = vec(permutedims(reshape(collect(Float32, 1:1000), 2, 5, 10, 10), (4, 2, 3, 1)))
    x_ra = Reactant.to_rarray(x)

    hlo = Reactant.with_config(; lower_partialsort_to_approx_top_k=true) do
        @code_hlo partialsortperm(x_ra, 4:15)
    end

    @test contains(repr(hlo), "ApproxTopK")
    @test contains(repr(hlo), "top_k = 15 : i64")

    hlo = Reactant.with_config(; lower_partialsort_to_approx_top_k=false) do
        @code_hlo partialsortperm(x_ra, 4:15)
    end

    @test !contains(repr(hlo), "ApproxTopK")
    @test contains(repr(hlo), "chlo.top_k")

    idxs = partialsortperm(x, 4:15)
    idxs_ra = Reactant.with_config(; lower_partialsort_to_approx_top_k=false) do
        @jit partialsortperm(x_ra, 4:15)
    end
    @test idxs == idxs_ra

    idxs_ra = Reactant.with_config(; lower_partialsort_to_approx_top_k=true) do
        @jit partialsortperm(x_ra, 4:15)
    end
    @test idxs == idxs_ra

    idxs = partialsortperm(x, 4:15; rev=true)
    idxs_ra = Reactant.with_config(; lower_partialsort_to_approx_top_k=false) do
        @jit partialsortperm(x_ra, 4:15; rev=true)
    end
    @test idxs == idxs_ra

    idxs_ra = Reactant.with_config(; lower_partialsort_to_approx_top_k=true) do
        @jit partialsortperm(x_ra, 4:15; rev=true)
    end
    @test idxs == idxs_ra
end
