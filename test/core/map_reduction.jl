# Tests for reduction and mapreduce operations
using Reactant, Test, Enzyme, Statistics

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))

@testset "2D sum" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)

    r_res = sum(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar sum(a)
    @test c_res ≈ r_res

    @test @jit(sum(a)) ≈ r_res
end

@testset "Basic reduce max" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)

    r_res = fastmax(x)

    a = Reactant.to_rarray(x)

    c_res = @allowscalar fastmax(a)
    @test c_res ≈ r_res

    @test @jit(fastmax(a)) ≈ r_res
end

@testset "Empty reduce (#2226)" begin
    empty_prod() = prod(Int[])
    @test empty_prod() == @jit(empty_prod())

    empty_sum() = sum(Int[])
    @test empty_sum() == @jit(empty_sum())
end

sumexp(x) = sum(exp, x)

sum_compare(x) = sum(x) > 0

@testset "Basic mapreduce" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 10)
    a = Reactant.to_rarray(x)
    r_res = sumexp(x)

    f_res = @jit sumexp(a)

    @test f_res ≈ r_res

    # Ensure we are tracing as scalars. Else this will fail due to > not being defined on
    # arrays
    @test @jit(sum_compare(a)) == sum_compare(x)
end

function mysoftmax!(x)
    max_ = fastmax(x)
    return x .- max_
end

@testset "Basic softmax" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)
    r_res = mysoftmax!(x)

    a = Reactant.to_rarray(x)

    f_res = @jit mysoftmax!(a)
    @test f_res ≈ r_res
end

sum_xxᵀ(x) = sum(x .* x')

@testset "sum(x .* x')" begin
    @testset "size(x): $(size(x))" for x in (
        Reactant.TestUtils.construct_test_array(Float64, 4, 4),
        Reactant.TestUtils.construct_test_array(Float64, 4),
    )
        x_ca = Reactant.to_rarray(x)

        @test @jit(sum_xxᵀ(x_ca)) ≈ sum_xxᵀ(x)
    end
end

@testset "reduce integers" begin
    x = [isodd(i) for i in 1:100]
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra)) == sum(x)

    x = collect(Int16, 1:100)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra)) == sum(x)
end

function fntest1(x)
    y = similar(x, 1, 1, 8)
    sum!(y, x)
    return y
end

function fntest2(x)
    y = similar(x, 2, 1, 8)
    sum!(y, x)
    return y
end

function fntest3(x)
    y = similar(x, 2, 1, 1)
    sum!(abs2, y, x)
    return y
end

@testset "mapreducedim!" begin
    x = reshape(collect(Float32, 1:64), 2, 4, 8) ./ 64
    x_ra = Reactant.to_rarray(x)

    @test Array(@jit(fntest1(x_ra))) ≈ fntest1(x)
    @test Array(@jit(fntest2(x_ra))) ≈ fntest2(x)
    @test Array(@jit(fntest3(x_ra))) ≈ fntest3(x)
end

@testset "mapreduce with unitrange dims" begin
    x = reshape(collect(Float32, 1:64), 2, 4, 8)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sum(x_ra; dims=1:2)) ≈ sum(x; dims=1:2)
end

accum_fn(x, y) = abs2(x) + abs2(y)

@testset "accumulate" begin
    a = collect(Float32, 1:10) ./ 10
    a_ra = Reactant.to_rarray(a)

    b = reshape(collect(Float32, 1:60), (3, 4, 5)) ./ 60
    b_ra = Reactant.to_rarray(b)

    @testset "cumsum" begin
        @test @jit(cumsum(a_ra)) ≈ cumsum(a)

        @test @jit(cumsum(b_ra; dims=1)) ≈ cumsum(b; dims=1)
        @test @jit(cumsum(b_ra; dims=2)) ≈ cumsum(b; dims=2)
        @test @jit(cumsum(b_ra; dims=3)) ≈ cumsum(b; dims=3)

        @test begin
            z = similar(a_ra)
            @jit(cumsum!(z, a_ra))
            z
        end ≈ cumsum(a)

        @test begin
            z = similar(b_ra)
            @jit(cumsum!(z, b_ra; dims=1))
            z
        end ≈ cumsum(b; dims=1)
        @test begin
            z = similar(b_ra)
            @jit(cumsum!(z, b_ra; dims=2))
            z
        end ≈ cumsum(b; dims=2)
        @test begin
            z = similar(b_ra)
            @jit(cumsum!(z, b_ra; dims=3))
            z
        end ≈ cumsum(b; dims=3)
    end

    @testset "cumprod" begin
        @test @jit(cumprod(a_ra)) ≈ cumprod(a)

        @test @jit(cumprod(b_ra; dims=1)) ≈ cumprod(b; dims=1)
        @test @jit(cumprod(b_ra; dims=2)) ≈ cumprod(b; dims=2)
        @test @jit(cumprod(b_ra; dims=3)) ≈ cumprod(b; dims=3)

        @test begin
            z = similar(a_ra)
            @jit(cumprod!(z, a_ra))
            z
        end ≈ cumprod(a)
        @test begin
            z = similar(b_ra)
            @jit(cumprod!(z, b_ra; dims=1))
            z
        end ≈ cumprod(b; dims=1)
        @test begin
            z = similar(b_ra)
            @jit(cumprod!(z, b_ra; dims=2))
            z
        end ≈ cumprod(b; dims=2)
        @test begin
            z = similar(b_ra)
            @jit(cumprod!(z, b_ra; dims=3))
            z
        end ≈ cumprod(b; dims=3)
    end

    @testset "accumulate" begin
        @test @jit(accumulate(accum_fn, a_ra; init=0.0f0)) ≈
            accumulate(accum_fn, a; init=0.0f0) broken = RunningOnTPU

        @test @jit(accumulate(accum_fn, b_ra; init=0.0f0, dims=1)) ≈
            accumulate(accum_fn, b; dims=1, init=0.0f0) broken = RunningOnTPU
        @test @jit(accumulate(accum_fn, b_ra; init=0.0f0, dims=2)) ≈
            accumulate(accum_fn, b; dims=2, init=0.0f0) broken = RunningOnTPU
        @test @jit(accumulate(accum_fn, b_ra; init=0.0f0, dims=3)) ≈
            accumulate(accum_fn, b; dims=3, init=0.0f0) broken = RunningOnTPU

        @test begin
            z = similar(a_ra)
            @jit(accumulate!(accum_fn, z, a_ra; init=0.0f0))
            z
        end ≈ accumulate(accum_fn, a; init=0.0f0) broken = RunningOnTPU

        @test begin
            z = similar(b_ra)
            @jit(accumulate!(accum_fn, z, b_ra; init=0.0f0, dims=1))
            z
        end ≈ accumulate(accum_fn, b; dims=1, init=0.0f0) broken = RunningOnTPU
        @test begin
            z = similar(b_ra)
            @jit(accumulate!(accum_fn, z, b_ra; init=0.0f0, dims=2))
            z
        end ≈ accumulate(accum_fn, b; dims=2, init=0.0f0) broken = RunningOnTPU
        @test begin
            z = similar(b_ra)
            @jit(accumulate!(accum_fn, z, b_ra; init=0.0f0, dims=3))
            z
        end ≈ accumulate(accum_fn, b; dims=3, init=0.0f0) broken = RunningOnTPU
    end
end

sameunitrange(x, y) = first(x) == first(y) && last(x) == last(y)

@testset "searchsorted" begin
    x = [1, 2, 4, 5, 5, 7]
    x_ra = Reactant.to_rarray(x)

    @testset "searchsortedfirst" begin
        @testset for val in (4, 5, 3, 9, 0)
            @test @jit(searchsortedfirst(x_ra, val)) == searchsortedfirst(x, val)
            @test @jit(searchsortedfirst(x_ra, ConcreteRNumber(val))) ==
                searchsortedfirst(x, val)
        end
    end

    @testset "searchsortedlast" begin
        @testset for val in (4, 5, 3, 9, 0)
            @test @jit(searchsortedlast(x_ra, val)) == searchsortedlast(x, val)
            @test @jit(searchsortedlast(x_ra, ConcreteRNumber(val))) ==
                searchsortedlast(x, val)
        end
    end

    @testset "searchsorted" begin
        @testset for val in (4, 5, 3, 9, 0)
            @test sameunitrange(@jit(searchsorted(x_ra, val)), searchsorted(x, val))
            @test sameunitrange(
                @jit(searchsorted(x_ra, ConcreteRNumber(val))), searchsorted(x, val)
            )
        end
    end
end

function mapreduce_with_closure(a, A)
    return sum(A) do Ax
        return log(a + Ax)
    end
end

@testset "mapreduce with closure" begin
    ρr = ConcreteRNumber(2.0)
    x = Reactant.TestUtils.construct_test_array(Float64, 5, 5)

    hlo = repr(@code_hlo mapreduce_with_closure(ρr, x))
    @test contains(hlo, "stablehlo.reduce")

    @test @jit(mapreduce_with_closure(ρr, x)) ≈ mapreduce_with_closure(2.0, x)
end

mapreduce_closure_not_traced(x) = prod(Base.Fix1(size, x), [1, 3])

@testset "mapreduce closure not traced" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 5, 32, 7)
    x_ra = Reactant.to_rarray(x)

    @test @jit(mapreduce_closure_not_traced(x_ra)) == prod(size(x)[[1, 3]])
end

@testset "scalar indexing in any #1434" begin
    xr = Reactant.to_rarray(ones(4, 4))
    @test @jit(any(<(0), xr)) == any(<(0), Array(xr))
end

zip_iterator(a, b) = mapreduce(splat(*), +, zip(a, b))
zip_iterator2(a, b) = mapreduce(splat(.-), +, zip(a, b))
enumerate_iterator(a) = mapreduce(splat(*), +, enumerate(a))
enumerate_iterator2(a) = mapreduce(splat(.-), +, enumerate(a))

function nested_mapreduce_zip(x, y)
    return mapreduce(+, zip(eachcol(x), eachcol(y)); init=0.0f0) do (x, y)
        return sum(abs2, x) + sum(abs2, y)
    end
end

function nested_mapreduce_hcat(x, y)
    return mapreduce(
        hcat, zip(eachcol(x), eachcol(y)); init=similar(x, size(x, 1), 0)
    ) do (x, y)
        return x .+ y
    end
end

function f_generator(points, params)
    return sum(params * point for point in points)
end

@testset "Base.Iterators" begin
    @testset "zip" begin
        N = 10
        a = collect(range(1.0, 5.0; length=N))
        x = collect(range(10.0, 15.0; length=N + 2))
        x_ra = Reactant.to_rarray(x)

        @test @jit(zip_iterator(a, x_ra)) ≈ zip_iterator(a, x)

        a = [Reactant.TestUtils.construct_test_array(Float32, 2, 3) for _ in 1:10]
        x = [Reactant.TestUtils.construct_test_array(Float32, 2, 3) for _ in 1:10]
        a_ra = Reactant.to_rarray(a)
        x_ra = Reactant.to_rarray(x)

        @test @jit(zip_iterator2(a_ra, x_ra)) ≈ zip_iterator2(a, x)
    end

    @testset "enumerate" begin
        x = collect(range(1.0, 5.0; length=10))
        x_ra = Reactant.to_rarray(x)

        @test @jit(enumerate_iterator(x_ra)) ≈ enumerate_iterator(x)

        x = [Reactant.TestUtils.construct_test_array(Float32, 2, 3) for _ in 1:10]
        x_ra = Reactant.to_rarray(x)

        @test @jit(enumerate_iterator2(x_ra)) ≈ enumerate_iterator2(x)
    end

    @testset "nested mapreduce" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)
        @test @jit(nested_mapreduce_zip(x_ra, y_ra)) ≈ nested_mapreduce_zip(x, y)
    end

    @testset "nested mapreduce hcat" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 3)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(nested_mapreduce_hcat(x_ra, y_ra)) ≈ nested_mapreduce_hcat(x, y)
    end
end

@testset "Base.Generator" begin
    points = eachcol(Reactant.TestUtils.construct_test_array(Float32, 2, 6))
    params = Reactant.TestUtils.construct_test_array(Float32, 4, 2)
    points_ra = Reactant.to_rarray(points)
    params_ra = Reactant.to_rarray(params)

    @test @jit(f_generator(points_ra, params_ra)) ≈ f_generator(points, params)
end

@testset "Slices" begin
    @testset "drop=true" begin
        x = eachslice(
            Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5); dims=(3, 1)
        )
        x_ra = Reactant.to_rarray(x)

        @test @jit(sum(x_ra)) ≈ sum(x)

        @testset for dims in (1, 2, (1, 2), (2, 1))
            res_ra = @jit sum(x_ra; dims)
            res = sum(x; dims)
            @test size(res_ra) == size(res)
            for (gt, comp) in zip(res_ra, res)
                @test gt ≈ comp
            end
        end
    end

    @testset "drop=false" begin
        x = eachslice(
            Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4, 5);
            dims=(3, 1),
            drop=false,
        )
        x_ra = Reactant.to_rarray(x)

        @test @jit(sum(x_ra)) ≈ sum(x)

        @testset for dims in (1, 2, 3, 4, (1, 2), (1, 2, 4), (3, 4, 1), (2, 1))
            res_ra = @jit sum(x_ra; dims)
            res = sum(x; dims)
            @test size(res_ra) == size(res)
            for (gt, comp) in zip(res_ra, res)
                @test gt ≈ comp
            end
        end
    end
end

mapped_sub(xs...) = stack(map(-, xs...))

@testset "map of slices" begin
    @testset "Vector of Slices" begin
        x_full = Reactant.TestUtils.construct_test_array(Float32, 10, 5, 3)
        y_full = Reactant.TestUtils.construct_test_array(Float32, 10, 5, 3)
        x = [view(x_full, :, i, :) for i in 1:size(x_full, 2)]
        y = [view(y_full, :, i, :) for i in 1:size(y_full, 2)]
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(mapped_sub(x_ra, y_ra)) ≈ mapped_sub(x, y) atol = 1e-5 rtol = 1e-5
    end

    @testset "Slices" begin
        x_full = Reactant.TestUtils.construct_test_array(Float32, 10, 5)

        @testset "ColumnSlices" begin
            x_sliced = eachcol(x_full)
            x_ra = Reactant.to_rarray(x_sliced)

            @test @jit(mapped_sub(x_ra)) ≈ mapped_sub(x_sliced) atol = 1e-5 rtol = 1e-5
        end
    end
end
