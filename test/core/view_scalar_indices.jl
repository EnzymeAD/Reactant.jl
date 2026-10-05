using Reactant, Test

@testset "Scalar indices of one-dimensional traced views" begin
    x = Float64[0.2, 0.4, 0.6, 0.8]
    x_ra = Reactant.to_rarray(x)
    for select in (
        x -> view(x, 2:4),
        x -> view(view(x, 2:4), 1:2),
        x -> SubArray(view(x, 2:4), (1:2,)),
        x -> view(x, 1:2:4),
    )
        for index in (1, 2, CartesianIndex(1), CartesianIndex(2))
            f = x -> (@allowscalar select(x)[index])
            @test (@jit f(x_ra)) ≈ f(x)
        end
    end

    x = reshape(Float64[0.2, 0.4, 0.6, 0.8, 1.0, 1.2], 2, 3)
    x_ra = Reactant.to_rarray(x)
    for select in (x -> view(x, :, 2), x -> view(x, 2, :))
        for index in (1, 2, CartesianIndex(1), CartesianIndex(2))
            f = x -> (@allowscalar select(x)[index])
            @test (@jit f(x_ra)) ≈ f(x)
        end
    end

    f_dynamic(x, i) = @allowscalar view(x, 2:4)[i]
    i_ra = Reactant.to_rarray(2; track_numbers=true)
    x_vector = Float64[0.2, 0.4, 0.6, 0.8]
    x_vector_ra = Reactant.to_rarray(x_vector)
    compiled = @compile f_dynamic(x_vector_ra, i_ra)
    @test compiled(x_vector_ra, i_ra) ≈ f_dynamic(x_vector, 2)
    @test compiled(x_vector_ra, Reactant.to_rarray(1; track_numbers=true)) ≈
        f_dynamic(x_vector, 1)

    # Preserve the separate array-indexing path and result shape.
    f(x) = view(x, :, 2)[[2, 1]]
    actual = @jit f(x_ra)
    @test actual ≈ f(x)
    @test size(actual) == (2,)
end
