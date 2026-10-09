using Reactant, Test, VectorInterface
using VectorInterface: One

add_one(y, x, α) = add.(y, x, α, One())
add_scalars(y, x, α, β) = add.(y, x, α, β)
scale_scalars(x, α) = scale.(x, α)
add_arrays(y, x, α, β) = add(y, x, α, β)
add_arrays!(y, x, α, β) = add!(y, x, α, β)
scale_array(x, α) = scale(x, α)

same(a, b) = isapprox(a, b; nans=true)

@testset "strong zeros: α=$α β=$β traced coefficients=$traced" for α in (0.0, 0.7),
    β in (0.0, 0.3),
    traced in (true, false)

    x = [1.5, NaN, -Inf, 2.0]
    y = [0.5, -1.0, NaN, Inf]
    x_ra, y_ra = Reactant.to_rarray(x), Reactant.to_rarray(y)
    coeff(c) = traced ? Reactant.to_rarray(c; track_numbers=Number) : c
    α_ra, β_ra = coeff(α), coeff(β)

    @test same(Array(@jit(add_one(y_ra, x_ra, α_ra))), add_one(y, x, α))
    @test same(Array(@jit(add_scalars(y_ra, x_ra, α_ra, β_ra))), add_scalars(y, x, α, β))
    @test same(Array(@jit(scale_scalars(x_ra, α_ra))), scale_scalars(x, α))
    @test same(Array(@jit(add_arrays(y_ra, x_ra, α_ra, β_ra))), add_arrays(y, x, α, β))
    @test same(
        Array(@jit(add_arrays!(copy(y_ra), x_ra, α_ra, β_ra))), add_arrays!(copy(y), x, α, β)
    )
    @test same(Array(@jit(scale_array(x_ra, α_ra))), scale_array(x, α))
end
