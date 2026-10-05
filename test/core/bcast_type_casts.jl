using Reactant, Test

@testset "Fused primitive conversion broadcasts" begin
    for T in (Float64, Float32, Int32)
        x = T <: Integer ? T[1, 2, 3, 4] : T[0.2, 0.4, 0.6, 0.8]
        x_ra = Reactant.to_rarray(x)
        for select in (x -> x[1:3], x -> view(x, 1:3))
            for f in (
                x -> sum(Float64.(select(x))),
                x -> sum(Float64.(select(x)) .+ 1.0),
                x -> sum(Float64.(select(x)) .+ log.(1:3)),
                x -> sin.(Float64.(select(x)) .+ 1.0),
                x -> Float64.(Float32.(select(x))) .+ 1.0,
            )
                expected = f(x)
                actual = @jit f(x_ra)
                @test actual ≈ expected
                if expected isa AbstractArray
                    @test size(actual) == size(expected)
                    @test eltype(actual) == eltype(expected)
                end
            end
        end
        @test Array(x_ra) == x
    end

    x = reshape(Float64[0.2, 0.4, 0.6, 0.8, 1.0, 1.2], 2, 3)
    x_ra = Reactant.to_rarray(x)
    for select in (x -> x[:, 1:2], x -> view(x, :, 1:2))
        f = x -> Float32.(select(x)) .+ reshape(Float32[1, 2], 2, 1)
        actual = @jit f(x_ra)
        @test actual ≈ f(x)
        @test size(actual) == (2, 2)
        @test eltype(actual) == Float32
    end

    x = Float64[0.2, 0.4, 0.6, 0.8]
    x_ra = Reactant.to_rarray(x)
    for select in (x -> x[1:0], x -> view(x, 1:0))
        for f in (
            x -> Float64.(select(x)) .+ 1.0,
            x -> Float32.(select(x)) .+ log.(1:0),
            x -> sin.(Float64.(Float32.(select(x))) .+ 1.0),
        )
            # Empty outputs expose a separate tensor.empty XLA-export limitation.
            # Check trace shape/type here, and executable empty-reduction parity.
            hlo = string(@code_hlo optimize=false f(x_ra))
            @test occursin("tensor<0xf$(8 * sizeof(eltype(f(x))))>", hlo)
            g = x -> sum(f(x))
            @test (@jit g(x_ra)) ≈ g(x)
        end
    end

    x0 = reshape(Float32[0.2], ())
    x0_ra = Reactant.to_rarray(x0)
    f0(x) = Float64.(x) .+ 1.0
    actual0 = @jit f0(x0_ra)
    # Reactant intentionally unwraps a zero-dimensional broadcast result.
    @test actual0 isa Reactant.ConcreteRNumber
    @test actual0 ≈ f0(x0)[]

    # Check both narrowing and widening conversions survive in the traced graph.
    f(x) = Float64.(Float32.(view(x, 1:3))) .+ 1.0
    hlo = string(@code_hlo optimize = false f(x_ra))
    @test count("stablehlo.convert", hlo) >= 2
    @test occursin("tensor<3xf32>", hlo)
    @test occursin("tensor<3xf64>", hlo)
    # A separate optimizer bug also reproduces with staged standalone casts
    # on unmodified Reactant 0.2.289: it incorrectly elides f64 -> f32 -> f64.
    # Keep this exact dependency boundary visible and self-firing once fixed.
    @test_broken Array(@jit f(x_ra)) == f(x)

    narrow = x -> Float32.(view(x, 1:3))
    actual32 = @jit narrow(x_ra)
    @test Array(actual32) == narrow(x)
    @test eltype(actual32) == Float32
    x32 = Float32.(x)
    x32_ra = Reactant.to_rarray(x32)
    widen = x -> Float64.(view(x, 1:3))
    actual64 = @jit widen(x32_ra)
    @test Array(actual64) == widen(x32)
    @test eltype(actual64) == Float64
end
