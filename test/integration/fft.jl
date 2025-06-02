using FFTW, Reactant, Test

@testset "fft" begin
    x = rand(ComplexF32, 2, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test_throws AssertionError @jit(fft(x_ra)) # TODO: support this

    x = rand(ComplexF32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(fft(x_ra)) ≈ fft(x)
    @test @jit(fft(x_ra, (1, 2))) ≈ fft(x, (1, 2))
    @test @jit(fft(x_ra, (1, 2, 3))) ≈ fft(x, (1, 2, 3))
    @test @jit(fft(x_ra, (2, 3))) ≈ fft(x, (2, 3))
    @test @jit(fft(x_ra, (1, 3))) ≈ fft(x, (1, 3))

    @test @jit(fft(x_ra, (3, 2))) ≈ fft(x, (3, 2))
    @test_throws AssertionError @jit(fft(x_ra, (1, 4)))

    y_ra = @jit(fft(x_ra))
    @test @jit(ifft(y_ra)) ≈ x

    @testset "fft real input" begin
        x = rand(Float32, 2, 3, 4)
        x_ra = Reactant.to_rarray(x)

        @test @jit(fft(x_ra)) ≈ fft(x)
        @test @jit(fft(x_ra, (1, 2))) ≈ fft(x, (1, 2))
        @test @jit(fft(x_ra, (1, 2, 3))) ≈ fft(x, (1, 2, 3))
    end
end

@testset "rfft" begin
    x = rand(2, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test_throws AssertionError @jit(rfft(x_ra)) # TODO: support this

    x = rand(2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(rfft(x_ra)) ≈ rfft(x)
    @test @jit(rfft(x_ra, (1, 2))) ≈ rfft(x, (1, 2))
    @test @jit(rfft(x_ra, (1, 2, 3))) ≈ rfft(x, (1, 2, 3))
    @test @jit(rfft(x_ra, (2, 3))) ≈ rfft(x, (2, 3))
    @test @jit(rfft(x_ra, (1, 3))) ≈ rfft(x, (1, 3))

    @test @jit(rfft(x_ra, (3, 2))) ≈ rfft(x, (3, 2))
    @test_throws AssertionError @jit(rfft(x_ra, (1, 4)))

    y_ra = @jit(rfft(x_ra))
    @test @jit(irfft(y_ra, 2)) ≈ x
    @test @jit(irfft(y_ra, 3)) ≈ irfft(rfft(x), 3)

    @testset "irfft real input" begin
        y_ra_real = @jit(real(y_ra))
        y_real = Array(y_ra_real)

        @test @jit(rfft(x_ra)) ≈ rfft(x)
        @test @jit(rfft(x_ra, (1, 2))) ≈ rfft(x, (1, 2))
        @test @jit(rfft(x_ra, (1, 2, 3))) ≈ rfft(x, (1, 2, 3))
    end
end
