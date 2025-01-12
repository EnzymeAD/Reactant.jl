using FFTW, Reactant

@testset "fft" begin
    x = rand(ComplexF32, 2, 2, 3, 4)
    x_ra = Reactant.ConcreteRArray(x)

    @test_throws AssertionError @jit(fft(x_ra))

    x = rand(ComplexF32, 2, 3, 4)
    x_ra = Reactant.ConcreteRArray(x)

    @test @jit(fft(x_ra)) ≈ fft(x)
    @test @jit(fft(x_ra, (1, 2))) ≈ fft(x, (1, 2))
    @test @jit(fft(x_ra, (1, 2, 3))) ≈ fft(x, (1, 2, 3))
    @test @jit(fft(x_ra, (2, 3))) ≈ fft(x, (2, 3))
    @test @jit(fft(x_ra, (1, 3))) ≈ fft(x, (1, 3))

    @test_throws AssertionError @jit(fft(x_ra, (3, 2)))
    @test_throws AssertionError @jit(fft(x_ra, (1, 4)))

    y_ra = @jit(fft(x_ra))
    @test @jit(ifft(y_ra)) ≈ x
end

@testset "rfft" begin
    x = rand(2, 2, 3, 4)
    x_ra = Reactant.ConcreteRArray(x)

    @test_throws AssertionError @jit(rfft(x_ra))

    x = rand(2, 3, 4)
    x_ra = Reactant.ConcreteRArray(x)

    @test @jit(rfft(x_ra)) ≈ rfft(x)
    @test @jit(rfft(x_ra, (1, 2))) ≈ rfft(x, (1, 2))
    @test @jit(rfft(x_ra, (1, 2, 3))) ≈ rfft(x, (1, 2, 3))
    @test @jit(rfft(x_ra, (2, 3))) ≈ rfft(x, (2, 3))
    @test @jit(rfft(x_ra, (1, 3))) ≈ rfft(x, (1, 3))

    @test_throws AssertionError @jit(rfft(x_ra, (3, 2)))
    @test_throws AssertionError @jit(rfft(x_ra, (1, 4)))

    y_ra = @jit(rfft(x_ra))
    @test @jit(irfft(y_ra, 2)) ≈ x
    @test @jit(irfft(y_ra, 3)) ≈ irfft(rfft(x), 3)
end
