using FFTW, Reactant, Test

@testset "fft" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test_throws AssertionError @jit(fft(x_ra)) # TODO: support this

    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
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

    shifted_fft = @jit(fftshift(y_ra))
    @test shifted_fft ≈ fftshift(Array(y_ra))
    @test @jit(ifftshift(shifted_fft)) ≈ Array(y_ra)

    @testset "fft real input" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4)
        x_ra = Reactant.to_rarray(x)

        @test @jit(fft(x_ra)) ≈ fft(x)
        @test @jit(fft(x_ra, (1, 2))) ≈ fft(x, (1, 2))
        @test @jit(fft(x_ra, (1, 2, 3))) ≈ fft(x, (1, 2, 3))
    end
end

@testset "fft!" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)
    x_ra_copy = copy(x_ra)

    y_ra = @jit(fft!(x_ra))
    @test y_ra ≈ fft(x)
    @test x_ra ≈ y_ra
    @test x_ra ≉ x_ra_copy

    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)
    x_ra_copy = copy(x_ra)
    y_ra = @jit(fft!(x_ra, (1, 2)))
    @test y_ra ≈ fft(x, (1, 2))
    @test x_ra ≈ y_ra
    @test x_ra ≉ x_ra_copy

end

@testset "rfft" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test_throws AssertionError @jit(rfft(x_ra)) # TODO: support this

    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4)
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

@testset "Planned FFTs" begin
    @testset "Out-of-place [$(fft), size $(size)]" for size in ((16,), (16, 16)),
        (plan, fft) in (
            (FFTW.plan_fft, FFTW.fft),
            (FFTW.plan_ifft, FFTW.ifft),
            (FFTW.plan_rfft, FFTW.rfft),
        )

        x = randn(fft === FFTW.rfft ? Float32 : ComplexF32, size)
        x_r = Reactant.to_rarray(x)
        # We make a copy of the original array to make sure the operation does
        # not modify the input.
        copied_x_r = copy(x_r)

        planned_fft(x) = plan(x) * x
        compiled_planned_fft = @compile planned_fft(x_r)
        # Make sure the result is correct
        @test compiled_planned_fft(x_r) ≈ fft(x)
        # Make sure the operation is not in-place
        @test x_r == copied_x_r

        if length(size) > 1
            planned_fft_dims(x, dims) = plan(x, dims) * x
            compiled_planned_fft_dims = @compile planned_fft_dims(x_r, (1,))
            # Make sure the result is correct
            @test compiled_planned_fft_dims(x_r) ≈ fft(x, (1,))
            # Make sure the operation is not in-place
            @test x_r == copied_x_r
        end
    end

    @testset "In-place [$(fft!), size $(size)]" for size in ((16,), (16, 16)),
        (plan!, fft!) in ((FFTW.plan_fft!, FFTW.fft!), (FFTW.plan_ifft!, FFTW.ifft!))

        x = randn(ComplexF32, size)
        x_r = Reactant.to_rarray(x)
        # We make a copy of the original array to make sure the operation
        # modifies the input.
        copied_x_r = copy(x_r)

        planned_fft!(x) = plan!(x) * x
        compiled_planned_fft! = @compile planned_fft!(x_r)
        planned_y_r = compiled_planned_fft!(x_r)
        # Make sure the result is correct
        @test planned_y_r ≈ fft!(x)
        # Make sure the operation is in-place
        @test planned_y_r ≈ x_r
        @test x_r ≉ copied_x_r
    end
end

@testset "spectral convolution" begin
    function spectral_conv(x, weight; dims=(1, 2))
        ω = fft(x, dims)
        ω_inverse = ifftshift(fftshift(ω, dims) .* weight, dims)
        return sum(abs2, ifft(ω_inverse, dims))
    end

    x = Reactant.TestUtils.construct_test_array(Float32, 8, 8, 4, 1)
    weight = ones(Float32, 1, 1) .* 4

    x_r = Reactant.to_rarray(x)
    weight_r = Reactant.to_rarray(weight)

    @test spectral_conv(x, weight) ≈ @jit(spectral_conv(x_r, weight_r))
end
