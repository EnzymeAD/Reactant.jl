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
            @test compiled_planned_fft_dims(x_r, (1,)) ≈ fft(x, (1,))
            # Make sure the operation is not in-place
            @test x_r == copied_x_r
        end
    end

    @testset "Out-of-place irfft" begin
        size = (16, 16)
        x = randn(ComplexF32, size)
        x_r = Reactant.to_rarray(x)
        # We make a copy of the original array to make sure the operation does
        # not modify the input.
        copied_x_r = copy(x_r)

        d = 31 # original real length
        planned_irfft(x, d) = FFTW.plan_irfft(x, d) * x
        compiled_planned_irfft = @compile planned_irfft(x_r, d)
        # Make sure the result is correct
        @test compiled_planned_irfft(x_r, d) ≈ irfft(x, d)
        # Make sure the operation is not in-place
        @test x_r == copied_x_r

        planned_irfft_dims(x, d, dims) = FFTW.plan_irfft(x, d, dims) * x
        compiled_planned_irfft_dims = @compile planned_irfft_dims(x_r, d, (1,))
        # Make sure the result is correct
        @test compiled_planned_irfft_dims(x_r, d, (1,)) ≈ irfft(x, d, (1,))
        # Make sure the operation is not in-place
        @test x_r == copied_x_r
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

@testset "FFTW Plans with Traced Arrays" begin
    @testset "Complex Plans" begin
        x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
        x_r = Reactant.to_rarray(x)
        p = plan_fft(copy(x))
        p12 = plan_fft(copy(x), (1, 2))

        @test @jit(p * x_r) ≈ p * x
        @test @jit(p12 * x_r) ≈ p12 * x

        y_r = similar(x_r)
        @jit LinearAlgebra.mul!(y_r, p, x_r)
        @test y_r ≈ @jit(p * x_r)

        ip = plan_ifft(copy(x))
        @test @jit(ip * x_r) ≈ ip * x

        ip12 = plan_ifft(copy(x), (1, 2))
        @test @jit(ip12 * x_r) ≈ ip12 * x

        y_r = similar(x_r)
        @jit LinearAlgebra.mul!(y_r, ip, x_r)
        @test y_r ≈ @jit(ip * x_r)

        # In-place plans
        p! = plan_fft!(copy(x))
        x_r = Reactant.to_rarray(x)
        @test @jit(p! * x_r) ≈ p! * x
        fill!(y_r, 0)
        @jit LinearAlgebra.mul!(y_r, p!, copy(x_r))
        @jit(p! * x_r)
        @test y_r ≈ x_r

        ip! = plan_ifft!(copy(x))
        x_r = Reactant.to_rarray(x)
        @test @jit(ip! * x_r) ≈ ip! * x
        fill!(y_r, 0)
        @jit LinearAlgebra.mul!(y_r, ip!, copy(x_r))
        @jit(ip! * x_r)
        @test y_r ≈ x_r
    end

    @testset "r2c plans" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 6, 8, 7)
        x_r = Reactant.to_rarray(x)
        p = plan_rfft(copy(x))
        p12 = plan_rfft(copy(x), (1, 2))
        p23 = plan_rfft(copy(x), (2, 3))

        @test @jit(p * x_r) ≈ p * x
        @test @jit(p12 * x_r) ≈ p12 * x
        @test @jit(p23 * x_r) ≈ p23 * x

        y_r = similar(Reactant.to_rarray(rfft(x)))
        @jit LinearAlgebra.mul!(y_r, p, x_r)
        @test y_r ≈ @jit(p * x_r)

        y_r12 = similar(Reactant.to_rarray(rfft(x, (1, 2))))
        @jit LinearAlgebra.mul!(y_r12, p12, x_r)
        @test y_r12 ≈ @jit(p12 * x_r)

        y_r23 = similar(Reactant.to_rarray(rfft(x, (2, 3))))
        @jit LinearAlgebra.mul!(y_r23, p23, x_r)
        @test y_r23 ≈ @jit(p23 * x_r)

        c = rand(ComplexF32, size(y_r)...)
        ip = plan_irfft(copy(c), size(x, 1))
        c_r = Reactant.to_rarray(c)
        @test @jit(ip * c_r) ≈ ip * c
        y_r = similar(Reactant.to_rarray(irfft(c, size(x, 1))))
        @jit LinearAlgebra.mul!(y_r, ip, c_r)
        @test y_r ≈ @jit(ip * c_r)

        c12 = rand(ComplexF32, size(p12*x)...)
        ip12 = plan_irfft(copy(c12), size(x, 1), (1, 2))
        c12_r = Reactant.to_rarray(c12)
        @test @jit(ip12 * c12_r) ≈ ip12 * c12
        y_r = similar(Reactant.to_rarray(irfft(c12, size(x, 1), (1, 2))))
        @jit LinearAlgebra.mul!(y_r, ip12, c12_r)
        @test y_r ≈ @jit(ip12 * c12_r)

        c23 = rand(ComplexF32, size(p23*x)...)
        ip23 = plan_irfft(copy(c23), size(x, 2), (2, 3))
        c23_r = Reactant.to_rarray(c23)
        @test @jit(ip23 * c23_r) ≈ ip23 * c23
        y_r = similar(Reactant.to_rarray(irfft(c23, size(x, 2), (2, 3))))
        @jit LinearAlgebra.mul!(y_r, ip23, c23_r)
        @test y_r ≈ @jit(ip23 * c23_r)
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
