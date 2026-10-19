using FFTW, Reactant, Test, LinearAlgebra

@testset "fft" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test_throws AssertionError @jit(fft(x_ra)) # TODO(#2243): support this

    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(fft(x_ra)) ≈ fft(x)
    @test @jit(fft(x_ra, (1, 2))) ≈ fft(x, (1, 2))
    @test @jit(fft(x_ra, (1, 2, 3))) ≈ fft(x, (1, 2, 3))
    @test @jit(fft(x_ra, (2, 3))) ≈ fft(x, (2, 3))
    @test @jit(fft(x_ra, (1, 3))) ≈ fft(x, (1, 3))
    @test @jit(fft(x_ra, 1)) ≈ fft(x, 1)
    @test @jit(fft(x_ra, 2)) ≈ fft(x, 2)
    @test @jit(fft(x_ra, 3)) ≈ fft(x, 3)
    @test @jit(ifft(x_ra, 1)) ≈ ifft(x, 1)
    @test @jit(ifft(x_ra, 2)) ≈ ifft(x, 2)
    @test @jit(ifft(x_ra, 3)) ≈ ifft(x, 3)

    @test @jit(fft(x_ra, (3, 2))) ≈ fft(x, (3, 2))
    @test_throws AssertionError @jit(fft(x_ra, (1, 4)))

    y_ra = @jit(fft(x_ra))
    @test @jit(ifft(y_ra)) ≈ x
    @test @jit(bfft(y_ra)) ≈ x * length(x)

    shifted_fft = @jit(fftshift(y_ra))
    @test shifted_fft ≈ fftshift(Array(y_ra))
    @test @jit(ifftshift(shifted_fft)) ≈ Array(y_ra)

    @testset "fft real input" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4)
        x_ra = Reactant.to_rarray(x)

        @test @jit(fft(x_ra)) ≈ fft(x)
        @test @jit(fft(x_ra, (1, 2))) ≈ fft(x, (1, 2))
        @test @jit(fft(x_ra, (1, 2, 3))) ≈ fft(x, (1, 2, 3))
        @test @jit(ifft(x_ra)) ≈ ifft(x)
        @test @jit(bfft(x_ra)) ≈ bfft(x)
        @test @jit(fft(x_ra, 1)) ≈ fft(x, 1)
        @test @jit(fft(x_ra, 2)) ≈ fft(x, 2)
        @test @jit(fft(x_ra, 3)) ≈ fft(x, 3)
    end
end

@testset "fft!" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)
    x_ra_copy = copy(x_ra)

    y_ra = @jit(fft!(x_ra))
    @test x_ra ≈ fft(x)
    @test y_ra == x_ra # These really should be the same memory right?
    @test x_ra ≉ x_ra_copy

    y_ra = @jit(ifft!(x_ra))
    @test y_ra == x_ra # These really should be the same memory right?
    @test x_ra ≈ x

    x_ra .= fft(x)
    y_ra = @jit(bfft!(x_ra))
    @test x_ra ≈ (bfft(fft(x)))
    @test y_ra ≈ x_ra
    @test y_ra == x_ra # These really should be the same memory right?
    @test x_ra ≈ x .* length(x)

    x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)
    x_ra_copy = copy(x_ra)
    @jit(fft!(x_ra, (1, 2)))
    @test x_ra ≈ fft(x, (1, 2))
    @test x_ra ≉ x_ra_copy

    y_ra = @jit(ifft!(x_ra, (1, 2)))
    @test y_ra ≈ x_ra
    @test y_ra == x_ra # These really should be the same memory right?
    @test x_ra ≈ x

    x_ra .= fft(x, (1, 2))
    y_ra = @jit(bfft!(x_ra, (1, 2)))
    @test x_ra ≈ (bfft(fft(x, (1, 2)), (1, 2)))
    @test y_ra ≈ x_ra
    @test y_ra == x_ra # These really should be the same memory right?
    @test x_ra ≈ x ./ AbstractFFTs.normalization(real(eltype(x)), size(x), (1, 2))
end

@testset "rfft" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test_throws AssertionError @jit(rfft(x_ra)) # TODO(#2243): support this

    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(rfft(x_ra)) ≈ rfft(x)
    @test @jit(rfft(x_ra, (1, 2))) ≈ rfft(x, (1, 2))
    @test @jit(rfft(x_ra, (1, 2, 3))) ≈ rfft(x, (1, 2, 3))
    @test @jit(rfft(x_ra, (2, 3))) ≈ rfft(x, (2, 3))
    @test @jit(rfft(x_ra, (1, 3))) ≈ rfft(x, (1, 3))
    @test @jit(rfft(x_ra, 1)) ≈ rfft(x, 1)
    @test @jit(rfft(x_ra, 2)) ≈ rfft(x, 2)
    @test @jit(rfft(x_ra, 3)) ≈ rfft(x, 3)

    @test @jit(rfft(x_ra, (3, 2))) ≈ rfft(x, (3, 2))
    @test_throws AssertionError @jit(rfft(x_ra, (1, 4)))

    y_ra = @jit(rfft(x_ra))
    @test @jit(irfft(y_ra, 2)) ≈ x
    @test @jit(irfft(y_ra, 3)) ≈ irfft(rfft(x), 3)
    @test @jit(brfft(y_ra, 2)) ≈ x * length(x)
    @test @jit(brfft(y_ra, 3)) ≈ brfft(rfft(x), 3)

    @testset "irfft real input" begin
        y_ra_real = @jit(real(y_ra))
        y_real = Array(y_ra_real)

        @test @jit(irfft(y_ra_real, 2)) ≈ irfft(y_real, 2)
        @test @jit(irfft(y_ra_real, 2, (1, 2))) ≈ irfft(y_real, 2, (1, 2))
        @test @jit(brfft(y_ra_real, 2)) ≈ brfft(y_real, 2)
        @test @jit(brfft(y_ra_real, 2, (1, 2))) ≈ brfft(y_real, 2, (1, 2))
        @test @jit(irfft(y_ra_real, 2, 1)) ≈ irfft(y_real, 2, 1)
    end
end

@testset "Planned FFTs" begin
    @testset "Out-of-place [$(fft), size $(size)]" for size in ((16,), (16, 16)),
        (plan, fft) in (
            (FFTW.plan_fft, FFTW.fft),
            (FFTW.plan_ifft, FFTW.ifft),
            (FFTW.plan_bfft, FFTW.bfft),
            (FFTW.plan_rfft, FFTW.rfft),
        )

        x = randn(fft === FFTW.rfft ? Float32 : ComplexF32, size)
        x_r = Reactant.to_rarray(x)
        # We make a copy of the original array to make sure the operation does
        # not modify the input.
        copied_x_r = copy(x_r)

        # Also use some keyword arguments defined by FFTW.jl planner methods.
        planned_fft(x) = plan(x; timelimit=-1.0, num_threads=1) * x
        compiled_planned_fft = @compile planned_fft(x_r)
        # Make sure the result is correct
        @test compiled_planned_fft(x_r) ≈ fft(x)
        # Make sure the operation is not in-place
        @test x_r == copied_x_r

        y = fft(x)
        y_r = Reactant.to_rarray(similar(y))
        @jit LinearAlgebra.mul!(y_r, plan(x), x_r)
        @test y_r ≈ y

        if length(size) > 1
            planned_fft_dims(x, dims) = plan(x, dims) * x
            compiled_planned_fft_dims = @compile planned_fft_dims(x_r, (1,))
            # Make sure the result is correct
            @test compiled_planned_fft_dims(x_r, (1,)) ≈ fft(x, (1,))
            # Make sure the operation is not in-place
            @test x_r == copied_x_r

            y = fft(x, (1,))
            y_r = Reactant.to_rarray(similar(y))
            @jit LinearAlgebra.mul!(y_r, plan(x, (1,)), x_r)
            @test y_r ≈ y
        end
    end

    @testset "Out-of-place irfft/brfft" begin
        for size in ((16,), (16, 16)),
            (plan, fft) in ((FFTW.plan_irfft, FFTW.irfft), (FFTW.plan_brfft, FFTW.brfft))

            x = randn(ComplexF32, size)
            x_r = Reactant.to_rarray(x)
            copied_x_r = copy(x_r) # I think FFTW may sometimes modify input?

            d = 31 # original real length
            planned_fft(x, d) = plan(x, d) * x
            compiled_planned_fft = @compile planned_fft(x_r, d)
            @test compiled_planned_fft(x_r, d) ≈ fft(x, d)
            # Make sure the operation is not in-place
            @test x_r == copied_x_r

            planned_fft_dims(x, d, dims) = plan(x, d, dims) * x
            compiled_planned_fft_dims = @compile planned_fft_dims(x_r, d, (1,))
            @test compiled_planned_fft_dims(x_r, d, (1,)) ≈ fft(x, d, (1,))
            # Make sure the operation is not in-place
            @test x_r == copied_x_r

            y_r = Reactant.to_rarray(similar(fft(x, d)))
            @jit LinearAlgebra.mul!(y_r, plan(x, d), x_r)
            @test y_r ≈ plan(x, d) * x
        end
    end

    @testset "In-place [$(fft!), size $(size)]" for size in ((16,), (16, 16)),
        (plan!, fft!) in (
            (FFTW.plan_fft!, FFTW.fft!),
            (FFTW.plan_ifft!, FFTW.ifft!),
            (FFTW.plan_bfft!, FFTW.bfft!),
        )

        x = randn(ComplexF32, size)
        x_r = Reactant.to_rarray(x)
        copied_x_r = copy(x_r)

        planned_fft!(x) = plan!(x) * x
        compiled_planned_fft! = @compile planned_fft!(x_r)
        planned_y_r = compiled_planned_fft!(x_r)
        @test planned_y_r ≈ fft!(x)
        # Make sure the operation is in-place
        @test planned_y_r ≈ x_r
        @test x_r ≉ copied_x_r
    end
end

@testset "FFTW Plans with Traced Arrays" begin
    @testset "Complex Plans" begin
        for (plan, fft) in (
            (FFTW.plan_fft, FFTW.fft),
            (FFTW.plan_ifft, FFTW.ifft),
            (FFTW.plan_bfft, FFTW.bfft),
        )
            x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
            x_r = Reactant.to_rarray(x)
            y_r = similar(x_r)

            p = plan(copy(x))
            p12 = plan(copy(x), (1, 2))

            @test @jit(p * x_r) ≈ p * x
            @test @jit(p12 * x_r) ≈ p12 * x

            y_r = similar(x_r)
            @jit LinearAlgebra.mul!(y_r, p, x_r)
            @test y_r ≈ @jit(p * x_r)
        end

        for (plan!, fft) in (
            (FFTW.plan_fft!, FFTW.fft),
            (FFTW.plan_ifft!, FFTW.ifft),
            (FFTW.plan_bfft!, FFTW.bfft),
        )
            x = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4)
            x_r = Reactant.to_rarray(x)
            y_r = similar(x_r)

            # In-place plans
            p! = plan!(copy(x))
            @jit(p! * x_r)
            @test x_r ≈ fft(x)
            @jit LinearAlgebra.mul!(y_r, p!, Reactant.to_rarray(x))
            @test y_r ≈ fft(x)

            pd! = plan!(copy(x), (1, 2))
            x_r = Reactant.to_rarray(x)
            @jit(pd! * x_r)
            @test x_r ≈ fft(x, (1, 2))
            @jit LinearAlgebra.mul!(y_r, pd!, Reactant.to_rarray(x))
            @test y_r ≈ fft(x, (1, 2))
        end
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

        for plan in (FFTW.plan_irfft, FFTW.plan_brfft)
            r_r = zero(x_r)
            c = rand(ComplexF32, size(y_r)...)
            ip = plan(copy(c), size(x, 1))
            c_r = Reactant.to_rarray(c)
            @test @jit(ip * c_r) ≈ ip * c
            @jit LinearAlgebra.mul!(r_r, ip, c_r)
            @test r_r ≈ @jit(ip * c_r)

            c12 = rand(ComplexF32, size(y_r12)...)
            ip12 = plan(copy(c12), size(x, 1), (1, 2))
            c12_r = Reactant.to_rarray(c12)
            @test @jit(ip12 * c12_r) ≈ ip12 * c12
            @jit LinearAlgebra.mul!(r_r, ip12, c12_r)
            @test r_r ≈ @jit(ip12 * c12_r)

            c23 = rand(ComplexF32, size(y_r23)...)
            ip23 = plan(copy(c23), size(x, 2), (2, 3))
            c23_r = Reactant.to_rarray(c23)
            @test @jit(ip23 * c23_r) ≈ ip23 * c23
            @jit LinearAlgebra.mul!(r_r, ip23, c23_r)
            @test r_r ≈ @jit(ip23 * c23_r)
        end
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

@testset "Wrapped Plan" begin
    struct MyPlan{T}
        plan::T
    end

    myfft(mp::MyPlan, x) = mp.plan * x

    x = rand(ComplexF32, 16)
    mp = MyPlan(plan_fft!(copy(x)))

    xr = Reactant.to_rarray(x)
    cmp = @compile myfft(mp, xr)
    @test cmp(mp, xr) ≈ myfft(mp, x)
end
