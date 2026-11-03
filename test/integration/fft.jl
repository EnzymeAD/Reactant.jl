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

module FFT3259Global
using Reactant: Reactant
const GLOBAL_RA = Ref{Any}()
end

@testset "Untraced, concrete, and mismatched arrays (#3259)" begin
    AFTEx = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

    @testset "Host Array outside @compile throws ArgumentError" begin
        x_c = randn(ComplexF64, 4, 4, 4)
        x_r = randn(Float64, 4, 4, 4)
        x_rc = rfft(x_r, (1, 2))
        y_c = similar(x_c)
        y_rc = similar(x_rc)
        y_r = similar(x_r)

        p_ip = AFTEx.ReactantFFTInPlacePlan{ComplexF64}((1, 2))
        p_iip = AFTEx.ReactantIFFTInPlacePlan{ComplexF64}((1, 2))
        p_oop = AFTEx.ReactantFFTPlan{ComplexF64}((1, 2))
        p_ioop = AFTEx.ReactantIFFTPlan{ComplexF64}((1, 2))
        p_rfft = AFTEx.ReactantRFFTPlan{Float64}((1, 2))
        p_irfft = AFTEx.ReactantIRFFTPlan{ComplexF64}((1, 2), 4)

        for p in (p_ip, p_iip, p_oop, p_ioop, 2.0 * p_ip, 2.0 * p_oop)
            @test_throws ArgumentError (p * x_c)
            @test_throws ArgumentError LinearAlgebra.mul!(y_c, p, x_c)
        end
        @test_throws ArgumentError (p_rfft * x_r)
        @test_throws ArgumentError LinearAlgebra.mul!(y_rc, p_rfft, x_r)
        @test_throws ArgumentError (p_irfft * x_rc)
        @test_throws ArgumentError LinearAlgebra.mul!(y_r, p_irfft, x_rc)
        @test_throws ArgumentError ((2.0 * p_irfft) * x_rc)
        @test_throws ArgumentError LinearAlgebra.mul!(y_r, 2.0 * p_irfft, x_rc)
    end

    @testset "Eager ConcreteRArray and reshaped ConcreteRArray" begin
        x0 = Reactant.TestUtils.construct_test_array(ComplexF64, 4, 4, 4)
        p_ip = AFTEx.ReactantFFTInPlacePlan{ComplexF64}((1, 2))
        p_iip = AFTEx.ReactantIFFTInPlacePlan{ComplexF64}((1, 2))
        p_oop = AFTEx.ReactantFFTPlan{ComplexF64}((1, 2))
        p_rfft = AFTEx.ReactantRFFTPlan{Float32}((1, 2))
        p_irfft = AFTEx.ReactantIRFFTPlan{ComplexF32}((1, 2), 4)

        # In-place forward, inverse, and scaled plans mutate ConcreteRArray in-place
        x_ra = Reactant.to_rarray(copy(x0))
        res = p_ip * x_ra
        @test Array(x_ra) ≈ fft(x0, (1, 2))
        @test Array(res) ≈ fft(x0, (1, 2))

        p_iip * x_ra
        @test Array(x_ra) ≈ x0

        (2.0 * p_ip) * x_ra
        @test Array(x_ra) ≈ 2.0 .* fft(x0, (1, 2))

        # Reshaped 1D ConcreteRArray buffer mutated in-place (Oceananigans buffer pattern)
        buf = Reactant.to_rarray(copy(vec(x0)))
        p_ip * reshape(buf, 4, 4, 4)
        @test Array(buf) ≈ vec(fft(x0, (1, 2)))
        (0.5 * p_iip) * reshape(buf, 4, 4, 4)
        @test Array(buf) ≈ vec(0.5 .* x0)

        # Out-of-place plans do not mutate input ConcreteRArray
        x_ra = Reactant.to_rarray(copy(x0))
        y_ra = p_oop * x_ra
        @test Array(y_ra) ≈ fft(x0, (1, 2))
        @test Array(x_ra) ≈ x0
        @test Array((3.0 * p_oop) * x_ra) ≈ 3.0 .* fft(x0, (1, 2))
        @test Array(x_ra) ≈ x0

        xr0 = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 4)
        xr_ra = Reactant.to_rarray(copy(xr0))
        yr_ra = p_rfft * xr_ra
        @test Array(yr_ra) ≈ rfft(xr0, (1, 2))
        @test Array(p_irfft * yr_ra) ≈ xr0

        # Eager mul! on ConcreteRArray (including ScaledPlan)
        dest_ra = similar(x_ra)
        LinearAlgebra.mul!(dest_ra, p_oop, x_ra)
        @test Array(dest_ra) ≈ fft(x0, (1, 2))
        LinearAlgebra.mul!(dest_ra, 2.5 * p_oop, x_ra)
        @test Array(dest_ra) ≈ 2.5 .* fft(x0, (1, 2))

        # Real-input plan on complex ConcreteRArray throws ArgumentError
        xc32_ra = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(ComplexF32, 4, 4, 4)
        )
        @test_throws ArgumentError (p_rfft * xc32_ra)
        @test_throws ArgumentError ((2.0f0 * p_rfft) * xc32_ra)
        @test_throws ArgumentError LinearAlgebra.mul!(similar(yr_ra), p_rfft, xc32_ra)
    end

    @testset "Untraced arrays inside @compile" begin
        x_const = Reactant.TestUtils.construct_test_array(ComplexF32, 8, 4)
        xr_const = Reactant.TestUtils.construct_test_array(Float32, 8, 4)
        xrc_const = rfft(xr_const)
        FFT3259Global.GLOBAL_RA[] = Reactant.to_rarray(copy(x_const))

        # Out-of-place plans promote untraced constant Array and untraced global ConcreteRArray
        for (plan_fn, ref_fn) in (
            (FFTW.plan_fft, FFTW.fft),
            (FFTW.plan_ifft, FFTW.ifft),
            (FFTW.plan_bfft, FFTW.bfft),
        )
            p = plan_fn(copy(x_const))
            f_const(p) = p * x_const
            f_global_ra(p) =
                p * (FFT3259Global.GLOBAL_RA[]::typeof(FFT3259Global.GLOBAL_RA[]))
            @test Array((@compile f_const(p))(p)) ≈ ref_fn(x_const)
            @test Array((@compile f_global_ra(p))(p)) ≈ ref_fn(x_const)
        end

        pr = FFTW.plan_rfft(copy(xr_const))
        run_rfft_const(p) = p * xr_const
        @test Array((@compile run_rfft_const(pr))(pr)) ≈ rfft(xr_const)

        for (plan_fn, ref_fn) in
            ((FFTW.plan_irfft, FFTW.irfft), (FFTW.plan_brfft, FFTW.brfft))
            pir = plan_fn(copy(xrc_const), size(xr_const, 1))
            run_irfft_const(p) = p * xrc_const
            @test Array((@compile run_irfft_const(pir))(pir)) ≈
                ref_fn(xrc_const, size(xr_const, 1))
        end

        # Traced planner applied to constant host Array inside @compile
        x_seed = Reactant.to_rarray(copy(x_const))
        f_traced_plan_const(seed) = plan_fft(seed) * x_const
        @test Array((@compile f_traced_plan_const(x_seed))(x_seed)) ≈ fft(x_const)

        # mul! with traced destination and untraced constant input succeeds
        y_out = Reactant.to_rarray(similar(x_const))
        p_oop = FFTW.plan_fft(copy(x_const))
        p_bfft = FFTW.plan_bfft(copy(x_const))
        f_mul_const!(y, p) = LinearAlgebra.mul!(y, p, x_const)
        (@compile f_mul_const!(y_out, p_oop))(y_out, p_oop)
        @test Array(y_out) ≈ fft(x_const)
        (@compile f_mul_const!(y_out, p_bfft))(y_out, p_bfft)
        @test Array(y_out) ≈ bfft(x_const)

        # Closure-captured ConcreteRArray is traced and mutated in-place, while untraced
        # host Arrays and untraced global ConcreteRArrays throw ArgumentError
        for (plan_fn!, ref_fn) in (
            (FFTW.plan_fft!, FFTW.fft),
            (FFTW.plan_ifft!, FFTW.ifft),
            (FFTW.plan_bfft!, FFTW.bfft),
        )
            p! = plan_fn!(copy(x_const))
            closed_ra = Reactant.to_rarray(copy(x_const))
            f_ip_closed(p) = p * closed_ra
            (@compile f_ip_closed(p!))(p!)
            @test Array(closed_ra) ≈ ref_fn(x_const)

            f_ip_const(p) = p * copy(x_const)
            f_ip_global_ra(p) =
                p * (FFT3259Global.GLOBAL_RA[]::typeof(FFT3259Global.GLOBAL_RA[]))
            @test_throws ArgumentError @compile f_ip_const(p!)
            @test_throws ArgumentError @compile f_ip_global_ra(p!)
        end
        f_ip_traced_plan(seed) = plan_fft!(seed) * copy(x_const)
        @test_throws ArgumentError @compile f_ip_traced_plan(x_seed)

        # mul! into untraced destination array inside @compile throws ArgumentError
        f_mul_bad_dest(p, x) = LinearAlgebra.mul!(similar(x_const), p, x)
        @test_throws ArgumentError @compile f_mul_bad_dest(p_oop, x_seed)
        @test_throws ArgumentError @compile f_mul_bad_dest(p_bfft, x_seed)
    end

    @testset "Element-type promotion and validation on TracedRArray" begin
        x_f64 = Reactant.TestUtils.construct_test_array(Float64, 8)
        x_cf32 = Reactant.TestUtils.construct_test_array(ComplexF32, 8)
        x_f64_ra = Reactant.to_rarray(x_f64)
        x_cf32_ra = Reactant.to_rarray(x_cf32)

        p_cf64 = FFTW.plan_fft(zeros(ComplexF64, 8))
        res_f64 = @jit(p_cf64 * x_f64_ra)
        @test eltype(res_f64) == ComplexF64
        @test Array(res_f64) ≈ p_cf64 * x_f64

        res_cf32 = @jit(p_cf64 * x_cf32_ra)
        @test eltype(res_cf32) == ComplexF64
        @test Array(res_cf32) ≈ p_cf64 * x_cf32

        # In-place ComplexF64 plan applied to Float64 TracedRArray allocates/promotes without mutating x_f64_ra
        p_cf64! = FFTW.plan_fft!(zeros(ComplexF64, 8))
        x_f64_copy = copy(x_f64_ra)
        res_ip_f64 = @jit(p_cf64! * x_f64_ra)
        @test eltype(res_ip_f64) == ComplexF64
        @test Array(res_ip_f64) ≈ p_cf64! * x_f64
        @test Array(x_f64_ra) == Array(x_f64_copy)

        # mul! with mismatched input element type promotes to plan precision
        y_cf64_ra = Reactant.to_rarray(zeros(ComplexF64, 8))
        @jit LinearAlgebra.mul!(y_cf64_ra, p_cf64, x_cf32_ra)
        @test Array(y_cf64_ra) ≈ p_cf64 * x_cf32

        # Real-input rfft plan on Complex TracedRArray throws ArgumentError (does not drop imag)
        p_rfft32 = FFTW.plan_rfft(zeros(Float32, 8))
        y_rfft32_ra = Reactant.to_rarray(zeros(ComplexF32, 5))
        @test_throws ArgumentError @jit(p_rfft32 * x_cf32_ra)
        @test_throws ArgumentError @jit LinearAlgebra.mul!(y_rfft32_ra, p_rfft32, x_cf32_ra)
    end
end
