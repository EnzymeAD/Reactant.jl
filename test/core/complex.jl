using Enzyme
using Reactant
using Test

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

@testset "Complex runtime: $CT" for CT in (ComplexF32, ComplexF64)
    @test begin
        a = Reactant.to_rarray(ones(CT, 2))
        b = Reactant.to_rarray(ones(CT, 2))
        c = Reactant.compile(+, (a, b))(a, b)
        c == ones(CT, 2) + ones(CT, 2)
    end skip = CT == ComplexF64 && RunningOnTPU
end

const SCALAR_LIST = (1.0, 1.0 + 2.0im)

const ARRAY_LIST = (
    fill(1.0 + 2.0im),
    fill(1.0),
    [1.0 + 2.0im; 3.0 + 4.0im],
    [1.0; 3.0],
    [1.0 + 2.0im 3.0 + 4.0im],
    [1.0 2.0],
    [1.0+2.0im 3.0+4.0im; 5.0+6.0im 7.0+8.0im],
    [1.0 3.0; 5.0 7.0],
)

@testset "$(string(fn))" for fn in (conj, conj!, real, imag)
    if !endswith(string(fn), "!")
        @testset "$(typeof(x))" for x in SCALAR_LIST
            @test begin
                x_concrete = Reactant.to_rarray(x)
                only(@jit(fn(x_concrete))) == fn(x)
            end skip = RunningOnTPU && eltype(x) == ComplexF64
        end
    end

    @testset "$(typeof(x))" for x in ARRAY_LIST
        @test begin
            x_concrete = Reactant.to_rarray(x)
            @jit(fn(x_concrete)) == fn(x)
        end skip = RunningOnTPU && eltype(x) == ComplexF64
    end
end

@testset "abs: $T" for T in (Float32, ComplexF32)
    x = Reactant.TestUtils.construct_test_array(T, 10)
    x_concrete = Reactant.to_rarray(x)
    @test @jit(abs.(x_concrete)) ≈ abs.(x)
end

@testset "promote_to Complex" begin
    x = ComplexF32(1.0 + 2.0im)
    y = ConcreteRNumber(x)

    f = Reactant.compile((y,)) do z
        z + Reactant.promote_to(Reactant.TracedRNumber{ComplexF32}, ComplexF32(1.0 - 3.0im))
    end

    @test isapprox(f(y), ComplexF32(2.0 - 1.0im))
end

@testset "multiplication by im" begin
    x = ConcreteRNumber(42.0f0)

    mul_im(x) = x * im
    mul_re(x) = x * Complex(true, false)
    mul_re_im(x) = x * Complex(true, true)
    mul_nothing(x) = x * Complex(false, false)

    @test (@jit mul_im(x)) == Complex(0.0f0, 42.0f0)
    @test (@jit mul_re(x)) == Complex(42.0f0, 0.0f0)
    @test (@jit mul_re_im(x)) == Complex(42.0f0, 42.0f0)
    @test (@jit mul_nothing(x)) == Complex(0.0f0, 0.0f0)
end

@testset "broadcasting with im / 1im (#3395)" begin
    fns = (
        (x, y) -> x .- im .* y,
        (x, y) -> x .- 1im .* y,
        (x, y) -> im .* x,
        (x, y) -> x .* 1im,
        (x, y) -> x .+ im,
        (x, y) -> x .+ 1im,
        (x, y) -> x ./ im,
        (x, y) -> im ./ x,
        (x, y) -> x .^ im,
        (x, y) -> exp.(im .* x),
        (x, y) -> ((a, b) -> a * im + b).(x, y),
        (x, y) -> x .+ ComplexF16(1, 2),
        (x, y) -> x .* Ref(im),
    )

    @testset "$(T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        skip_test = RunningOnTPU && (T == Float64 || T == ComplexF64)
        x = Reactant.TestUtils.construct_test_array(T, 3)
        y = Reactant.TestUtils.construct_test_array(T, 3)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @testset for f in fns
            # eltype must follow Base's promotion (e.g. `Float32 * im` is `ComplexF32`)
            @test begin
                ref = f(x, y)
                res = @jit f(x_ra, y_ra)
                res isa Reactant.AbstractConcreteArray{eltype(ref)} && Array(res) ≈ ref
            end skip = skip_test
        end

        # 0-dimensional broadcast with a traced scalar / 0-d array
        x_rn = ConcreteRNumber(x[1])
        @test @jit(x_rn .* im) ≈ x[1] * im skip = skip_test
        @test @jit(x_rn .+ 1im) ≈ x[1] + 1im skip = skip_test
        @test @jit(x_rn .* Ref(im)) ≈ x[1] * im skip = skip_test
        x_0d = Reactant.to_rarray(fill(x[1]))
        @test @jit(x_0d .* im) ≈ fill(x[1]) .* im skip = skip_test
    end

    @testset "in-place and autodiff" begin
        x = Float32[1, 2, 3]
        y = Float32[4, 5, 6]
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        inplace_expr!(d, a, b) = (d .= a .- im .* b; return nothing)
        dest = Reactant.to_rarray(zeros(ComplexF32, 3))
        @jit inplace_expr!(dest, x_ra, y_ra)
        @test Array(dest) ≈ x .- im .* y

        loss_im(a, b) = sum(abs2, a .- im .* b)
        grad_im(a, b, da, db) = begin
            Enzyme.autodiff(Reverse, loss_im, Active, Duplicated(a, da), Duplicated(b, db))
            return da, db
        end
        dx_ra = Reactant.to_rarray(zeros(Float32, 3))
        dy_ra = Reactant.to_rarray(zeros(Float32, 3))
        @jit grad_im(x_ra, y_ra, dx_ra, dy_ra)
        @test Array(dx_ra) ≈ 2 .* x
        @test Array(dy_ra) ≈ 2 .* y
    end

    # XLA only supports ComplexF32 and ComplexF64; make sure unsupported complex element
    # types fail with a clear error message instead of crashing or returning garbage.
    int_ra = Reactant.to_rarray([1, 2, 3])
    @test_throws "eltype of complex must be Float32 or Float64" @jit(int_ra .* 1im)
    @test_throws "eltype of complex must be Float32 or Float64" @jit(int_ra .* im)
    @test_throws "eltype of complex must be Float32 or Float64" @jit(
        ConcreteRNumber(1) * im
    )
    f16_ra = Reactant.to_rarray(Float16[1, 2])
    @test_throws "eltype of complex must be Float32 or Float64" @jit(f16_ra .* im)
end

@testset "complex reduction" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 10, 10)
    x_ra = Reactant.to_rarray(x)
    @test @jit(sum(abs2, x_ra)) ≈ sum(abs2, x)
end

@testset "create complex numbers" begin
    x = ComplexF32(1.0 - 2.0im)
    x_ra = Reactant.to_rarray(x; track_numbers=true)
    @test @jit(Complex(x_ra)) == x_ra

    x = 1.5f0
    y = -2.25
    x_ra = Reactant.to_rarray(x; track_numbers=true)
    y_ra = Reactant.to_rarray(y; track_numbers=true)
    @test @jit(Complex(x_ra, y_ra)) == Complex(x, y) skip = RunningOnTPU
    @test @jit(Complex(x_ra, y)) == Complex(x, y) skip = RunningOnTPU
    @test @jit(Complex(x, y_ra)) == Complex(x, y) skip = RunningOnTPU
    @test @jit(Complex(x_ra)) == Complex(x) == @jit(Complex(x_ra, 0))
end

@testset "Base.complex" begin
    @test ComplexF32(@jit(complex(ConcreteRNumber(1.0f0)))) == complex(1.0f0)
    @test ComplexF32(@jit(complex(ConcreteRNumber(1.0f0), ConcreteRNumber(2.0f0)))) ==
        complex(1.0f0, 2.0f0)
    @test ComplexF32(@jit(complex(ConcreteRNumber(1.0f0 + 2.0f0im)))) ==
        complex(1.0f0 + 2.0f0im)

    @test Array(@jit(complex(ConcreteRArray(ones(Float32, 2))))) ≈ complex(ones(Float32, 2))
    @test Array(@jit(complex(ConcreteRArray(ones(ComplexF32, 2))))) ≈
        complex(ones(ComplexF32, 2))
end

@testset "isreal" begin
    @testset "scalar" for x in (1.0, 1.0 + 2.0im, 1.0 + 0.0im)
        x_concrete = Reactant.to_rarray(x)
        @test only(@jit(isreal(x_concrete))) == isreal(x)
    end

    @testset "broadcast" begin
        x = [1.0 + 0.0im, 1.0 + 2.0im, 1.0 - 2.0im, -5.0 + 0.0im]
        x_concrete = Reactant.to_rarray(x)
        @test @jit(isreal.(x_concrete)) == isreal.(x)
    end
end
