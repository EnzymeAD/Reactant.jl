using Reactant, Test, FillArrays

fn(x, y) = (2 .* x .- 3) * y'

@testset "Fill" begin
    x = Fill(2.0f0, 4, 5)
    rx = Reactant.to_rarray(x)

    @test @jit(fn(rx, rx)) ≈ fn(x, x)

    @testset "Ones" begin
        y = Ones(Float32, 4, 5)
        ry = Reactant.to_rarray(y)
        @test @jit(fn(rx, ry)) ≈ fn(x, y)
    end

    @testset "Zeros" begin
        y = Zeros(Float32, 4, 5)
        ry = Reactant.to_rarray(y)
        @test @jit(fn(rx, ry)) ≈ fn(x, y)
    end
end

fn2(x, y) = ((2 .* x .- 3) * y')[:, 3]

@testset "OneElement" begin
    x = OneElement(3.4f0, (3, 4), (32, 32))
    rx = Reactant.to_rarray(x)

    @test @jit(fn2(rx, rx)) ≈ fn2(x, x)
end

@testset "indexing" begin
    @testset "scalar" begin
        fscalar(x) = x[3]

        N = 4
        for x in (Fill(2.0f0, N), Ones(Float32, N), Zeros(Float32, N))
            rx = Reactant.to_rarray(x)
            @test @jit(fscalar(rx)) ≈ fscalar(x)
        end

        N = 2
        for x in (Fill(2.0f0, N), Ones(Float32, N), Zeros(Float32, N))
            rx = Reactant.to_rarray(x)
            @test_throws BoundsError @jit(fscalar(rx))
        end
    end

    @testset "slice" begin
        fslice(x) = x[3:5, 2]

        N = (6, 3)
        for x in (Fill(2.0f0, N...), Ones(Float32, N...), Zeros(Float32, N...))
            rx = Reactant.to_rarray(x)
            @test @jit(fslice(rx)) ≈ fslice(x)
        end

        N = (4, 3)
        for x in (Fill(2.0f0, N...), Ones(Float32, N...), Zeros(Float32, N...))
            rx = Reactant.to_rarray(x)
            @test_throws BoundsError @jit(fslice(rx))
        end
    end
end

# Plain (non-`to_rarray`'d) `Fill`-like arguments must behave like compile-time constants.
# Previously the scalar inside was unconditionally promoted to a `TracedRNumber` which
# became a spurious output that the thunk tried to `setfield!` into the immutable struct.
@testset "plain Fill-like arguments (#2223, #2070)" begin
    @testset "setindex! with Fill (#2223)" begin
        set_fill!(a, idx, f) = (a[idx] = f; a)

        a = rand(32)
        ar = Reactant.to_rarray(a)
        f = Fill(0.0, 4)

        hlo = repr(@code_hlo set_fill!(ar, 1:4, f))
        @test !occursin("-> (tensor<32xf64>, tensor<f64>)", hlo)

        @test Array(@jit(set_fill!(ar, 1:4, f))) ≈ set_fill!(copy(a), 1:4, f)
        @test Array(ar) ≈ set_fill!(copy(a), 1:4, f) # mutated in place
    end

    @testset "broadcast with plain $(nameof(typeof(y)))" for y in (
        Fill(2.0, 32), Zeros(32), Ones(32), OneElement(3.0, 2, 32)
    )
        addf(x, y) = x .+ y

        a = rand(32)
        ar = Reactant.to_rarray(a)
        @test Array(@jit(addf(ar, y))) ≈ addf(a, y)
    end

    @testset "Fill from traced scalar" begin
        fill_add(x, s) = x .+ Fill(s, 32)

        a = rand(32)
        ar = Reactant.to_rarray(a)
        @test Array(@jit(fill_add(ar, Reactant.ConcreteRNumber(3.0)))) ≈ fill_add(a, 3.0)
    end

    @testset "Fill inside immutable struct + mapreduce (#2070)" begin
        α = Fill(1.0, 4, 4)
        lmnB = sum(log, α) - log(sum(α))

        function dirichlet_lpdf(α, lmnB, x)
            s = mapreduce(+, x, α) do xi, αi
                result = (αi - 1) * log(xi)
                ifelse(iszero(αi - 1) & isnan(xi), zero(result), result)
            end
            return s - lmnB
        end

        x = rand(4, 4)
        x ./= sum(x)
        xr = Reactant.to_rarray(x)

        d = (; α, lmnB)
        logpdf(d, x) = dirichlet_lpdf(d.α, d.lmnB, x)
        @test @jit(logpdf(d, xr)) ≈ logpdf(d, x)
    end
end
