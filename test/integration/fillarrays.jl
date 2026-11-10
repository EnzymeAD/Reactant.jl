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

function _setindex_fill!(a, idx, f)
    a[idx] = f
    return a
end

return_fill_arg(_x, f) = f
make_fill_const(x) = Fill(2.0f0, length(x))

function _if_fill(c, x, f)
    @trace if c > 0
        x = x .+ f
    end
    return x
end

function _while_fill(x, f)
    i = 1
    @trace while i <= 2
        x = x .+ f
        i += 1
    end
    return x
end

struct ImageDirichlet2070{T,A<:AbstractMatrix{T},S}
    α::A
    α0::T
    lmnB::S
    function ImageDirichlet2070(α::AbstractMatrix{T}) where {T}
        α0 = sum(α)
        lmnB = sum(log, α) - log(α0)
        return new{T,typeof(α),typeof(lmnB)}(α, α0, lmnB)
    end
end

ImageDirichlet2070(α::Real, nx::Int, ny::Int) = ImageDirichlet2070(Fill(α, nx, ny))

function _myxlogy_2070(x::Number, y::Number)
    result = x * log(y)
    return ifelse(iszero(x) && isnan(y), zero(result), result)
end

_dirichlet_term_2070(xi, αi) = _myxlogy_2070(αi - 1, xi)

function _dirichlet_lpdf_2070(d::ImageDirichlet2070, x::AbstractMatrix)
    return mapreduce(_dirichlet_term_2070, +, x, d.α) - d.lmnB
end

@testset "Untraced host Fill / OneElement arguments (#2223, #2070)" begin
    a = collect(Float64, 1:8)
    ar = Reactant.to_rarray(copy(a))
    f_fill = Fill(0.0, 4)
    res_fill = @jit _setindex_fill!(ar, 1:4, f_fill)
    @test Array(res_fill) == _setindex_fill!(copy(a), 1:4, f_fill)
    @test Array(ar) == _setindex_fill!(copy(a), 1:4, f_fill)
    @test f_fill.value === 0.0

    ar2 = Reactant.to_rarray(copy(a))
    f_one = OneElement(9.0, (2,), (4,))
    res_one = @jit _setindex_fill!(ar2, 1:4, f_one)
    @test Array(res_one) == _setindex_fill!(copy(a), 1:4, f_one)
    @test f_one.val === 9.0

    xr_f32 = Reactant.to_rarray(Float32[1, 2, 3, 4])
    ret_f = @jit return_fill_arg(xr_f32, Fill(3.0f0, 4))
    @test Array(ret_f) == fill(3.0f0, 4)
    ret_c = @jit make_fill_const(xr_f32)
    @test Array(ret_c) == fill(2.0f0, 4)

    cr = ConcreteRNumber(1)
    @test Array(@jit _if_fill(cr, xr_f32, Fill(0.5f0, 4))) ≈ Float32[1.5, 2.5, 3.5, 4.5]
    @test Array(@jit _while_fill(xr_f32, Fill(0.5f0, 4))) ≈ Float32[2.0, 3.0, 4.0, 5.0]

    d = ImageDirichlet2070(1.0, 4, 4)
    x = reshape(collect(Float64, 1:16), 4, 4)
    x ./= sum(x)
    xr = Reactant.to_rarray(x)
    @test @jit(_dirichlet_lpdf_2070(d, xr)) ≈ _dirichlet_lpdf_2070(d, x)
end
