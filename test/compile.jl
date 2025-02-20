using Reactant
using Test

Base.sum(x::NamedTuple{(:a,),Tuple{T}}) where {T<:Reactant.TracedRArray} = (; a=sum(x.a))

@testset "compile" begin
    @testset "create_result" begin
        @testset "NamedTuple" begin
            x = (; a=rand(4, 3))
            x2 = Reactant.to_rarray(x)

            res = @jit sum(x2)
            @test res isa
                @NamedTuple{a::Reactant.ConcretePJRTNumber{Float64,1,Sharding.NoShardInfo}}
            @test isapprox(res.a, sum(x.a))
        end

        @testset "Array" begin
            x = [1 2; 3 4; 5 6]
            x2 = Reactant.to_rarray(x)

            # TODO remove `x2` when #196 is fixed
            f = Reactant.compile((x2,)) do _
                x
            end

            @test f(x2) ≈ x
        end
    end

    @testset "world-age" begin
        a = ones(2, 10)
        b = ones(10, 2)
        a_ra = Reactant.ConcretePJRTArray(a)
        b_ra = Reactant.ConcretePJRTArray(b)

        fworld(x, y) = @jit(x * y)

        @test fworld(a_ra, b_ra) ≈ ones(2, 2) * 10
    end

    @testset "type casting & optimized out returns" begin
        a = ones(2, 10)
        a_ra = Reactant.ConcretePJRTArray(a)

        ftype1(x) = Float64.(x)
        ftype2(x) = Float32.(x)

        y1 = @jit ftype1(a_ra)
        y2 = @jit ftype2(a_ra)

        @test y1 isa Reactant.ConcretePJRTArray{Float64,2}
        @test y2 isa Reactant.ConcretePJRTArray{Float32,2}

        @test y1 ≈ Float64.(a)
        @test y2 ≈ Float32.(a)
    end

    @testset "no variable name collisions in compile macros (#237)" begin
        f(x) = x
        g(x) = f(x)
        x = rand(2, 2)
        y = Reactant.to_rarray(x)
        @test (@jit g(y); true)
    end

    # disabled due to long test time (core tests go from 2m to 7m just with this test)
    # @testset "resource exhaustation bug (#190)" begin
    #     x = rand(2, 2)
    #     y = Reactant.to_rarray(x)
    #     @test try
    #         for _ in 1:10_000
    #             f = @compile sum(y)
    #         end
    #         true
    #     catch e
    #         false
    #     end
    # end
end

@testset "Module export" begin
    f(x) = sin.(cos.(x))
    x_ra = Reactant.to_rarray(rand(3))

    hlo_code = @code_hlo f(x_ra)
    @test !startswith(string(hlo_code), "Module")
    @test startswith(string(hlo_code), "module")
end

@testset "Bool attributes" begin
    x_ra = Reactant.to_rarray(false; track_numbers=Number)
    @test @jit(iszero(x_ra)) == true
    x_ra = Reactant.to_rarray(true; track_numbers=Number)
    @test @jit(iszero(x_ra)) == false
end

@testset "Vararg compilation: Issue #293" begin
    x = rand(2, 2)
    x_ra = Reactant.to_rarray(x)

    @test @allowscalar(x_ra[1]) ≈ x[1]
    @test @allowscalar(x_ra[1:1]) ≈ x[1:1]
end

@testset "no_nan passes" begin
    x_ra = Reactant.to_rarray(rand(Float32, 4, 16))
    y_ra = Reactant.to_rarray(rand(Float32, 4, 16))

    fn(x) = x .- x

    hlo = @code_hlo fn(x_ra)
    @test occursin("subtract", repr(hlo))
    @test !occursin("constant", repr(hlo))
    hlo = @code_hlo no_nan = true fn(x_ra)
    @test !occursin("subtract", repr(hlo))
    @test occursin("constant", repr(hlo))

    fn(x, y) = begin
        c = x .+ y
        return c .- y
    end

    hlo = @code_hlo fn(x_ra, y_ra)
    @test occursin("subtract", repr(hlo))
    @test occursin("add", repr(hlo))
    hlo = @code_hlo no_nan = true fn(x_ra, y_ra)
    @test !occursin("subtract", repr(hlo))
    @test !occursin("add", repr(hlo))
end

# While a bit specific, the following is used to check for a bug in `should_rewrite_call`
function sinusoidal_embedding(
    x::AbstractArray{T,4}, min_freq, max_freq, embedding_dims::Int
) where {T}
    if size(x)[1:3] != (1, 1, 1)
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    lower, upper = log(T(min_freq)), log(T(max_freq))
    n = embedding_dims ÷ 2
    x_ = 2 .* x .* exp.(reshape(range(lower, upper; length=n), 1, 1, n, 1))
    return cat(sinpi.(x_), cospi.(x_); dims=Val(3))
end

@testset "sinusoidal_embedding" begin
    x_ra = Reactant.to_rarray(rand(Float32, 1, 1, 1, 4))
    hlo = @code_hlo sinusoidal_embedding(x_ra, 0.1, 10.0, 4)
end

# test #493
@testset "unique(::Vector{Symbol}) (#493)" begin
    x = [:a, :b, :a]
    @test @jit(unique(x)) == [:a, :b]
end
