using Reactant
using Test

Base.sum(x::NamedTuple{(:a,),Tuple{T}}) where {T<:Reactant.TracedRArray} = (; a=sum(x.a))

@testset "compile" begin
    @testset "create_result" begin
        @testset "NamedTuple" begin
            x = (; a=rand(4, 3))
            x2 = Reactant.to_rarray(x)

            f = @compile sum(x2)

            @test f(x2) isa @NamedTuple{a::Reactant.ConcreteRNumber{Float64}}
            @test isapprox(f(x2).a, sum(x.a))
        end
    end

    @testset "world-age" begin
        a = Reactant.ConcreteRArray(ones(2, 10))
        b = Reactant.ConcreteRArray(ones(10, 2))

        function fworld(x, y)
            g = @compile *(a, b)
            return g(x, y)
        end

        @test fworld(a, b) ≈ ones(2, 2) * 10
    end

    @testset "type casting & optimized out returns" begin
        a = Reactant.ConcreteRArray(rand(2, 10))

        ftype1(x) = Float64.(x)
        ftype2(x) = Float32.(x)

        ftype1_compiled = @compile ftype1(a)
        ftype2_compiled = @compile ftype2(a)

        @test ftype1_compiled(a) isa Reactant.ConcreteRArray{Float64,2}
        @test ftype2_compiled(a) isa Reactant.ConcreteRArray{Float32,2}

        @test ftype1_compiled(a) ≈ Float64.(a)
        @test ftype2_compiled(a) ≈ Float32.(a)
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
