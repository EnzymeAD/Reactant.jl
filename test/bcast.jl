using Test
using Reactant
using Enzyme, NNlib
using Reactant.MLIR
using Statistics

@noinline function no(@nospecialize(x))
    x = @ccall $(Base.@cfunction(identity, Any, (Any,)))(x::Any)::Any
    return x[]::Any
end

mutable struct Data
    v::Reactant.TracedRArray{Float64,1}
end
@noinline function tmp(a, b, d)
    @show d
    @show typeof(d)
    c = d.v
    @show typeof(c)

    return reshape(a, (4,)) ./ sqrt.(b .+ a)
end

function test()
    ctx = MLIR.IR.Context()
    Base.append!(Reactant.registry[]; context=ctx)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())
        modbody = MLIR.IR.body(mod)

        in_tys = [MLIR.IR.TensorType([4], MLIR.IR.Type(Float64))]

        func = MLIR.Dialects.func.func_(;
            sym_name="main_tmp",
            function_type=MLIR.IR.FunctionType(in_tys, []),
            body=MLIR.IR.Region(),
        )

        fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for _ in in_tys])
        push!(MLIR.IR.region(func, 1), fnbody)

        GC.@preserve mod func fnbody begin
            MLIR.IR.block!(fnbody) do
                a = ones(4)
                b = ones(4)
                d = Data(
                    Reactant.TracedRArray{Float64,1}((), MLIR.IR.argument(fnbody, 1), (4,))
                )

                return tmp(a, b, d)
            end
        end

        return println(string(mod))
    end
end
test()

@testset "ConcreteRArray broadcasting" begin
    x = ones(10, 10)
    y = ones(10, 10)

    x_ca = Reactant.ConcreteRArray(x)
    y_ca = Reactant.ConcreteRArray(y)

    @testset "Broadcasting" begin
        # TODO make `@compile` work with broadcasting syntax
        @test x .+ y ≈ Reactant.Compiler.compile(.+, (x_ca, y_ca))(x_ca, y_ca)
        @test x .- y ≈ Reactant.Compiler.compile(.-, (x_ca, y_ca))(x_ca, y_ca)
        @test x .* y ≈ Reactant.Compiler.compile(.*, (x_ca, y_ca))(x_ca, y_ca)
        @test x ./ y ≈ Reactant.Compiler.compile(./, (x_ca, y_ca))(x_ca, y_ca)
    end
end

function scalar_bcast(x)
    sc = sum(x)
    return sc .+ x
end

@testset "Scalar broadcasting" begin
    x = rand(2, 3)
    x_ra = Reactant.to_rarray(x)

    scalar_bcast_compiled = @compile scalar_bcast(x_ra)

    @test scalar_bcast_compiled(x_ra) ≈ scalar_bcast(x)
end

function custom_ln(x)
    mu = mean(x; dims=1)
    sigma = var(x; dims=3)
    return (x .- mu) ./ sqrt.(sigma)
end

@testset "Custom layernorm" begin
    x = rand(Float32, 3, 3, 4, 2)
    x_ra = Reactant.to_rarray(x)

    ln_comp = @compile custom_ln(x_ra)
    @test ln_comp(x_ra) ≈ custom_ln(x)
end

pow(x, n) = x .^ n

@testset "Pow" begin
    x = rand(2, 3)
    x_ra = Reactant.to_rarray(x)

    pow_compiled = @compile pow(x_ra, 2)

    @test pow_compiled(x_ra) ≈ pow(x, 2)
end

struct CustomBCastFunction{X}
    x::X
end

(f::CustomBCastFunction)(x::Number) = f.x + x

function custombcast(x)
    fn = CustomBCastFunction(3.0)
    return fn.(x)
end

@testset "Broadcasting closures / functors" begin
    x = rand(2, 3)
    x_ra = Reactant.to_rarray(x)

    custombcast_compiled = @compile custombcast(x_ra)

    @test custombcast_compiled(x_ra) ≈ custombcast(x)
end
