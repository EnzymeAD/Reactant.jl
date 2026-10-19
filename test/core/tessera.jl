using Reactant, Test
using Reactant: MLIR

@tessera_op "tessera.pow" function tessera_power(x, n)
    return x .^ n
end

@tessera_op "tessera.mul" function tessera_mul(x, y)
    return x .* y
end

@tessera_op "tessera.inv(x)" pure = false function tessera_inv(x)
    return 1 ./ x
end

@tessera_op function tessera_unnamed(x)
    return sin.(x)
end

@tessera_op "tessera.bad(x, y)" function tessera_bad_signature(x)
    return x
end

@tessera_op "tessera.add!" pure = false function tessera_add!(x, y)
    x .+= y
    return nothing
end

@tessera_optimize "tessera.pow(x, 2) -> tessera.mul(x, x)"
@tessera_optimize "tessera.inv(tessera.inv(x)) -> x"

double_inv(x) = tessera_inv(tessera_inv(x))
plain(x) = x .+ 1
# `tessera.mul`, which the rule rewrites this to, is not called by the program itself
square(x) = tessera_power(x, Reactant.Ops.constant(Int32(2)))
mul_self(x) = tessera_mul(x, x)
add_self!(x) = (tessera_add!(x, x); x)

unopt_hlo(f, args...) = repr(@code_hlo optimize = false f(args...))

@testset "Tessera annotations" begin
    x = Reactant.to_rarray(Float32[1, 2, 3])
    n = Reactant.to_rarray(Float32[2, 2, 2])

    @testset "tessera_op attribute" begin
        hlo = unopt_hlo(tessera_power, x, n)
        @test occursin("pure_tessera_op = \"tessera.pow(x, n)\"", hlo)
        @test occursin(r"call @\"#+tessera_powerinternal", hlo)

        hlo = unopt_hlo(tessera_unnamed, x)
        @test occursin("pure_tessera_op = \"tessera_unnamed(x)\"", hlo)

        hlo = unopt_hlo(double_inv, x)
        @test occursin("tessera_op = \"tessera.inv(x)\"", hlo)
        @test !occursin("pure_tessera_op", hlo)
        # both calls reuse the same annotated function
        @test count("tessera_op = ", hlo) == 1
        @test count(r"= call @", hlo) == 2

        @test_throws ErrorException unopt_hlo(tessera_bad_signature, x)
    end

    @testset "rewrite rules" begin
        hlo = unopt_hlo(tessera_power, x, n)
        @test occursin("tessera.optimizations", hlo)
        @test occursin(
            "tessera.optimization \"tessera.pow(x, 2) -> tessera.mul(x, x)\"", hlo
        )
        @test occursin("tessera.optimization \"tessera.inv(tessera.inv(x)) -> x\"", hlo)
        @test [
            "tessera.pow(x, 2) -> tessera.mul(x, x)", "tessera.inv(tessera.inv(x)) -> x"
        ] ⊆ Reactant.Tessera.rules()

        # rules are only emitted into modules that use tessera ops
        @test !occursin("tessera", unopt_hlo(plain, x))

        # the emitted rules are consumed by the Enzyme-JAX tessera passes
        MLIR.IR.@with_context Reactant.ReactantContext() begin
            mod = parse(MLIR.IR.Module, hlo)
            Reactant.Compiler.run_pass_pipeline!(mod, "parse-optimization-rules")
            @test occursin("pdl.pattern", repr(mod))
        end
    end

    @testset "applying rules" begin
        # the inner call's result is the outer call's only operand
        hlo = unopt_hlo(double_inv, x)
        @test count(r"call @\S+\(%\d+\) : \(tensor<3xf32>\) -> tensor<3xf32>", hlo) == 2

        hlo = repr(@code_hlo double_inv(x))
        @test !occursin("tessera", hlo)
        @test !occursin("divide", hlo)

        hlo = repr(@code_hlo square(x))
        @test !occursin("tessera", hlo)
        @test !occursin("power", hlo)
        @test occursin("multiply", hlo)
        @test Array(@jit(square(x))) ≈ Float32[1, 4, 9]
    end

    @testset "aliased arguments" begin
        # an argument passed twice is still two operands
        hlo = unopt_hlo(mul_self, x)
        @test occursin(r"call @\S+\((%\d+), \1\) :", hlo)
        @test occursin("pure_tessera_op = \"tessera.mul(x, y)\"", hlo)
        @test Array(@jit(mul_self(x))) ≈ Float32[1, 4, 9]

        # a mutation through one alias reaches the original argument
        y = Reactant.to_rarray(Float32[1, 2, 3])
        @test Array(@jit(add_self!(y))) ≈ Float32[2, 4, 6]
        @test Array(y) ≈ Float32[2, 4, 6]
    end

    @testset "execution" begin
        # annotations are stripped before the module is handed to XLA
        @test Array(@jit(tessera_power(x, n))) ≈ Float32[1, 4, 9]
        @test Array(@jit(double_inv(x))) ≈ Float32[1, 2, 3]
        @test tessera_power(Float32[1, 2, 3], Float32[2, 2, 2]) ≈ Float32[1, 4, 9]
    end
end
