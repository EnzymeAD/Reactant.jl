using Reactant, Test, FileCheck
using Reactant: CompileOptions

struct Inner{W<:AbstractMatrix,B<:AbstractVector}
    weight::W
    bias::B
end

struct Model{E<:Inner,D<:Inner}
    encoder::E
    decoder::D
end

function forward(model::Model, x::AbstractMatrix)
    h = model.encoder.weight * x .+ model.encoder.bias
    return model.decoder.weight * h .+ model.decoder.bias
end

@testset "structured_hlo_call" begin
    enc = Inner(
        Reactant.to_rarray(ones(Float32, 4, 8)), Reactant.to_rarray(ones(Float32, 4))
    )
    dec = Inner(
        Reactant.to_rarray(ones(Float32, 2, 4)), Reactant.to_rarray(ones(Float32, 2))
    )
    model = Model(enc, dec)
    x = Reactant.to_rarray(ones(Float32, 8, 3))

    @testset "no reactant.path without the option" begin
        hlo = @code_hlo forward(model, x)
        @test @filecheck begin
            @check_not "reactant.path"
            repr(hlo)
        end
    end

    @testset "reactant.path present with the option" begin
        hlo = @code_hlo store_args_res_path = true forward(model, x)

        # Paths are rooted at :args / :result; repr(sym) keeps the leading colon.
        # Struct fields are navigated by integer index (from make_tracer):
        #   [":args", 1, 1, 1] → model.encoder.weight
        #   [":args", 1, 2, 1] → model.decoder.weight
        @test @filecheck begin
            @check "reactant.path"
            @check "\":args\""
            @check "\":result\""
            @check "1, 1, 1"
            @check "1, 2, 1"
            repr(hlo)
        end
    end

    @testset "structured_hlo_call round-trip" begin
        ir = sprint(show, @code_hlo store_args_res_path = true forward(model, x))

        result_direct = @jit forward(model, x)
        result_structured = @jit Reactant.Ops.structured_hlo_call(ir, model, x)

        @test result_structured ≈ result_direct
    end

    @testset "structured_hlo_call with mutation" begin
        function add_to_weight!(layer::Inner, delta::AbstractMatrix)
            layer.weight .+= delta
            return layer.weight
        end

        layer1 = Inner(
            Reactant.to_rarray(ones(Float32, 3, 4)), Reactant.to_rarray(ones(Float32, 3))
        )
        delta = Reactant.to_rarray(ones(Float32, 3, 4))
        ir = sprint(
            show, @code_hlo store_args_res_path = true add_to_weight!(layer1, delta)
        )

        # Inner has two fields: weight=field 1, bias=field 2
        @test @filecheck begin
            @check "\":args\", 1, 1"
            @check "\":args\", 1, 2"
            ir
        end

        layer2 = Inner(
            Reactant.to_rarray(ones(Float32, 3, 4)), Reactant.to_rarray(ones(Float32, 3))
        )
        layer3 = Inner(
            Reactant.to_rarray(ones(Float32, 3, 4)), Reactant.to_rarray(ones(Float32, 3))
        )

        result_direct = @jit add_to_weight!(layer2, delta)
        result_structured = @jit Reactant.Ops.structured_hlo_call(ir, layer3, delta)

        @test result_structured ≈ result_direct
    end
end
