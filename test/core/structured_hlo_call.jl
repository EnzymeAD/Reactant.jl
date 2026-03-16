using Reactant, Test
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
        Reactant.to_rarray(ones(Float32, 4, 8)),
        Reactant.to_rarray(ones(Float32, 4)),
    )
    dec = Inner(
        Reactant.to_rarray(ones(Float32, 2, 4)),
        Reactant.to_rarray(ones(Float32, 2)),
    )
    model = Model(enc, dec)
    x = Reactant.to_rarray(ones(Float32, 8, 3))

    @testset "no reactant.path without the option" begin
        hlo = @code_hlo forward(model, x)
        @test !contains(sprint(show, hlo), "reactant.path")
    end

    @testset "reactant.path present with the option" begin
        hlo = @code_hlo store_args_res_path = true forward(model, x)
        ir = sprint(show, hlo)

        @test contains(ir, "reactant.path")

        # Paths are rooted at :args / :result; repr(sym) keeps the leading colon
        @test contains(ir, "\":args\"")
        @test contains(ir, "\":result\"")

        # Struct fields are navigated by integer index (from make_tracer);
        # the nested model has paths like [":args", 1, 1, 1] (model.encoder.weight)
        # and [":args", 1, 1, 2] (model.encoder.bias), etc.
        @test contains(ir, "1, 1, 1")  # first field of first field of first arg (encoder.weight)
        @test contains(ir, "1, 2, 1")  # first field of second field of first arg (decoder.weight)
    end

    @testset "structured_hlo_call round-trip" begin
        # Compile with path metadata; do_transpose=false so shapes in the IR match
        # the Julia-shaped TracedRArrays that hlo_call receives during @jit tracing.
        ir = sprint(show, @code_hlo do_transpose=false store_args_res_path = true forward(model, x))

        # Replay via structured_hlo_call: it should auto-extract the 5 leaf arrays
        # (encoder.weight, encoder.bias, decoder.weight, decoder.bias, x) and return
        # the same result as calling forward directly.
        result_direct = @jit forward(model, x)
        result_structured = @jit Reactant.Ops.structured_hlo_call(ir, model, x)

        @test result_structured ≈ result_direct
    end

    @testset "structured_hlo_call with mutation" begin
        # Mutates a struct field and returns it; structured_hlo_call must navigate
        # layer.weight (path :args, 1, :weight) and layer.bias (path :args, 1, :bias)
        # to correctly linearize the Inner struct argument.
        function add_to_weight!(layer::Inner, delta::AbstractMatrix)
            layer.weight .+= delta
            return layer.weight
        end

        layer1 = Inner(
            Reactant.to_rarray(ones(Float32, 3, 4)),
            Reactant.to_rarray(ones(Float32, 3)),
        )
        delta = Reactant.to_rarray(ones(Float32, 3, 4))
        ir = sprint(show, @code_hlo do_transpose=false store_args_res_path = true add_to_weight!(layer1, delta))

        # Struct fields are navigated by integer index; Inner has two fields
        # so weight=field 1, bias=field 2 → paths like [":args", 1, 1] and [":args", 1, 2]
        @test contains(ir, "\":args\", 1, 1")   # layer.weight (field 1 of arg 1)
        @test contains(ir, "\":args\", 1, 2")   # layer.bias   (field 2 of arg 1)

        # Round-trip: structured_hlo_call navigates layer.weight / layer.bias correctly
        layer2 = Inner(
            Reactant.to_rarray(ones(Float32, 3, 4)),
            Reactant.to_rarray(ones(Float32, 3)),
        )
        layer3 = Inner(
            Reactant.to_rarray(ones(Float32, 3, 4)),
            Reactant.to_rarray(ones(Float32, 3)),
        )

        result_direct = @jit add_to_weight!(layer2, delta)
        result_structured = @jit Reactant.Ops.structured_hlo_call(ir, layer3, delta)

        @test result_structured ≈ result_direct
    end
end
