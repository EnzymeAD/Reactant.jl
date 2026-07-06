using Reactant, Test
using Reactant: ProbProg

# Pins the hand-written VI enum literals in VI.jl against ImpulseEnums.td.

@testset "VI enum guard" begin
    # Verbatim from ImpulseEnums.td `def GuideKind`.
    guide_ods = Dict(
        "AUTO_NORMAL" => Int32(0),
        "AUTO_MULTIVARIATE_NORMAL" => Int32(1),
        "CUSTOM" => Int32(2),
        "AUTO_DELTA" => Int32(3),
    )
    # Verbatim from ImpulseEnums.td `def OptimizerKind`.
    optimizer_ods = Dict(
        "ADAM" => Int32(0),
        "SGD" => Int32(1),
        "CLIPPED_ADAM" => Int32(2),
        "MOMENTUM" => Int32(3),
        "RMSPROP" => Int32(4),
        "ADAGRAD" => Int32(5),
        "RMSPROP_MOMENTUM" => Int32(6),
    )

    @testset "GuideKind" begin
        @test ProbProg.GUIDE_KIND_AUTO_NORMAL === guide_ods["AUTO_NORMAL"]
        @test ProbProg.GUIDE_KIND_AUTO_MVN === guide_ods["AUTO_MULTIVARIATE_NORMAL"]
        @test ProbProg.GUIDE_KIND_CUSTOM === guide_ods["CUSTOM"]
        @test ProbProg.GUIDE_KIND_AUTO_DELTA === guide_ods["AUTO_DELTA"]
    end

    @testset "OptimizerKind" begin
        @test ProbProg.OPTIMIZER_KIND_ADAM === optimizer_ods["ADAM"]
        @test ProbProg.OPTIMIZER_KIND_SGD === optimizer_ods["SGD"]
        @test ProbProg.OPTIMIZER_KIND_CLIPPED_ADAM === optimizer_ods["CLIPPED_ADAM"]
        @test ProbProg.OPTIMIZER_KIND_MOMENTUM === optimizer_ods["MOMENTUM"]
        @test ProbProg.OPTIMIZER_KIND_RMSPROP === optimizer_ods["RMSPROP"]
        @test ProbProg.OPTIMIZER_KIND_ADAGRAD === optimizer_ods["ADAGRAD"]
        @test ProbProg.OPTIMIZER_KIND_RMSPROP_MOMENTUM === optimizer_ods["RMSPROP_MOMENTUM"]
    end

    @testset "Int32 storage type" begin
        for v in (
            ProbProg.GUIDE_KIND_AUTO_NORMAL,
            ProbProg.GUIDE_KIND_AUTO_MVN,
            ProbProg.GUIDE_KIND_CUSTOM,
            ProbProg.GUIDE_KIND_AUTO_DELTA,
            ProbProg.OPTIMIZER_KIND_ADAM,
            ProbProg.OPTIMIZER_KIND_SGD,
            ProbProg.OPTIMIZER_KIND_CLIPPED_ADAM,
            ProbProg.OPTIMIZER_KIND_MOMENTUM,
            ProbProg.OPTIMIZER_KIND_RMSPROP,
            ProbProg.OPTIMIZER_KIND_ADAGRAD,
            ProbProg.OPTIMIZER_KIND_RMSPROP_MOMENTUM,
        )
            @test v isa Int32
        end
    end
end
