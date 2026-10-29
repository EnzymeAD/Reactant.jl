using Reactant, Enzyme, Test, FileCheck

const AD_COMPILER = Reactant.Compiler
const AD_FIXTURES = joinpath(@__DIR__, "..", "fixtures", "ad_optimizations")
ad_fixture(name) = read(joinpath(AD_FIXTURES, name * ".mlir"), String)
ad_ir(source, passes) = repr(AD_COMPILER.run_pass_pipeline_on_source(source, passes))
ad_call(source, args...) = Reactant.Ops.hlo_call(source, args...)

function ad_execute(source, args...; options=CompileOptions())
    inputs = Reactant.to_rarray.(args; track_numbers=true)
    return @jit compile_options = options ad_call(source, inputs...)
end

function ad_lower(source, options; prefix=String[])
    return ad_ir(
        source,
        join(
            [
                prefix...,
                AD_COMPILER.ad_pre_enzyme_passes(options)...,
                AD_COMPILER.enzyme_pass,
                "canonicalize",
                "inline",
                "symbol-dce",
            ],
            ',',
        ),
    )
end

@testset "Activity refinement changes batching compatibility" begin
    source = ad_fixture("activity_batching")
    separate = ad_ir(source, "enzyme-diff-batch")
    @test @filecheck implicit_check_not = "width =" begin
        @check_count 2 "enzyme.fwddiff"
        @check_not "enzyme.fwddiff"
        separate
    end
    refined = ad_ir(source, "enzyme-activity-opt,enzyme-diff-batch")
    @test @filecheck begin
        @check "enzyme.fwddiff"
        @check_same "ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_const>]"
        @check_same "width = 2"
        @check_not "enzyme.fwddiff"
        refined
    end
    reverse_source = ad_fixture("reverse_activity_batching")
    @test @filecheck implicit_check_not = "width =" begin
        @check_count 2 "enzyme.autodiff"
        @check_not "enzyme.autodiff"
        ad_ir(reverse_source, "enzyme-diff-batch")
    end
    @test @filecheck begin
        @check "enzyme.autodiff"
        @check_same "ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]"
        @check_same "width = 2"
        @check_not "enzyme.autodiff"
        ad_ir(reverse_source, "enzyme-activity-opt,enzyme-diff-batch")
    end
end

@testset "Region control moves independent work" begin
    source = ad_fixture("region_motion")
    unchanged = ad_ir(source, "canonicalize")
    @test @filecheck begin
        @check_label "func.func @main"
        @check_not "stablehlo.exponential"
        @check "enzyme.fwddiff"
        unchanged
    end
    moved = ad_ir(
        source,
        join(
            AD_COMPILER.ad_pre_enzyme_passes(ADOptimizationOptions(; region_hoist=true)),
            ',',
        ),
    )
    @test @filecheck begin
        @check_label "func.func @main"
        @check "stablehlo.exponential"
        @check "enzyme.fwddiff"
        @check_label "func.func private @main_to_fwddiff"
        @check_not "stablehlo.exponential"
        @check "stablehlo.multiply"
        moved
    end
end

@testset "Original StableHLO capture regression" begin
    source = ad_fixture("capture_loop")
    for hoist in (false, true)
        prefix = if hoist
            ["hoist-enzyme-regions", "outline-enzyme-regions"]
        else
            ["outline-enzyme-regions"]
        end
        lowered = ad_lower(source, ADOptimizationOptions(); prefix)
        @test @filecheck implicit_check_not = ["enzyme.autodiff", "enzyme.fwddiff"] begin
            @check "stablehlo.while"
            lowered
        end
        @test only(ad_execute(lowered)) ≈ 1.8 atol = 1e-12 rtol = 1e-12
    end
end

@testset "All AD option combinations preserve results" begin
    for activity in (false, true),
        diff_batch in (false, true),
        region_hoist in (false, true)

        ad = ADOptimizationOptions(; activity, diff_batch, region_hoist)
        @testset "activity=$activity diff_batch=$diff_batch region_hoist=$region_hoist" begin
            # Explicit AD fixtures verify both primal and derivative result mappings.
            cases = (
                (
                    "forward_mapping",
                    (2.0, 7.0, 5.0, -2.0, 3.0, 4.0, -6.0),
                    (17.0, -2.0, 17.0, 3.0),
                ),
                (
                    "reverse_mapping",
                    (2.0, 7.0, 5.0, -2.0, 3.0),
                    (17.0, -10.0, -4.0, 17.0, 15.0, 6.0),
                ),
                ("activity_batching", (2.0, -2.0, 3.0, 7.0), (-8.0, 7.0, 0.0, 12.0, 7.0)),
                (
                    "reverse_activity_batching",
                    (2.0, 7.0, -2.0, 3.0, 99.0),
                    (4.0, -8.0, 4.0, 12.0),
                ),
                ("region_motion", (2.0, -2.0, 3.0, log(2.0)), (-16.0, 24.0)),
                ("dependent_primals", (2.0, -2.0, 3.0), (-8.0, 12.0, -16.0, 24.0)),
            )
            for (name, inputs, expected) in cases
                lowered = ad_lower(ad_fixture(name), ad)
                actual = ad_execute(lowered, inputs...)
                @test length(actual) == length(expected)
                @test all(isapprox.(actual, expected; atol=1e-12, rtol=1e-12))
            end
            # The capture boundary must survive inline/outline with every setting.
            capture = ad_lower(
                ad_fixture("capture_loop"), ad; prefix=["outline-enzyme-regions"]
            )
            @test only(ad_execute(capture)) ≈ 1.8 atol = 1e-12 rtol = 1e-12
        end
    end
end

ad_product(x, c, y) = sum(x .* y) + c
function ad_julia_directions(x, c, y, dx1, dx2, dy1, dy2)
    d1 = only(
        Enzyme.autodiff(
            Forward, ad_product, Duplicated(x, dx1), Const(c), Duplicated(y, dy1)
        ),
    )
    d2 = only(
        Enzyme.autodiff(
            Forward, ad_product, Duplicated(x, dx2), Const(c), Duplicated(y, dy2)
        ),
    )
    return d1, d2
end

@testset "Production Julia pipeline option combinations" begin
    x, y = [2.0, -3.0], [5.0, 4.0]
    dx1, dx2, dy1, dy2 = [-2.0, 1.0], [3.0, 2.0], [4.0, 2.0], [-6.0, 1.0]
    inputs = Reactant.to_rarray.((x, 7.0, y, dx1, dx2, dy1, dy2))
    expected = (sum(dx1 .* y .+ x .* dy1), sum(dx2 .* y .+ x .* dy2))
    for activity in (false, true),
        diff_batch in (false, true),
        region_hoist in (false, true)

        options = CompileOptions(;
            ad_optimization_passes=ADOptimizationOptions(;
                activity, diff_batch, region_hoist
            ),
        )
        actual = @jit compile_options = options ad_julia_directions(inputs...)
        @test all(isapprox.(actual, expected; atol=1e-12, rtol=1e-12))
    end
end
