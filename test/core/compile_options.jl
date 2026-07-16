using Enzyme, Reactant, Serialization, Test

const Compiler = Reactant.Compiler

function ad_test_compile_options(; kwargs...)
    return CompileOptions(;
        shardy_passes=:none,
        raise_triton_custom_call=false,
        lower_triton=false,
        strip=:none,
        kwargs...,
    )
end

function dumped_pass_pipeline(compile_options::CompileOptions, key::String)
    return mktempdir() do dir
        dump_always = Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[]
        dump_dir = Reactant.MLIR.IR.DUMP_MLIR_DIR[]
        try
            Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
            Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir

            x = Reactant.to_rarray(Float32[1, 2])
            @code_hlo compile_options = compile_options identity(x)

            marker = "_pre_$(key)_pm.mlir"
            dump = only(filter(Base.Fix1(occursin, marker), readdir(dir; join=true)))
            lines = readlines(dump)
            @assert lines[1] == "// Pass pipeline:"
            return strip(last(split(lines[2], "// "; limit=2)))
        finally
            Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = dump_always
            Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dump_dir
        end
    end
end

function pass_count(pipeline::AbstractString, pass::AbstractString)
    pattern = Regex("(?<![A-Za-z0-9_-])$(pass)(?![A-Za-z0-9_-])")
    return length(collect(eachmatch(pattern, pipeline)))
end

const DIFF_BATCH_SOURCE = raw"""
module {
  func.func @square(%x: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.multiply %x, %x : tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  func.func @two_directions(
      %x: tensor<2xf32>, %dx1: tensor<2xf32>, %dx2: tensor<2xf32>
  ) -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = enzyme.fwddiff @square(%x, %dx1) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = enzyme.fwddiff @square(%x, %dx2) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
}
"""

@testset "AD optimization options" begin
    disabled_ad = ADOptimizationOptions()
    enabled_ad = ADOptimizationOptions(; diff_batch=true)

    @test !disabled_ad.diff_batch
    @test enabled_ad.diff_batch
    @test CompileOptions().ad_optimization_passes === false
    @test CompileOptions(; ad_optimization_passes=false).ad_optimization_passes === false
    @test CompileOptions(; ad_optimization_passes=true).ad_optimization_passes === true
    @test CompileOptions(; ad_optimization_passes=enabled_ad).ad_optimization_passes ===
        enabled_ad
    @test !CompileOptions().disable_post_enzyme_hlo_optimization_passes
    @test CompileOptions(;
        disable_post_enzyme_hlo_optimization_passes=true
    ).disable_post_enzyme_hlo_optimization_passes

    keyword_options = CompileOptions(; ad_optimization_passes=enabled_ad)
    positional_options = CompileOptions(
        (getfield(keyword_options, field) for field in fieldnames(CompileOptions))...
    )
    @test positional_options.ad_optimization_passes === enabled_ad

    io = IOBuffer()
    serialize(io, keyword_options)
    seekstart(io)
    @test deserialize(io).ad_optimization_passes == enabled_ad
end

@testset "AD option plumbing" begin
    enabled_ad = ADOptimizationOptions(; diff_batch=true)
    options = CompileOptions(;
        ad_optimization_passes=enabled_ad,
        transpose_propagate=:up,
        reshape_propagate=:down,
        disable_post_enzyme_hlo_optimization_passes=true,
        sync=false,
    )

    reversed = Reactant.__compile_options_with_reversed_propagation(options)
    @test reversed.ad_optimization_passes === enabled_ad
    @test reversed.transpose_propagate === :down
    @test reversed.reshape_propagate === :up

    synced = Reactant.__compile_options_with_updated_sync(options, true)
    @test synced.ad_optimization_passes === enabled_ad
    @test synced.sync
    @test Reactant.__compile_options_with_updated_sync(options, false) === options

    for field in fieldnames(CompileOptions)
        field in (:transpose_propagate, :reshape_propagate) && continue
        @test isequal(getfield(reversed, field), getfield(options, field))
        field === :sync && continue
        @test isequal(getfield(synced, field), getfield(options, field))
    end

    from_kwargs, remaining = Compiler.__get_compile_options_and_kwargs(;
        ad_optimization_passes=enabled_ad, test_only_kwarg=:preserved
    )
    @test from_kwargs.ad_optimization_passes === enabled_ad
    @test remaining[:test_only_kwarg] === :preserved

    supplied, remaining = Compiler.__get_compile_options_and_kwargs(;
        compile_options=options, ad_optimization_passes=false, test_only_kwarg=:preserved
    )
    @test supplied === options
    @test remaining[:test_only_kwarg] === :preserved
end

@testset "AD pre-Enzyme pipeline helper" begin
    disabled = Compiler.ad_pre_enzyme_passes(false)
    explicit_disabled = Compiler.ad_pre_enzyme_passes(ADOptimizationOptions())
    enabled = Compiler.ad_pre_enzyme_passes(true)
    explicit_enabled = Compiler.ad_pre_enzyme_passes(
        ADOptimizationOptions(; diff_batch=true)
    )

    @test isempty(disabled)
    @test isempty(explicit_disabled)
    @test enabled == explicit_enabled
    @test enabled == ["enzyme-diff-batch", "enzyme-batch-to-stablehlo"]
    @test count(==("enzyme-diff-batch"), enabled) == 1
    @test count(==("enzyme-batch-to-stablehlo"), enabled) == 1

    transformed = repr(
        Compiler.run_pass_pipeline_on_source(DIFF_BATCH_SOURCE, join(enabled, ','))
    )
    @test count("enzyme.fwddiff", transformed) == 1
    @test occursin("width = 2", transformed)
    @test occursin("stablehlo.concatenate", transformed)
    @test count("stablehlo.slice", transformed) == 2
    @test !occursin("enzyme.concat", transformed)
    @test !occursin("enzyme.extract", transformed)

    differentiated = repr(
        Compiler.run_pass_pipeline_on_source(
            DIFF_BATCH_SOURCE, join([enabled..., Compiler.enzyme_pass], ',')
        ),
    )
    @test !occursin("enzyme.fwddiff", differentiated)
    @test !occursin("enzyme.concat", differentiated)
    @test !occursin("enzyme.extract", differentiated)
end

@testset "Generated optimization pipelines" begin
    disabled_pipeline = dumped_pass_pipeline(
        ad_test_compile_options(; ad_optimization_passes=false), "all"
    )
    @test pass_count(disabled_pipeline, "enzyme-diff-batch") == 0
    @test pass_count(disabled_pipeline, "enzyme-batch-to-stablehlo") == 0
    @test pass_count(disabled_pipeline, "enzyme-batch") == 1
    @test occursin("enzyme{postpasses=", disabled_pipeline)

    enabled_pipeline = dumped_pass_pipeline(
        ad_test_compile_options(; ad_optimization_passes=true), "all"
    )
    @test pass_count(enabled_pipeline, "enzyme-diff-batch") == 1
    @test pass_count(enabled_pipeline, "enzyme-batch-to-stablehlo") == 1
    @test pass_count(enabled_pipeline, "enzyme-batch") == 1

    post_optimization_disabled_pipeline = dumped_pass_pipeline(
        ad_test_compile_options(;
            ad_optimization_passes=true,
            disable_post_enzyme_hlo_optimization_passes=true,
        ),
        "all",
    )
    @test pass_count(post_optimization_disabled_pipeline, "enzyme-diff-batch") == 1
    @test pass_count(
        post_optimization_disabled_pipeline, "enzyme-batch-to-stablehlo"
    ) == 2
    @test pass_count(post_optimization_disabled_pipeline, "enzyme-batch") == 1
    disabled_hlo_passes = pass_count(
        post_optimization_disabled_pipeline, "enzyme-hlo-generate-td"
    )
    enabled_hlo_passes = pass_count(enabled_pipeline, "enzyme-hlo-generate-td")
    @test enabled_hlo_passes == 2 * disabled_hlo_passes
    @test pass_count(post_optimization_disabled_pipeline, "canonicalize") ==
        pass_count(enabled_pipeline, "canonicalize")
    @test pass_count(post_optimization_disabled_pipeline, "cse") ==
        pass_count(enabled_pipeline, "cse")

    batch_position = first(findfirst("enzyme-batch", enabled_pipeline))
    diff_batch_position = first(findfirst("enzyme-diff-batch", enabled_pipeline))
    legalization_position = first(findfirst("enzyme-batch-to-stablehlo", enabled_pipeline))
    enzyme_position = first(findfirst("enzyme{postpasses=", enabled_pipeline))
    @test batch_position < diff_batch_position < legalization_position < enzyme_position

    for (mode, key) in (
        (:before_enzyme, "before_enzyme"),
        (:after_enzyme, "after_enzyme"),
        (:only_enzyme, "only_enzyme"),
    )
        pipeline = dumped_pass_pipeline(
            ad_test_compile_options(;
                optimization_passes=mode, ad_optimization_passes=true
            ),
            key,
        )
        @test pass_count(pipeline, "enzyme-batch") == 1
        @test pass_count(pipeline, "enzyme-diff-batch") == 1
        @test pass_count(pipeline, "enzyme-batch-to-stablehlo") == 1
        @test occursin("enzyme{postpasses=", pipeline)
    end

    just_batch_pipeline = dumped_pass_pipeline(
        ad_test_compile_options(;
            optimization_passes=:just_batch, ad_optimization_passes=true
        ),
        "enzyme-batch",
    )
    @test pass_count(just_batch_pipeline, "enzyme-batch") == 1
    @test pass_count(just_batch_pipeline, "enzyme-diff-batch") == 0
    @test pass_count(just_batch_pipeline, "enzyme-batch-to-stablehlo") == 0
    @test !occursin("enzyme{postpasses=", just_batch_pipeline)

    custom_pipeline = dumped_pass_pipeline(
        ad_test_compile_options(;
            optimization_passes="canonicalize", ad_optimization_passes=true
        ),
        "custom_pass",
    )
    @test pass_count(custom_pipeline, "canonicalize") == 1
    @test pass_count(custom_pipeline, "enzyme-batch") == 0
    @test pass_count(custom_pipeline, "enzyme-diff-batch") == 0
    @test pass_count(custom_pipeline, "enzyme-batch-to-stablehlo") == 0
    @test !occursin("enzyme{postpasses=", custom_pipeline)
end

ad_options_square(x) = sum(x .* x)
function ad_options_forward(x, dx)
    return only(
        Enzyme.autodiff(Enzyme.Forward, ad_options_square, Enzyme.Duplicated(x, dx))
    )
end

@testset "Core Enzyme with optional optimizations disabled" begin
    x = Float32[1, 2]
    dx = Float32[3, 4]
    x_ra = Reactant.to_rarray(x)
    dx_ra = Reactant.to_rarray(dx)
    options = ad_test_compile_options(; ad_optimization_passes=false)

    derivative = @jit compile_options = options ad_options_forward(x_ra, dx_ra)
    @test derivative ≈ sum(2 .* x .* dx)
end

@testset "Predefined optimization modes" begin
    modes = (
        :all,
        :before_kernel,
        :before_jit,
        :before_raise,
        :no_enzyme,
        :only_enzyme,
        :after_enzyme,
        :before_enzyme,
        :canonicalize,
        :just_batch,
        :none,
        :probprog,
        :noopt,
    )
    x = Reactant.to_rarray(Float32[1, 2])

    for mode in modes
        options = ad_test_compile_options(;
            optimization_passes=mode, ad_optimization_passes=false
        )
        @test options.optimization_passes === mode
        @test repr(@code_hlo compile_options = options identity(x)) isa String
    end
end
