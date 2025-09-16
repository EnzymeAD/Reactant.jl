using Reactant, Test, Aqua, ExplicitImports, MethodAnalysis

function get_all_dialects()
    mods = Module[]
    visit(Reactant.MLIR.Dialects) do obj
        if isa(obj, Module)
            push!(mods, obj)
            return true     # descend into submodules
        end
        false   # but don't descend into anything else (MethodTables, etc.)
    end
    return mods
end

@testset "Aqua" begin
    @testset "Ambiguities" begin
        @test_broken Aqua.test_ambiguities(
            Reactant;
            exclude=[
                Base.mapreducedim!,
                Base.:(==),
                Base.unsafe_convert,
                Base.replace,
                # These are not really ambiguous, Test is just weird about them
                Base.rem,
                Base.mod,
                Base.mod1,
            ],
        )
    end
    @testset "Undefined Exports" begin
        Aqua.test_undefined_exports(Reactant)
    end
    @testset "Unbound Args" begin
        Aqua.test_unbound_args(Reactant)
    end
    @testset "Project Extras" begin
        Aqua.test_project_extras(Reactant)
    end
    @testset "Stale Deps" begin
        Aqua.test_stale_deps(Reactant)
    end
    @testset "Deps Compat" begin
        Aqua.test_deps_compat(Reactant)
    end
    @testset "Piracies" begin
        Aqua.test_piracies(
            Reactant;
            treat_as_own=(
                Reactant.ReactantCore.MissingTracedValue,
                Reactant.ReactantCore.promote_to_traced,
                Reactant.ReactantCore.traced_call,
                Reactant.ReactantCore.traced_while,
            ),
        )
    end
    @testset "Undocumented Names" begin
        # TODO: Write more documentation!
        Aqua.test_undocumented_names(Reactant; broken=true)
    end
end

@testset "ExplicitImports" begin
    @testset "Explicit Imports" begin
        @test check_no_implicit_imports(
            Reactant;
            allow_unanalyzable=(
                Reactant.DotGeneralAlgorithmPreset,
                Reactant.MLIR.Dialects,
                get_all_dialects()...,
                Reactant.XLA.OpShardingType,
                Reactant.Accelerators.TPU.TPUVersion,
                Reactant.PrecisionConfig,
            ),
        ) === nothing
    end
    @testset "Import via Owner" begin
        @test check_all_explicit_imports_via_owners(Reactant) === nothing
    end
    @testset "Stale Explicit Imports" begin
        @test check_no_stale_explicit_imports(
            Reactant;
            allow_unanalyzable=(
                Reactant.DotGeneralAlgorithmPreset,
                Reactant.MLIR.Dialects,
                get_all_dialects()...,
                Reactant.XLA.OpShardingType,
                Reactant.Accelerators.TPU.TPUVersion,
                Reactant.PrecisionConfig,
            ),
            ignore=(:unzip, :ShardyPropagationOptions),
        ) === nothing
    end
    @testset "Qualified Accesses" begin
        @test check_all_qualified_accesses_via_owners(Reactant) === nothing
    end
    @testset "Self Qualified Accesses" begin
        @test check_no_self_qualified_accesses(
            Reactant;
            ignore=(
                :REACTANT_METHOD_TABLE,
                :__skip_rewrite_func_set,
                :__skip_rewrite_func_set_lock,
                :__skip_rewrite_type_constructor_list,
                :__skip_rewrite_type_constructor_list_lock,
            ),
        ) === nothing
    end
end
