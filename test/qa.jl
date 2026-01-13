using Reactant, Test, Aqua, ExplicitImports, MethodAnalysis

function get_all_submodules(base_module::Module)
    mods = Module[]
    visit(base_module) do obj
        if isa(obj, Module)
            push!(mods, obj)
            return true     # descend into submodules
        end
        false   # but don't descend into anything else (MethodTables, etc.)
    end
    return mods
end

function issubmodule(mod::Module, parent::Module)
    while parentmodule(mod) !== mod
        if parentmodule(mod) === parent
            return true
        end
        mod = parentmodule(mod)
    end
    return false
end

@testset "Aqua" begin
    @testset "Ambiguities" begin
        Aqua.test_ambiguities(
            Reactant;
            exclude=[
                Base.unsafe_convert,
                Base.replace,
                # These are not really ambiguous, Test is just weird about them
                Base.rem,
                Base.mod,
                Base.mod1,
                # The below ones are a bit tricky to get rid off. maybe we overlap them
                Base.mapreducedim!,
                Base.:(==),
                Base.copyto!,
                Base.fill!,
            ],
        )
    end
    @testset "Undefined Exports" begin
        Aqua.test_undefined_exports(Reactant)
    end
    @testset "Unbound Args" begin
        methods_with_unbound_args = Aqua.detect_unbound_args_recursively(Reactant)
        num_unbound_args = 0
        for method in methods_with_unbound_args
            if !issubmodule(parentmodule(method), Reactant.Proto)
                num_unbound_args += 1
                @warn "Method $(method) has unbound args"
            end
        end
        @test num_unbound_args == 0
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
                get_all_submodules(Reactant.MLIR.Dialects)...,
                get_all_submodules(Reactant.Proto)...,
                Reactant.XLA.OpShardingType,
                Reactant.Accelerators.TPU.TPUVersion,
                Reactant.PrecisionConfig,
            ),
            ignore=(Reactant.Proto,),
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
                get_all_submodules(Reactant.MLIR.Dialects)...,
                get_all_submodules(Reactant.Proto)...,
                Reactant.XLA.OpShardingType,
                Reactant.Accelerators.TPU.TPUVersion,
                Reactant.PrecisionConfig,
            ),
            # OneOf is used inside Proto files
            ignore=(:p7zip, :ShardyPropagationOptions, :OneOf),
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
