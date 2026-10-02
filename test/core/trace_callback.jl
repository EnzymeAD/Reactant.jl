using Reactant, Test

function record_trace(f, args...)
    recorded = Ref{Any}(nothing)
    world = Ref{UInt}(0)
    Reactant.compile_with_trace_callback(f, args...) do trace_world, codes
        world[] = trace_world
        recorded[] = codes
    end
    return world[], recorded[]
end

traced_mi(code::Core.CodeInstance) = code.def
traced_mi(code::Core.MethodInstance) = code

function traced_method_names(codes)
    names = Set{Symbol}()
    for code in codes
        mi = traced_mi(code)
        mi isa Core.MethodInstance || continue
        mi.def isa Method && push!(names, mi.def.name)
    end
    return names
end

inlined_callee(x) = x .* 2
@inline inlined_caller(x) = inlined_callee(x) .+ 1
outer_inlined(x) = inlined_caller(x)

@noinline noinline_callee(x) = x .* 2
noinline_caller(x) = noinline_callee(x) .+ 1

stable_fn(x) = x .- 1

toplevel_fn(x) = x .+ 1

@testset "compile_with_trace_callback" begin
    x = Reactant.to_rarray(ones(Float32, 4))

    @testset "records the traced code" begin
        world, codes = record_trace(noinline_caller, x)
        @test world isa UInt
        @test !isempty(codes)
        @test any(c -> c isa Core.CodeInstance, codes)
        @test :noinline_caller in traced_method_names(codes)
        @test isempty(Reactant.invalidated_traced_methods(codes, world))
    end

    @testset "second compile records the same code" begin
        _, first = record_trace(stable_fn, x)
        _, second = record_trace(stable_fn, x)
        @test traced_method_names(first) == traced_method_names(second)
        @test length(first) == length(second)
    end

    @testset "unrelated definitions do not invalidate" begin
        world, codes = record_trace(stable_fn, x)
        @eval unrelated_new_function(y) = y
        @test isempty(Reactant.invalidated_traced_methods(codes, world))
    end

    @testset "redefining the traced function invalidates" begin
        world, codes = record_trace(toplevel_fn, x)
        @test isempty(Reactant.invalidated_traced_methods(codes, world))
        @eval toplevel_fn(x) = x .+ 2
        invalidated = Reactant.invalidated_traced_methods(codes, world)
        @test :toplevel_fn in traced_method_names(invalidated)
    end

    @testset "redefining a non-inlined callee invalidates" begin
        world, codes = record_trace(noinline_caller, x)
        @test isempty(Reactant.invalidated_traced_methods(codes, world))
        @eval @noinline noinline_callee(x) = x .* 3
        invalidated = Reactant.invalidated_traced_methods(codes, world)
        @test !isempty(invalidated)
        @test :noinline_callee in traced_method_names(invalidated)
    end

    @testset "redefining an inlined method invalidates its caller" begin
        # `inlined_caller` is inlined into `outer_inlined` before the tracer
        # rewrites calls, so it never goes through a wrapper of its own and is
        # only detectable through `outer_inlined`'s `CodeInstance`.
        world, codes = record_trace(outer_inlined, x)
        @test :inlined_caller ∉ traced_method_names(codes)
        @test isempty(Reactant.invalidated_traced_methods(codes, world))
        @eval @inline inlined_caller(x) = inlined_callee(x) .+ 5
        invalidated = Reactant.invalidated_traced_methods(codes, world)
        @test :outer_inlined in traced_method_names(invalidated)
    end

    @testset "retracing after invalidation is clean" begin
        world, codes = record_trace(outer_inlined, x)
        @test isempty(Reactant.invalidated_traced_methods(codes, world))
        f = Reactant.compile(outer_inlined, (x,))
        @test Array(f(x)) == fill(7.0f0, 4)
    end
end
