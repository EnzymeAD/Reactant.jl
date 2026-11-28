# How much of one of Reactant's own test files (`control_flow` by default, or
# `autodiff`) passes with `@trace` on control flow made an identity macro and
# every compilation routed through `structured`? Not part of the test suite:
#     julia --project=lib/StructuredReactant/test lib/StructuredReactant/test/migration.jl [autodiff]
using Reactant, StructuredReactant, Test
using StructuredReactant: FrontendError

const compile_calls = Ref(0)

# Route user functions, callbacks included; leave Base and Reactant ones alone.
function route(f)
    f isa Base.Broadcast.BroadcastFunction &&
        return Base.Broadcast.BroadcastFunction(route(f.f))
    f isa Function || return f
    root = Base.moduleroot(parentmodule(typeof(f)))
    root in (Base, Core, Reactant) && return f
    return structured(f)
end

# The overrides are more specific than Reactant's methods (`args::Tuple`), so
# `invoke` on the general signature reaches the originals in the current world.
for entry in (:compile, :code_hlo, :code_mhlo, :code_xla)
    @eval function Reactant.Compiler.$entry(ctx, f, args::Tuple; kwargs...)
        compile_calls[] += 1
        result = Base.invoke(
            Reactant.Compiler.$entry,
            Tuple{Any,Any,Any},
            ctx,
            route(f),
            map(route, args);
            kwargs...,
        )
        # A compiled function's signature holds the routed function-valued
        # arguments, so calls made with the bare functions are routed the same way.
        return $(entry === :compile) ? (a...) -> result(map(route, a)...) : result
    end
end

# A bare `@trace` on control flow becomes an identity; its options and the call
# outlining forms are hints the frontend does not replace, and are kept.
const TRACE_WORLD = Base.get_world_counter()   # the original macro lives here
@eval Reactant.ReactantCore macro trace(args...)
    if length(args) == 1 && Meta.isexpr(only(args), (:if, :for, :while))
        return esc(only(args))
    end
    return Base.invoke_in_world($TRACE_WORLD, var"@trace", __source__, __module__, args...)
end

file = isempty(ARGS) ? "control_flow" : only(ARGS)
source = normpath(joinpath(@__DIR__, "..", "..", "..", "test", "core", file * ".jl"))

# Julia 1.13 keeps the active testset in a scoped value instead of a stack.
function within(f, ts::Test.AbstractTestSet)
    @static if isdefined(Test, :push_testset)
        Test.push_testset(ts)
        try
            return f()
        finally
            Test.pop_testset()
        end
    else
        return Base.ScopedValues.with(
            f, Test.CURRENT_TESTSET => ts, Test.TESTSET_DEPTH => 1
        )
    end
end

suite = Test.DefaultTestSet("Reactant $(file).jl through structured"; verbose=false)
within(suite) do
    try
        Base.include(Module(:Upstream), source)
    catch err
        println("The file itself failed to load: ", sprint(showerror, err))
    end
end

function classify(result)
    result isa Test.Fail && return "mismatch"
    message = sprint(show, result)
    m = match(r"StructuredReactant: ([^\n]*)", message)
    m !== nothing && return "unsupported: " * m.captures[1]
    m = match(
        r"(?:Test threw exception|Got exception outside of a @test)\n\s*([^\n]*)", message
    )
    return "error: " * (m === nothing ? "?" : m.captures[1])
end

function collect!(counts, examples, ts, path)
    for r in ts.results
        if r isa Test.AbstractTestSet
            collect!(counts, examples, r, path * " / " * r.description)
        elseif r isa Test.Pass
            counts["pass"] = get(counts, "pass", 0) + 1
        elseif r isa Test.Broken
            counts["broken"] = get(counts, "broken", 0) + 1
        else
            kind = classify(r)
            counts[kind] = get(counts, kind, 0) + 1
            push!(get!(Vector{String}, examples, kind), path)
        end
    end
end

counts = Dict{String,Int}()
examples = Dict{String,Vector{String}}()
collect!(counts, examples, suite, "")
counts["pass"] = Test.get_test_counts(suite).cumulative_passes   # passes are not stored

println("\nCompilation requests routed through `structured`: ", compile_calls[])
println("Results:")
for (kind, n) in sort!(collect(counts); by=last, rev=true)
    println(lpad(n, 5), "  ", kind)
    kind in ("pass", "broken") && continue
    for testset in unique(examples[kind])[1:min(end, 4)]
        println("       e.g. ", strip(testset, [' ', '/']))
    end
end
