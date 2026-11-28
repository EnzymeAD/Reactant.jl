module CacheCases
using Reactant
@noinline child(x) = 1
parent(x) = child(x)
independent(x) = x .+ 3.0f0
dispatch(x) = x
source_user(x) = sum(x) > 0 ? x .* 2 : -x
behind_leaf(x) = 1
leaf_parent(x) = Reactant.structured_cache_test_leaf(x)

@noinline world_child(x) = x .+ 1.0f0
world_parent(x) = world_child(x)
world_branch(x) = sum(x) > 0 ? world_child(x) : -world_child(x)
const entered = Channel{Nothing}(1)
const resume = Channel{Nothing}(1)
@noinline function checkpoint()
    put!(entered, nothing)
    take!(resume)
    return nothing
end
# Synchronization belongs to the test harness, outside Reactant's overlay.
Reactant.@skip_rewrite_func checkpoint
function world_twice(x)
    first = world_child(x)
    checkpoint()
    second = world_child(x)
    return first, second
end
end

@eval Reactant structured_cache_test_leaf(x::TracedRArray) = Main.CacheCases.behind_leaf(x)

@testset "CodeInstance cache" begin
    SR = StructuredReactant
    CC = SR.CC
    A = Reactant.TracedRArray{Float32,1}
    resolve(f) = SR.resolve(Tuple{typeof(f),A}, Base.get_world_counter())

    before = resolve(CacheCases.parent)
    independent = resolve(CacheCases.independent)
    @test before.ci isa Core.CodeInstance
    @eval CacheCases unrelated(x) = x
    reused = resolve(CacheCases.parent)
    @test reused.ci === before.ci
    @test reused.code === before.code

    # RESOLUTIONS only caches dispatch; the prepared code lives on the CI.
    empty!(SR.RESOLUTIONS)
    @test resolve(CacheCases.parent).code === before.code

    @eval CacheCases child(x) = 1.0
    changed = resolve(CacheCases.parent)
    @test changed.ci !== before.ci
    @test changed.code !== before.code
    @test changed.ci.rettype === Float64
    @test before.ci.max_world < Base.get_world_counter()
    @test resolve(CacheCases.independent).code === independent.code

    # Adding an overload need not invalidate the old generic MethodInstance.
    generic = resolve(CacheCases.dispatch)
    @eval CacheCases dispatch(x::Reactant.TracedRArray) = x .+ 5.0f0
    specific = resolve(CacheCases.dispatch)
    @test specific.ci !== generic.ci
    @test specific.code.method !== generic.code.method

    # A caller inferred as part of another root can already have a cached CI.
    # Preparing it must leave its shared inferred source intact.
    interp = SR.Interpreter()
    sig = Tuple{typeof(CacheCases.source_user),A}
    match, _ = CC.findsup(sig, CC.method_table(interp))
    mi = CC.specialize_method(match)
    ci = SR.CompilerCaching.typeinf!(interp, mi)
    src = SR.CompilerCaching.get_source(ci)
    original = string(src)
    prepared = resolve(CacheCases.source_user)
    @test prepared.ci === ci
    @test SR.CompilerCaching.get_source(ci) === src
    @test string(src) == original

    # Synthetic leaf inference must retain dependencies behind the Reactant
    # boundary, including changes to a helper without redefining the leaf.
    leaf = resolve(CacheCases.leaf_parent)
    @eval CacheCases behind_leaf(x) = 1.0
    helper_changed = resolve(CacheCases.leaf_parent)
    @test helper_changed.ci !== leaf.ci
    @test helper_changed.ci.rettype === Float64
    @eval Reactant structured_cache_test_leaf(x::TracedRArray) = 1.0f0
    leaf_changed = resolve(CacheCases.leaf_parent)
    @test leaf_changed.ci !== helper_changed.ci
    @test leaf_changed.ci.rettype === Float32

    # Cleanup runs in the compiler world, but a two-argument call can refer to
    # a global first defined in the user's later world.
    @eval CacheCases late_helper(x, scale) = x .* scale
    @eval CacheCases late_root(x) = late_helper(x, 2.0f0)
    @test resolve(CacheCases.late_root).code isa SR.Code
    @test SR.resolves(
        GlobalRef(CacheCases, :late_helper),
        CacheCases.late_helper,
        Base.get_world_counter(),
    )

    @static if VERSION >= v"1.12-"
        @eval CacheCases const selected = source_user
        previous_world = Base.get_world_counter()
        @eval CacheCases const selected = independent
        @test SR.resolves(
            GlobalRef(CacheCases, :selected), CacheCases.source_user, previous_world
        )
        @test SR.resolves(
            GlobalRef(CacheCases, :selected),
            CacheCases.independent,
            Base.get_world_counter(),
        )
    end
end

@testset "Tracing world" begin
    x = Reactant.to_rarray(Float32[1, 2])
    previous_world = Base.get_world_counter()
    @eval CacheCases @noinline world_child(x) = x .+ 2.0f0
    for f in (CacheCases.world_parent, CacheCases.world_branch)
        before = Base.invoke_in_world(previous_world, Reactant.compile, structured(f), (x,))
        after = Base.invokelatest(Reactant.compile, structured(f), (x,))
        @test Array(before(x)) == Float32[2, 3]
        @test Array(after(x)) == Float32[3, 4]
    end

    # A host callback yields while tracing, allowing a method edit between two
    # calls to the same non-inlined helper. Both calls must use the original world.
    job = @async Base.invokelatest(
        Reactant.compile, structured(CacheCases.world_twice), (x,)
    )
    ready = timedwait(() -> isready(CacheCases.entered) || istaskdone(job), 30.0)
    @test ready === :ok
    if isready(CacheCases.entered)
        take!(CacheCases.entered)
        @eval CacheCases @noinline world_child(x) = x .+ 3.0f0
        put!(CacheCases.resume, nothing)
        first, second = fetch(job)(x)
        @test Array(first) == Float32[3, 4]
        @test Array(second) == Float32[3, 4]
    else
        put!(CacheCases.resume, nothing) # release a late callback if the wait timed out
        fetch(job) # surface a compilation failure instead of waiting for a callback
    end
end
