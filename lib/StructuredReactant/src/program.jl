"""
    structured(f)

Wrap `f` so that Reactant captures its control flow from inferred structured IR
instead of executing it with traced values. `if`, `while`, and `for` over traced
values become `stablehlo.if` and `stablehlo.while` without `@trace`, in `f` and
in every function it calls. Outside compilation the wrapper simply calls `f`.

Use it wherever Reactant accepts a callable:

```julia
step(x) = sum(x) > 0 ? x .* 2 : x ./ 2
compiled = @compile structured(step)(x)
@jit structured(step)(x)
@code_hlo structured(step)(x)
Enzyme.gradient(Reverse, structured(loss), x)   # inside a compiled function
```

Methods that Reactant defines, and Base methods applied to traced values, are
emission leaves and behave exactly as they do today. User functions passed to
such leaves (the function given to `Enzyme.gradient` or mapped over an array)
are wrapped automatically.
"""
structured(f) = Program(f)

struct Program{F} <: Function
    f::F
end

structured(p::Program) = p

# The plain call runs `f` outside compilation. Reactant's interpreter types it
# for callbacks, under which `f` may "always throw" (a branch on a traced Bool),
# so the body is hidden behind a barrier and asserted to this frontend's type.
function program_call_generator(
    world::UInt, source, @nospecialize(P::Type), @nospecialize(argtypes::Tuple)
)
    @nospecialize
    rt = try
        Base.invoke_in_world(
            COMPILER_WORLD[],
            CC._return_type,
            Interpreter(world),
            Tuple{P.parameters[1],argtypes...},
        )
    catch
        Any
    end
    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(Symbol("#self#"), :args), Core.svec()
    )
    body = :(Base.inferencebarrier(getfield(var"#self#", :f))(args...)::$(rt))
    return stub(world, source, body)
end

@eval function (p::Program)(args...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, program_call_generator))
end

function Core.kwcall(kwargs::NamedTuple, p::Program, args...)
    return Base.inferencebarrier(p.f)(args...; kwargs...)
end

Base.show(io::IO, p::Program) = print(io, "structured(", p.f, ")")

# Reactant reaches every call through `call_with_reactant`: a `Program` is
# emitted from IR instead of executed. The barrier keeps native inference of
# these methods out of the emitter, where it would type `f` without Reactant's
# overlays (a nested `Enzyme.gradient` then aborts under native Enzyme).
function Reactant.call_with_reactant(p::Program, args...)
    return scalar_indexing_preserved() do
        return Base.inferencebarrier(emit_method)(p.f, args, nothing)
    end
end

function Reactant.call_with_reactant(
    ::typeof(Core.kwcall), kwargs::NamedTuple, p::Program, args...
)
    return scalar_indexing_preserved() do
        return Base.inferencebarrier(emit_method)(
            Core.kwcall, (kwargs, p.f, args...), nothing
        )
    end
end

# `@allowscalar` restores the scalar-indexing state in a `finally` body, of which
# only the normal path is emitted; a failed emission would leave it set.
function scalar_indexing_preserved(f)
    tls = task_local_storage()
    saved = get(tls, :ScalarIndexing, nothing)
    try
        return f()
    finally
        saved === nothing ? delete!(tls, :ScalarIndexing) : (tls[:ScalarIndexing] = saved)
    end
end

function Reactant.call_with_reactant(::Reactant.EnsureReturnType, p::Program, args...)
    return Reactant.call_with_reactant(p, args...)
end

function Reactant.call_with_reactant(
    ::Reactant.EnsureReturnType,
    kwcall::typeof(Core.kwcall),
    kwargs::NamedTuple,
    p::Program,
    args...,
)
    return Reactant.call_with_reactant(kwcall, kwargs, p, args...)
end
