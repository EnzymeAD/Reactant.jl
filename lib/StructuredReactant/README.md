# StructuredReactant

Ordinary Julia control flow for Reactant, without `@trace`.

```julia
using Reactant, StructuredReactant

function normalize(x)
    n = 0
    while sum(abs.(x)) > 1
        x = x ./ 2
        n += 1
    end
    return n > 3 ? x : x .* 2
end

compiled = @compile structured(normalize)(x)   # one executable, all paths
@jit structured(normalize)(x)
@code_hlo structured(normalize)(x)              # stablehlo.while and stablehlo.if
@jit structured(x -> Enzyme.gradient(Reverse, loss, x))(x)
```

`structured(f)` is a callable that Reactant compiles like any other. Outside
compilation it is just `f`.

The package also has its own `@compile`, `@jit` and `@code_hlo`, which are
Reactant's applied to `structured(f)`, and `compile(f, args)` and
`code_hlo(f, args)` for use without macros. They are not exported; importing
them explicitly makes the frontend the default:

```julia
using Reactant
using StructuredReactant: @compile, @jit, @code_hlo

@jit normalize(x)
```

## How it works

Reactant discovers a program by executing it with traced values. A branch on a
traced value cannot be executed, which is what `@trace` works around. This
package discovers the program from its inferred IR instead:

1. **Infer on traced types.** `f` is type-inferred with a small
   `AbstractInterpreter` that sees Reactant's overlay method table, so the
   program is typed exactly as Reactant would run it. Methods that Reactant
   defines, and methods of Base and the standard library applied to Reactant
   types, are *leaves*: their bodies are neither inferred nor inlined, and
   their return types come from Reactant's own interpreter. Base on host
   values inlines as usual, which is what makes `for` loops recognizable.
2. **Structurize.** [IRStructurizer](https://github.com/maleadt/IRStructurizer.jl)
   turns the SSA IR into nested `if`, `while`, and `for` regions.
3. **Emit.** The structured IR is interpreted. A call to a leaf is handed to
   `Reactant.call_with_reactant`, which is what Reactant's frontend does at
   every call site, so every operation is emitted by Reactant's existing
   methods. A call to any other method is emitted from its own structured IR,
   dispatched on the runtime types of its arguments. Regions are built with
   `Ops.if_condition` and `Ops.while_loop`, the same builders `@trace` uses.

The emitter never executes the user's function. The only semantics it adds to
Julia is that a traced `Bool` in boolean context branches symbolically instead
of throwing. Everything else, including dispatch on traced values, is what
Reactant does today. Code that still uses `@trace` keeps working unchanged.

A branch becomes `stablehlo.if` exactly when its condition is traced at
emission; a host `Bool` selects one side statically. A loop runs at emission
time, as it would in Julia, for as long as its condition and carried values are
host values (its body may still emit operations on traced values); from the
first iteration at which the condition or a carried value is traced, the
remainder becomes a `stablehlo.while`. A counted loop, a `for` over a range
with traced endpoints included, is rolled as `@trace for` emits it: a
zero-based counter compared against the carried iteration count, which is
what Enzyme's reverse pass indexes its caches by, so such loops differentiate
for any trip count known at compile time. A loop that exits through `break`
or `return` carries a `done` flag instead. A loop that stays on the host while its body
emits operations, such as a scalar-indexed kernel, is unrolled into the
program; past `UNROLL_WARNING[]` such iterations (256) a warning says so once
and points at `@trace for`, which rolls a loop regardless.

Because loop carries and induction variables become traced only at emission,
a traced number stands in for its element type: intrinsics on it map to the
`Base` operation Julia's own method computes, and objects whose type inference
specialized on the host type are rebuilt through their constructor.

The structured IR of each method is cached on its `CodeInstance` through
CompilerCaching.jl, so it is invalidated exactly when the inferred code is.
Method lookup and global reads use the world of the program being traced; the
compiler itself runs in the world the package was loaded in, which keeps its
precompiled code valid (and means edits to this package need a restart).

## Scope

Supported: `if`/`elseif`/`else`, `&&` and `||`, `return` from inside a branch,
`while`, counted `for` over integer ranges including `eachindex(x)`,
`axes(x, d)`, and ranges with traced endpoints, `break`, `continue`, and
`return` inside loops, nested and combined control flow, tuple and struct results,
helpers at any depth including `@noinline` ones and dynamic dispatch, keyword
and variadic arguments, closures, mutation inside branches, `@trace` inside
structured code, elementwise control flow through `structured(f).(x)`, user
callbacks handed to Reactant (`map`, `sum(f, x)`, the body given to
`Enzyme.autodiff`), Enzyme reverse mode, and `try` blocks such as
`@allowscalar x[i]`: exception handlers are dropped, since nothing throws in
the compiled program and an exception while tracing fails the compile, while
a `finally` body still runs on the normal path.

Rejected with a `FrontendError` naming the method: reading a value after a
`while` loop that may not have assigned it, recursion on traced values, a branch
on a traced condition that always throws, `===` on traced floating-point values,
and anything IRStructurizer cannot structure. Calls whose arguments carry no
traced values run as ordinary Julia, so host-side work is unaffected. Base
functions applied to traced values behave exactly as under Reactant today,
including where that fails. Reverse-mode differentiation through a
data-dependent `while` still needs the checkpointing hints that only `@trace`
can express.

A variable that a closure reassigns but that also lives in an enclosing scope is
boxed by Julia, and its "possibly undefined" paths surface as throwing branches
once a loop over it is rolled; declare such a counter `local`.

## Running the tests

```sh
julia --project=lib/StructuredReactant/test lib/StructuredReactant/test/runtests.jl
```

`test/migration.jl` is a separate experiment: it makes `@trace` an identity
macro, routes every compilation through `structured`, and runs one of
Reactant's own test files unchanged, `test/core/control_flow.jl` by default or
`autodiff` when named as an argument, reporting what passes and why the rest
does not.

Requires Julia 1.11, 1.12 or 1.13. The compiler interfaces that changed in
1.12 (`Future`s, `MethodCallResult`, the inlining hook) are gated in
`interpreter.jl`; CompilerCaching covers the differences in the inferred source
cache. Julia 1.11 also lowers `finally` to one shared body that dispatches on a
state value; once the handler is dropped that dispatch compares two literals and
is decided before structurization, so its dead `rethrow` never becomes a loop
exit.
