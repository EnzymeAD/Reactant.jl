# [Partial Evaluation](@id partial-evaluation)

When compiling functions with Reactant, the function arguments (and possible
closure fields) may contain non-Reactant values, i.e. numbers and arrays that
are not of type `Reactant.AbstractConcreteNumber` or
`Reactant.AbstractConcreteArray`.

The Reactant compiler may (but is not guaranteed to) treat these non-Reactant
values as constant and partially evaluate the function to be compiled based
on this.

For example, the function

```@example partial_evaluation_tutorial
using Reactant

function add(a, b)
   a + b
end
```

when compiled with two `ConcreteRNumber` arguments

```@example partial_evaluation_tutorial
using Reactant

x = ConcreteRNumber(3)
y = ConcreteRNumber(4)

addxy = @compile add(x, y)

res = addxy(x, y)
@assert Int(res) == 7 #hide
res #hide
```

returns a result that depends on both arguments `x` and `y`:

```@example partial_evaluation_tutorial
res = addxy(ConcreteRNumber(7), ConcreteRNumber(8))
@assert Int(res) == 15 #hide
res #hide
```

The StableHLO IR code generated here is:

```@example partial_evaluation_tutorial
@code_hlo add(x, y)
```

So at HLO-level, there a are two variable inputs `%arg0` and `%arg1`.

However, if argument `y` has a non-Reactant value during compilation, (`4` in
this example) then the result when executing the compiled function

```@example partial_evaluation_tutorial
addx4 = @compile add(x, 4)

res = addx4(x, 4)
@assert Int(res) == 7 #hide
res #hide
```

will only change based on `x`, not on the non-Reactant argument `y`, we get
`7 + 4 == 11`, not `7 + 8 == 15`:

```@example partial_evaluation_tutorial
res = addx4(ConcreteRNumber(7), 8)
@assert Int(res) == 11 #hide
res #hide
```

The StableHLO code shows that the second argument has been replaced by a
constant `%c` during partial evaluation. When the compiled function is
executed, the value of `y` is ignored - at HLO-level, there is only one
variable input `%arg0`:

```@example partial_evaluation_tutorial
@code_hlo add(x, 4)
```

## Tracking Scalar Numbers with `track_numbers`

By default, [`Reactant.to_rarray`](@ref) converts arrays into `ConcreteRArray`s but
leaves plain Julia numbers (scalars) as-is. This means scalar arguments are treated
as compile-time constants and will be frozen at their tracing-time values.

To make scalar numbers participate in tracing (so they become `ConcreteRNumber`s that
can vary at runtime), use the `track_numbers` keyword argument:

```julia
# Default: scalar `t` is NOT tracked (frozen at trace time)
t = Reactant.to_rarray(0.5)          # returns plain Float64, not a ConcreteRNumber

# With track_numbers: scalar `t` IS tracked (varies at runtime)
t = Reactant.to_rarray(0.5; track_numbers=true)  # returns ConcreteRNumber{Float64}
```

The `track_numbers` keyword accepts:
- `false` (default): scalars are left as plain Julia numbers (constant at compile time)
- `true`: all scalar numbers are converted to `ConcreteRNumber`
- A type, e.g. `Number`, `Int`, `Float64`: only scalars matching that type are tracked

This also works when converting nested structures:

```julia
struct MyParams
    values::Vector{Float64}
    scale::Float64
end

params = MyParams([1.0, 2.0], 3.0)

# Without track_numbers: params.scale stays as Float64 (frozen)
rparams = Reactant.to_rarray(params)

# With track_numbers: params.scale becomes ConcreteRNumber{Float64}
rparams = Reactant.to_rarray(params; track_numbers=true)
```

!!! warning "Common pitfall: frozen scalar arguments"
    If a compiled function produces correct results at the tracing value but wrong
    results for other inputs, check whether any scalar arguments were not tracked.
    This is especially common when passing scalar parameters (like a time value `t`)
    to functions inside `Enzyme.autodiff` with `Enzyme.Const` — the `Const` wrapper
    does not cause the issue, but the underlying scalar being a plain Julia number
    (rather than a `ConcreteRNumber`) means it was baked in as a constant during
    compilation. Use `track_numbers=true` in `to_rarray` to fix this.
