# [Control Flow](@id control-flow)

Reactant currently uses a tracing system to capture array operations into a new
program. As such, the provided function is executed with [`TracedRArray`](@ref)
as inputs instead of [`ConcreteRArray`](@ref). This means that during tracing
only operations affecting such arrays are captured by Reactant into the new
program.

In practice, this means that Julia native control flow constructs are not
captured by Reactant.

Consider the following function which has a conditional control flow depending
on one of its argument which is a boolean:

```julia
using Reactant

function maybe_square(cond, x)
    if cond
        x = x .^ 2
    else
        x = x
    end
    return x
end
```

We can confirm by compiling our function and noticing that the result does not
depend on the argument provided to the compiled function.

```julia
x = Reactant.ConcreteRArray(randn(Float32, 100))

maybe_square_compiled = @compile maybe_square(true, x)
maybe_square_compiled(false, x) == maybe_square_compiled(true, x)
```

But instead, it depends on the value that was provided during tracing to the
initial `@compile` invocation.

The same behaviour can be observed when using loops.

```julia
function add_n(x, n)
    for _ in 1:n
        x .+= 1
    end
    return x
end
```

## Data-dependent Control Flow using [`@trace`](@ref)

During tracing the arrays contain no data and only information about their shape
and data type. As such, it is not possible to execute conditions that would
depend on the value of an array. For this cases, ReactantCore provides the
[`@trace`](@ref) macro to allow capturing control flow expressions in the
compiled program.

Taking our same function from before and adding the [`@trace`](@ref) macro
before the if expression will allow our compiled function to contain the
condition.

```@example control_flow_tutorial
using Reactant

function maybe_square(cond, x)
    @trace if cond
        x = x ^ 2
    else
        x = x
    end
    return x
end
```

First, let's note that [`@trace`](@ref) has no impact when the program is not
run in a Reactant trace. As such, the function can still be used with plain
Julia values. That makes it possible to include `@trace` in library code.

```@example control_flow_tutorial
x = 2.
maybe_square(true, x) == x ^ 2
```

Then in our compiled version, we can pass a Reactant concrete boolean to
conditionally control the output of the function.

```@example control_flow_tutorial
cond = Reactant.ConcreteRNumber(true)
x = Reactant.ConcreteRNumber(2.)

@jit(maybe_square(cond, x))[1] == Reactant.ConcreteRNumber(4.)
```
