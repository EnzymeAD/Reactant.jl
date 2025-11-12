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

```@example control_flow_tutorial
using Reactant

function maybe_square(cond, x)
    if cond
        x = x .^ 2
    else
        x = x .^ 3
    end
    return x
end
```

We can confirm by compiling our function and noticing that the result does not
depend on the argument provided to the compiled function.

```@example control_flow_tutorial
x = Reactant.ConcreteRArray(randn(Float32, 100))

maybe_square_compiled = @compile maybe_square(true, x)
maybe_square_compiled(false, x) == maybe_square_compiled(true, x)
```

But instead, it depends on the value that was provided during tracing to the
initial `@compile` invocation. This is also confirmed when looking at the
code generated during tracing which does not contain any conditional.

```@example control_flow_tutorial
@code_hlo maybe_square(false, x)
```

The same behaviour can be observed when using loops. In the following example,
the loop is "unrolled" because it is not captured in the program. The optimizer
then fuses all additions to add `n = 10` directly to the argument.

```@example control_flow_tutorial
function add_n(x, n)
    for _ in 1:n
        x .+= 1
    end
    return x
end

x = Reactant.to_rarray(zeros(Int, 10))
n = 10
@code_hlo add_n(x, n)
```

In the next section, we will see what mechanism Reactant offers to integrate
data-dependent control flow in the captured programs.

## Data-dependent Control Flow using [`@trace`](@ref)

During tracing the arrays contain no data and only information about their shape
and data type. As such, it is not possible to execute conditions that would
depend on the value of an array. For these cases, ReactantCore provides the
[`@trace`](@ref) macro to allow capturing control flow expressions in the
compiled program.

### Conditional Control Flow

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

This can also be confirmed by looking at the generated MLIR code which
will contain a `stablehlo.if` operation:

```@example control_flow_tutorial
@code_hlo maybe_square(cond, x)
```

In our simple example, the condition is passed directly as an argument but
the same mechanism is applied to conditions which are computed from within
a function from traced arguments, leading to a traced condition.

### Loops

In addition to conditional evaluations, [`@trace`](@ref) also supports capturing
loops. This is possible in the form of both for and while loops.
This enables one to write algorithms that would not be possible otherwise such as
performing computations until convergence or running a computation for an certain
number of iterations which is only known during runtime.

Here is an example of a function which computes the cumsum in non-optimized manner
using a for loop:

```@example control_flow_tutorial
function cumsum!(x)
    v = zero(eltype(x))
    @trace for i in eachindex(x)
        v += @allowscalar x[i]
        @allowscalar x[i] = v
    end
    return x
end

x = Reactant.to_rarray([1., 2., 3.])
@jit(cumsum!(x)) 
```

Similarly, one can trace while loops. The following is a minimal implementation of the
[Sinkhorn-Knopp algorithm](https://en.wikipedia.org/wiki/Sinkhorn's_theorem) which aims
to solve the entropic optimal transport problem:

```@example control_flow_tutorial
using LinearAlgebra: Diagonal

function sinkhorn(μ, ν, C)
    λ = eltype(C)(0.8)
    K = @. exp(-C / λ)

    u = fill!(similar(μ), one(eltype(μ)))
    v = similar(ν)

    π = Diagonal(u) * K * Diagonal(v)
    err = typemax(eltype(π))

    @trace while err >= 0.001
        v = ν ./ (K' * u)
        u = μ ./ (K * v)

        new_π = Diagonal(u) * K * Diagonal(v)
        err = sum(abs2, new_π .- π)
        π = new_π
    end

    return π
end

a = Reactant.to_rarray(ones(Float32, 10) ./ 10)
b = Reactant.to_rarray(ones(Float32, 12) ./ 12)
C = Reactant.to_rarray(randn(Float32, 10, 12))

π = @jit sinkhorn(a, b, C)

# The sum of the transport plan is 1.
sum(π)
```

This implementation runs the algorithm until convergence (the transport plan has seen little change in the last iteration). Without [`@trace`](@ref) this would not be possible to implement since the termination condition is depending on traced values (in this case, the value of the transport plan).

!!! warning "Current limitations"

    It is currently not allowed to include mutations as part of the while loop condition.

    The for loop tracing does not support any arbitrary iterable. It supports integer ranges.
