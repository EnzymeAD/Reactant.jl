# [Partial Evaluation](@id partial-evaluation)

When compiling functions with Reactant, the function arguments (and possible
closure fields) may contain non-Reactant values, so numbers and arrays that
are not of type `Reactant.AbstractConcreteNumber` or
`Reactant.AbstractConcreteArray`.

The Reactant compiler may (but is not guaranteed to) treat these non-Reactant
values as constant and partially evaluate the function to be compiled based
on this.

For example, the function


```@example partial_evaluation_tutorial
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

addxy(x, y)
```

returns a result that depends on both arguments:


```@example partial_evaluation_tutorial
addxy(ConcreteRNumber(7), ConcreteRNumber(8))
```

The StableHLO IR code generated here is:

```@example partial_evaluation_tutorial
@code_hlo add(x, y)
```

and shows two variable inputs.

However, if one of the arguments is a non-Reactant value, then the result

```@example partial_evaluation_tutorial
addx4 = @compile add(x, 4)

addx4(x, 4)
```

will only change based on the first argument, not the second (non-Reactant)
argument:

```@example partial_evaluation_tutorial
addx4(ConcreteRNumber(7), 8)
```

The StableHLO code shows that the second argument has been replaced by a
constant during partial evaluation and is ignored during execution of the
compiled function:

```@example partial_evaluation_tutorial
@code_hlo add(x, 4)
```
