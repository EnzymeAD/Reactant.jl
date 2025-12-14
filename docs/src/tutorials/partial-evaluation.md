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
