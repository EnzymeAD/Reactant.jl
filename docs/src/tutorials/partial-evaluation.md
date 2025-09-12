# [Partial Evaluation](@id partial-evaluation)

When compiling functions with Reactant, the function arguments (and possible
closure fields) may contain non-Reactant values, i.e. numbers and arrays that
are not of type `Reactant.AbstractConcreteNumber` or
`Reactant.AbstractConcreteArray`.

The Reactant compiler may (but is not guaranteed to) treat these non-Reactant
values as constant and partially evaluate the function to be compiled based
on this.

For example, the function

```jldoctest partial_evaluation_tutorial
using Reactant
function add(a, b)
   a + b
end;

# output

add (generic function with 1 method)
```

when compiled with two `ConcreteRNumber` arguments

```jldoctest partial_evaluation_tutorial
using Reactant

x = ConcreteRNumber(3)
y = ConcreteRNumber(4)

addxy = @compile add(x, y)

addxy(x, y)

# output

ConcretePJRTNumber{Int64, 1, Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding, Nothing}}(7)
```

returns a result that depends on both arguments `x` and `y`:

```jldoctest partial_evaluation_tutorial
addxy(ConcreteRNumber(7), ConcreteRNumber(8))

# output

ConcretePJRTNumber{Int64, 1, Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding, Nothing}}(15)
```

The StableHLO IR code generated here is:

```jldoctest partial_evaluation_tutorial
@code_hlo add(x, y)

# output

module @reactant_add attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
```

So at HLO-level, there a are two variable inputs `%arg0` and `%arg1`.

However, if argument `y` has a non-Reactant value during compilation, (`4` in
this example) then the result when executing the compiled function

```jldoctest partial_evaluation_tutorial
addx4 = @compile add(x, 4)

addx4(x, 4)

# output

ConcretePJRTNumber{Int64, 1, Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding, Nothing}}(7)
```

will only change based on `x`, not on the non-Reactant argument `y`, we get
`7 + 4 == 11`, not `7 + 8 == 15`:

```jldoctest partial_evaluation_tutorial
addx4(ConcreteRNumber(7), 8)

# output

ConcretePJRTNumber{Int64, 1, Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding, Nothing}}(11)
```

The StableHLO code shows that the second argument has been replaced by a
constant `%c` during partial evaluation. When the compiled function is
executed, the value of `y` is ignored - at HLO-level, there is only one
variable input `%arg0`:

```jldoctest partial_evaluation_tutorial
@code_hlo add(x, 4)

# output

module @reactant_add attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<4> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    return %0 : tensor<i64>
  }
}
```
