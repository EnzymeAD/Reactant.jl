# [Automatic Differentiation](@id automatic-differentiation)

Reactant integrates seamlessly with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) to provide high-performance automatic differentiation (AD) capabilities. This tutorial will guide you through using Enzyme.jl with Reactant to compute gradients using forward and reverse mode automatic differentiation.

```@example autodiff_tutorial
using Reactant, Enzyme, Random
using Test # hide
```

## Forward Mode Automatic Differentiation

### Basic Forward Mode

```@example autodiff_tutorial
# Define a simple function
square(x) = x .^ 2

# Create input data
x = Reactant.to_rarray(Float32[3.0, 4.0, 5.0])

function sq_fwd(x)
    return Enzyme.autodiff(Forward, square, Duplicated(x, fill!(similar(x), true)))[1]
end

# Compute forward-mode autodiff
# Forward mode with Duplicated activity
result = @jit sq_fwd(x)

println("Result: ", result)
@test result ≈ 2 .* Array(x) # hide
nothing # hide
```

The `Duplicated` activity type means both the primal value and its derivative are computed.

### Forward Mode with Primal

You can also get both the function value and its derivative:

```@example autodiff_tutorial
# Forward mode with primal value
function sq_fwd_primal(x)
    return Enzyme.autodiff(
        ForwardWithPrimal, square, Duplicated(x, fill!(similar(x), true))
    )
end

tangent, primal = @jit sq_fwd_primal(x)

println("Primal: ", primal)
println("Tangent: ", tangent)
@test primal ≈ Array(x) .^ 2 # hide
@test tangent ≈ 2 .* Array(x) # hide
nothing # hide
```

### Computing Gradients

For computing gradients of scalar-valued functions:

```@example autodiff_tutorial
sum_squares(x) = sum(abs2, x)

x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0])

# Compute gradient using forward mode
grad_result = @jit Enzyme.gradient(Forward, sum_squares, x)

println("Gradient: ", grad_result[1])
x_arr = Array(x) # hide
@test grad_result[1][1] ≈ 2 * x_arr[1] # hide
@test grad_result[1][2] ≈ 2 * x_arr[2] # hide
@test grad_result[1][3] ≈ 2 * x_arr[3] # hide
nothing # hide
```

## Reverse Mode Automatic Differentiation

### Basic Reverse Mode

```@example autodiff_tutorial
loss_function(x) = sum(x .^ 3)

x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0])

# Compute gradient using reverse mode
grad = @jit Enzyme.gradient(Reverse, loss_function, x)

println("Gradient: ", grad[1])
@test grad[1] ≈ 3 .* Array(x) .^ 2 # hide
nothing # hide
```

### Reverse Mode with Primal

Get both the function value and gradient:

```@example autodiff_tutorial
# Reverse mode with primal
result = @jit Enzyme.gradient(ReverseWithPrimal, loss_function, x)

println("Value: ", result.val)
println("Gradient: ", result.derivs[1])
@test result.val ≈ loss_function(Array(x)) # hide
@test result.derivs[1] ≈ 3 .* Array(x) .^ 2 # hide
nothing # hide
```

## More Examples

### Multi-argument Functions

```@example autodiff_tutorial
function multi_arg_func(x, y)
    return sum(x .* y .^ 2)
end

x = Reactant.to_rarray(Float32[1.0, 2.0])
y = Reactant.to_rarray(Float32[3.0, 4.0])

# Gradient w.r.t. both arguments
grad = @jit Enzyme.gradient(Reverse, multi_arg_func, x, y)

println("∂f/∂x: ", grad[1])
println("∂f/∂y: ", grad[2])
@test grad[1] ≈ Array(y) .^ 2 # hide
@test grad[2] ≈ Array(x) .* Array(y) .* 2 # hide
nothing # hide
```

### Vector Mode AD

Vector mode computes multiple derivatives simultaneously:

```@example autodiff_tutorial
vector_func(x) = sum(abs2, x)

x = Reactant.to_rarray(collect(Float32, 1:4))

# Create onehot vectors for vector mode
onehot_vectors = @jit Enzyme.onehot(x)

# Vector forward mode
result = @jit Enzyme.autodiff(Forward, vector_func, BatchDuplicated(x, onehot_vectors))

println("Vector gradients: ", result[1])
nothing # hide
```

### Nested Automatic Differentiation

Compute higher-order derivatives:

```@example autodiff_tutorial
power4(x) = x^4

x = Reactant.ConcreteRNumber(3.1)

# First derivative
first_deriv(x) = Enzyme.gradient(Reverse, power4, x)[1]

# Second derivative
second_deriv(x) = Enzyme.gradient(Reverse, first_deriv, x)[1]

# Compute second derivative
result = @jit second_deriv(x)
result_enz = second_deriv(Float32(x))
println("Second derivative: ", result)
@test result ≈ result_enz # hide
nothing # hide
```

### Division by Zero with Strong Zero

```@example autodiff_tutorial
div_by_zero(x) = min(1.0, 1 / x)

x = Reactant.ConcreteRNumber(0.0)

# Regular gradient (may be NaN)
regular_grad = @jit Enzyme.gradient(Reverse, div_by_zero, x)

# Strong zero gradient (handles singularities better)
strong_zero_grad = @jit Enzyme.gradient(Enzyme.set_strong_zero(Reverse), div_by_zero, x)

println("Regular gradient: ", Float32(regular_grad[1]))
println("Strong zero gradient: ", Float32(strong_zero_grad[1]))
nothing # hide
```

### Ignoring Derivatives

Use [`EnzymeCore.ignore_derivatives`](@ref) to exclude parts of computation from gradient:

```@example autodiff_tutorial
function func_with_ignore(x)
    # This part won't contribute to gradient
    ignored_sum = Enzyme.ignore_derivatives(sum(x))
    # This part will contribute
    return sum(x .^ 2) + ignored_sum
end

x = Reactant.to_rarray([1.0, 2.0, 3.0])

grad = @jit Enzyme.gradient(Reverse, func_with_ignore, x)
println("Gradient: ", grad[1])
@test grad[1] ≈ Array(x) .* 2 # hide
nothing # hide
```

### Complex Numbers and Special Arrays

Reactant supports complex numbers and various array types:

```@example autodiff_tutorial
# Complex arrays
x_complex = Reactant.to_rarray([1.0 + 2.0im, 3.0 + 4.0im])

function complex_func(z)
    return sum(abs2, z)
end

grad_complex = @jit Enzyme.gradient(Reverse, complex_func, x_complex)
println("Complex gradient: ", grad_complex[1])
@test grad_complex[1] ≈ Array(x_complex) .* 2 # hide
nothing # hide
```

### Loops

When performing computations in a loop using the [`@trace`](@ref) (see the [tutorial about Control Flow](@ref control-flow)), Enzyme has to save intermediary results during the primal computation to be used in the reverse pass.

The [`@trace`](@ref) macro offers two parameters to limit the amount of memory that has to be saved. The first one in the `mincut` option which strictly reduces the amount of saved memory by saving only the minimal amount of memory needed for each iteration and recomputing variables if needed.

Secondly, it is possible to enable the `checkpointing` option which will save intermediary loop carried values every $\sqrt{N}$ iterations and perform complete recomputation during the reverse pass. This is a way to trade compute time against memory.

```@example autodiff_tutorial
function f_checkpointing(x, enable_checkpointing)
    y = copy(x)

    # The intermediary values of y will need to be cached
    # to be reused in the reverse pass. With checkpointing enabled,
    # the cache will be of size `Int(sqrt(9)) * length(y)` instead of
    # `9 * length(y)`.
    @trace checkpointing=enable_checkpointing for i in 1:9
        y .*= x
    end

    return y
end

f_checkpointing_diff(x, enable_checkpointing) =
    Enzyme.gradient(Reverse, f_checkpointing, x, Const(enable_checkpointing))
```

!!! note
    The currently implemented checkpointing scheme only supports a constant number
    of iterations which has an integer square root. If $N$ is the number of iterations,
    the values will be cached $\sqrt N$ times against $N$ times if checkpointing
    is disabled.

### Complete Example: Neural Network Training

!!! tip "Training Lux Neural Networks"
    If you are using [Lux.jl](https://lux.csail.mit.edu/) for neural networks, prefer using the
    [TrainState API](https://lux.csail.mit.edu/stable/manual/compiling_lux_models#compile_lux_model_trainstate)
    that abstracts away a lot of these details.

Here's a complete example of training a simple neural network:

```@example autodiff_tutorial
# Define network
function neural_net(x, w1, w2, b1, b2)
    h = tanh.(w1 * x .+ b1)
    return w2 * h .+ b2
end

# Loss function
function loss(x, y, w1, w2, b1, b2)
    pred = neural_net(x, w1, w2, b1, b2)
    return sum(abs2, pred .- y)
end

# Generate data
x = Reactant.to_rarray(rand(Float32, 10, 32))
y = Reactant.to_rarray(2 .* sum(abs2, Array(x); dims=1) .+ rand(Float32, 1, 32) .* 0.001f0)

# Initialize parameters
w1 = Reactant.to_rarray(rand(Float32, 20, 10))
w2 = Reactant.to_rarray(rand(Float32, 1, 20))
b1 = Reactant.to_rarray(rand(Float32, 20))
b2 = Reactant.to_rarray(rand(Float32, 1))

# Training step
function train_step(x, y, w1, w2, b1, b2, lr)
    # Compute gradients
    (; val, derivs) = Enzyme.gradient(
        ReverseWithPrimal, loss, Const(x), Const(y), w1, w2, b1, b2
    )

    # Update parameters (simple gradient descent)
    w1 .-= lr .* derivs[3]
    w2 .-= lr .* derivs[4]
    b1 .-= lr .* derivs[5]
    b2 .-= lr .* derivs[6]

    return val, w1, w2, b1, b2
end

# Training loop
compiled_train_step = @compile train_step(x, y, w1, w2, b1, b2, 0.001f0)

for epoch in 1:100
    global w1, w2, b1, b2
    loss_val, w1, w2, b1, b2 = compiled_train_step(x, y, w1, w2, b1, b2, 0.001f0)
    if epoch % 10 == 0
        @info "Epoch: $epoch, Loss: $loss_val"
    end
end

println("Training completed!")
nothing # hide
```
