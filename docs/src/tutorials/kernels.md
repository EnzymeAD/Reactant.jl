# [GPU Kernels](@id gpu-kernels)

```@meta
ShareDefaultModule = true
```

Suppose your code base contains custom GPU kernels, such as those defined with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) or directly with a backend like [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

## Example

```@example
using KernelAbstractions
```

Here we define a very simple squaring kernel:

```@example
@kernel function square_kernel!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i] * x[i]
end

function square(x)
    y = similar(x)
    backend = KernelAbstractions.get_backend(x)
    kernel! = square_kernel!(backend)
    kernel!(y, x; ndrange=length(x))
    return y
end
```

Let's test it to make sure it works:

```@example
x = float.(1:5)
y = square(x)
@assert y == x .^ 2  # hide
```

## Kernel compilation

To compile this kernel with Reactant, the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package needs to be loaded (even on non-NVIDIA hardware).

```@example
import CUDA
using Reactant
```

The rest of the compilation works as usual:

```@example
xr = ConcreteRArray(x)
square_compiled = @compile square(xr)
```

```@example
yr = square_compiled(xr)
@assert yr == xr .^ 2  # hide
```

## Kernel raising

Kernel raising refer to Reactant's ability to transform a program written in a GPU kernel style (that is, kernel functions which are evaluated in a grid of blocks and threads, where operations are done at the scalar level).
The transformation raises the program to a tensor-style function (in the StableHLO dialect) where operations are broadcasted.

Raising is achieved by passing the keyword `raise = true` during compilation:

```@example
square_compiled_raised = @compile raise=true square(xr)
```

```@example
yr2 = square_compiled_raised(xr)
@assert yr2 == xr .^ 2  # hide
```

This transformation unlocks several features:

- Running the raised compute kernel on hardware where the original kernel was not designed to run on (_i.e._ running a CUDA kernel on a TPU).
- Enabling further optimizations: since the raised kernel is now indiscernible from the rest of the program, it can be optimized with it. For example, two sequential kernel launches operating on the result of each other can be fused if they are both raised. This results in a single kernel launch for the final optimized StableHLO program.
- Supporting automatic differentiation, which Reactant currently cannot handle for GPU kernels. Raising kernels enables Enzyme to differentiate the raised kernel (more on this below).

!!! note
    Not all classes of kernels are currently raisable to StableHLO. If your kernel encounters an error while being raised, please open an issue on [the Reactant repository](https://github.com/EnzymeAD/Reactant.jl/issues/new?labels=raising).

## Kernel differentiation

If you want to compute derivatives of your kernel, combining Reactant with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is the best choice.

```@example
import Enzyme
```

You must use the `raise_first = true` compilation option to make sure the kernel is raised before Enzyme performs automatic differentiation on the program.

```@example
sumsquare(x) = sum(square(x))
gradient_compiled = @compile raise=true raise_first=true Enzyme.gradient(Enzyme.Reverse, sumsquare, xr)
```

Note that the mode and function argument are partially evaluated at compilation time, but we still need to provide them again at execution time:

```@example
gr = gradient_compiled(Enzyme.Reverse, sumsquare, xr)[1]
@assert gr == 2xr  # hide
```
