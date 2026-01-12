# [Computational kernels](@id computational-kernels)

```@meta
ShareDefaultModule = true
```

Suppose your code base contains custom computational kernels, such as GPU kernels defined with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) or directly with a backend like [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

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

The Reactant-compiled function `square_compiled` now runs on whatever device Reactant requests, including CPU, GPU, TPU or distributed settings.
It will not run on the device it was written for, nor will it require a CUDA-enabled device.

## Kernel raising

Reactant has the ability to detect and optimize high-level tensor representations of existing kernels, through a process called raising.
For more information, see the [corresponding documentation](@ref "Raising").

## Kernel differentiation

If you want to compute derivatives of your kernel, combining Reactant with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is the best choice.

```@example
import Enzyme
```

Currently, you must use the `raise = true` and `raise_first = true` compilation options to make sure the kernel is raised before Enzyme performs automatic differentiation on the program.
This restriction will be removed in future versions.

```@example
sumsquare(x) = sum(square(x))
gradient_compiled = @compile raise=true raise_first=true Enzyme.gradient(Enzyme.Reverse, sumsquare, xr)
```

Note that the mode and function argument are partially evaluated at compilation time, but we still need to provide them again at execution time:

```@example
gr = gradient_compiled(Enzyme.Reverse, sumsquare, xr)[1]
@assert gr == 2xr  # hide
```
