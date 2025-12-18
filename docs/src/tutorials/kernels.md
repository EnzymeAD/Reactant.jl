# Kernels

Suppose your codebase contains custom GPU kernels, typically those defined with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Example

```@example kernels
using KernelAbstractions

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

```jldoctest kernels
x = float.(1:5)
y = square(x)

# output

5-element Vector{Float64}:
  1.0
  4.0
  9.0
 16.0
 25.0
```

## Kernel compilation

To compile such kernels with Reactant, you need to pass the option `raise=true` to the `@compile` or `@jit` macro.
Furthermore, the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package needs to be loaded (even on non-NVIDIA hardware).

```jldoctest kernels
import CUDA
using Reactant

xr = ConcreteRArray(x)
yr = @jit raise=true square(xr)

# output

5-element ConcretePJRTArray{Float64,1}:
  1.0
  4.0
  9.0
 16.0
 25.0
```

## GPU Kernel raising

Kernel raising refer to Reactant's ability to transform a program written in a GPU kernel style. That is, kernel functions which are evaluated in a grid of blocks and threads where operations are done at the scalar level. The transformation raises the program to a tensor style function (in the StableHLO dialect) where operations are broadcasted.

This transformation enables several features:

 - Running the raised compute kernel on hardware where the original kernel was not designed to run on (_i.e._ running a CUDA kernel on a TPU).
 - Enabling further optimizations, since the raised kernel is now indiscernible from the rest of the program, it can be optimized with it. For example, two sequential kernel launches operating on the result of each others can be fused if they are both raised. Resulting in a single kernel launch, in the final optimized StableHLO program.
 - Lastly, automatic-differentiation in Reactant is currently not supported for GPU kernels. Raising kernels enables Enzyme to differentiate the raised kernel. For this to function, one must use the `raise_first` compilation option to make sure the kernel are raised before Enzyme performs automatic-differentiation on the program.

!!! note
    Not all classes of kernels are currently raisable to StableHLO. If your kernel encounters an error while being raised, please open an issue on [the Reactant.jl repository](https://github.com/EnzymeAD/Reactant.jl/issues/new?labels=raising).

## Differentiated kernel

In addition, if you want to compute derivatives of your kernel with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), the option `raise_first=true` also becomes necessary.

```jldoctest kernels
import Enzyme

sumsquare(x) = sum(square(x))
gr = @jit raise=true raise_first=true Enzyme.gradient(Enzyme.Reverse, sumsquare, xr)

# output

(ConcretePJRTArray{Float64, 1, 1}([2.0, 4.0, 6.0, 8.0, 10.0]),)
```
