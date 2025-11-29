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

## Differentiated kernel

In addition, if you want to compute derivatives of your kernel with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), the option `raise_first=true` also becomes necessary.

```jldoctest kernels
import Enzyme

sumsquare(x) = sum(square(x))
gr = @jit raise=true raise_first=true Enzyme.gradient(Enzyme.Reverse, sumsquare, xr)

# output

(ConcretePJRTArray{Float64, 1, 1}([2.0, 4.0, 6.0, 8.0, 10.0]),)
```
