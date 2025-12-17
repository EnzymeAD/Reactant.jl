# Raising

## Raising GPU Kernels

Kernel raising refer to Reactant's ability to transform a program written in a GPU kernel
style. That is, kernel functions which are evaluated in a grid of blocks and threads where
operations are done at the scalar level. The transformation raises the program to a tensor
style function (in the StableHLO dialect) where operations are broadcasted.

This transformation enables several features:

- Running the raised compute kernel on hardware where the original kernel was not designed
  to run on (_i.e._ running a CUDA kernel on a TPU).
- Enabling further optimizations, since the raised kernel is now indiscernible from the rest
  of the program, it can be optimized with it. For example, two sequential kernel launches
  operating on the result of each others can be fused if they are both raised. Resulting in
  a single kernel launch, in the final optimized StableHLO program.
- Lastly, automatic-differentiation in Reactant is currently not supported for GPU kernels.
  Raising kernels enables Enzyme to differentiate the raised kernel. For this to function,
  one must use the `raise_first` compilation option to make sure the kernel are raised
  before Enzyme performs automatic-differentiation on the program.

!!! note

    Not all classes of kernels are currently raisable to StableHLO. If your kernel
    encounters an error while being raised, please open an issue on
    [the Reactant.jl repository](https://github.com/EnzymeAD/Reactant.jl/issues/new?labels=raising).

### Example

```@example raising_kernelabstractions_to_stablehlo
using Reactant
using KernelAbstractions
using CUDA # needs to be loaded for raising even if CUDA is not functional on your system
```

!!! tip

    We could have also directly implemented the kernel using CUDA.jl instead of KernelAbstractions.jl.

We will implement a simple kernel to compute the square of a vector.

```@example raising_kernelabstractions_to_stablehlo
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

```@example raising_kernelabstractions_to_stablehlo
x = Reactant.to_rarray(collect(1:1:64) ./ 64)
nothing # hide
```

Let's see what the HLO IR looks like for this function. Note that raising is automatically
enabled for backends like TPU, where the original kernel was not designed to run on. To
enable raising on other backends, pass the `raise=true` option.

```@example raising_kernelabstractions_to_stablehlo
@code_hlo raise=true square(x)
```

## Raising Scalar Loops to Tensor IR

We will implement a simple N body simulation code in Reactant. Instead of using
broadcasting or high-level abstractions, we will use loops and scalar operations
to implement this.

```@example raising_stablehlo
using Reactant, PrettyChairmarks

Reactant.allowscalar(true) # generally not recommended to turn on globally
```

We will implement a naive function to compute the attractive force between each
pair of particles in a system.

```@example raising_stablehlo
function compute_attractive_force(
    positions::AbstractMatrix, masses::AbstractVector, G::Number
)
    N = size(positions, 2)
    F = similar(positions, N, N)

    @trace for i in 1:N
        @trace for j in 1:N
            dx = positions[1, i] - positions[1, j]
            dy = positions[2, i] - positions[2, j]
            dz = positions[3, i] - positions[3, j]

            invr² = ifelse(i == j, dx, inv(dx^2 + dy^2 + dz^2))

            Fx = G * masses[i] * masses[j] * invr² * dx
            Fy = G * masses[i] * masses[j] * invr² * dy
            Fz = G * masses[i] * masses[j] * invr² * dz
            F[i, j] = Fx + Fy + Fz
        end
    end

    return F
end
```

```@example raising_stablehlo
positions = randn(Float32, 3, 1024)
masses = rand(Float32, 1024) .* 10

positions_ra = Reactant.to_rarray(positions)
masses_ra = Reactant.to_rarray(masses)
nothing # hide
```

Let's see what the HLO IR looks like for this function (without enabling the loop
raising).

```@example raising_stablehlo
@code_hlo compile_options = CompileOptions(;
    disable_loop_raising_passes=true
) compute_attractive_force(positions_ra, masses_ra, 2.0f0)
```

This IR has a nested loop, but that won't work nicely for GPUs/TPUs. Even for CPUs, XLA
often doens't do a great job with loops. We will attempt to raise loops to a
tensor IR.

```@example raising_stablehlo
hlo = @code_hlo compile_options=CompileOptions(;
    disable_loop_raising_passes=false
) compute_attractive_force(positions_ra, masses_ra, 2.0f0)
@assert !contains(repr(hlo), "stablehlo.while") #hide
hlo
```

This IR won't have any loops, instead it will be written in a tensor IR! Let ensure that
the values are identical.

```@example raising_stablehlo
y_jl = compute_attractive_force(positions, masses, 2.0f0)
y_ra = @jit compile_options=CompileOptions(;
    disable_loop_raising_passes=false
) compute_attractive_force(positions_ra, masses_ra, 2.0f0)
maximum(abs, Array(y_ra) .- y_jl)
```

Let's time the execution of the two versions.

```@example raising_stablehlo
fn1 = @compile sync=true compile_options=CompileOptions(;
    disable_loop_raising_passes=true
) compute_attractive_force(positions_ra, masses_ra, 2.0f0)
fn2 = @compile sync=true compute_attractive_force(positions_ra, masses_ra, 2.0f0)
```

Runtime for non-raised function:

```@example raising_stablehlo
fn1(positions_ra, masses_ra, 2.0f0) #hide
@bs fn1(positions_ra, masses_ra, 2.0f0)
```

Runtime for raised function:

```@example raising_stablehlo
fn2(positions_ra, masses_ra, 2.0f0) #hide
@bs fn2(positions_ra, masses_ra, 2.0f0)
```
