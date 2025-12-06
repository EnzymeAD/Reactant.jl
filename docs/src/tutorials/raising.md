# Raising

## Raising GPU Kernels

<!-- TODO: write this section -->

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
    disable_auto_batching_passes=true
) compute_attractive_force(positions_ra, masses_ra, 2.0f0)
```

This IR has a nested loop, but that won't work nicely for GPUs/TPUs. Even for CPUs, XLA
often doens't do a great job with loops. By default, we will attempt to raise loops to a
tensor IR.

```@example raising_stablehlo
hlo = @code_hlo compute_attractive_force(positions_ra, masses_ra, 2.0f0)
@assert !contains(repr(hlo), "stablehlo.while") #hide
hlo
```

This IR won't have any loops, instead it will be written in a tensor IR! Let ensure that
the values are identical.

```@example raising_stablehlo
y_jl = compute_attractive_force(positions, masses, 2.0f0)
y_ra = @jit compute_attractive_force(positions_ra, masses_ra, 2.0f0)
maximum(abs, Array(y_ra) .- y_jl)
```

Let's time the execution of the two versions.

```@example raising_stablehlo
fn1 = @compile sync=true compile_options=CompileOptions(;
    disable_auto_batching_passes=true
) compute_attractive_force(positions_ra, masses_ra, 2.0f0)
fn2 = @compile sync=true compute_attractive_force(positions_ra, masses_ra, 2.0f0)
```

Runtime for non-raised function:

```@example raising_stablehlo
@bs fn1(positions_ra, masses_ra, 2.0f0)
```

Runtime for raised function:

```@example raising_stablehlo
@bs fn2(positions_ra, masses_ra, 2.0f0)
```
