# [Automatic Sharding-based Distributed Parallelism](@id sharding)

!!! tip "Use XLA IFRT Runtime"

    While PJRT does support some minimal sharding capabilities on CUDA GPUs, sharding
    support in Reactant is primarily provided via IFRT. Before loading Reactant, set the
    "xla_runtime" preference to be "IFRT". This can be done with:

    ```julia
    using Preferences, UUIDs

    Preferences.set_preference!(
        UUID("3c362404-f566-11ee-1572-e11a4b42c853"),
        "xla_runtime" => "IFRT"
    )
    ```

Sharding is one mechanism supported within Reactant that tries to make it easy to program for multiple devices (including [multiple nodes](@ref distributed)).

```@example sharding_tutorial
using Reactant

@assert length(Reactant.devices()) > 1 # hide
Reactant.devices()
```

Sharding provides Reactant users a [PGAS (parallel-global address space)](https://en.wikipedia.org/wiki/Partitioned_global_address_space) programming model. Let's understand what this means through example.

Suppose we have a function that takes a large input array and computes sin for all elements of the array.

```@example sharding_tutorial
function big_sin(data)
    data .= sin(data)
    return nothing
end

N = 100
x = Reactant.to_array(collect(1:N))

compiled_big_sin = @compile big_sin(x)

compiled_big_sin(x)
```

This successfully allocates the array `x` on one device, and executes it on the same device. However, suppose we want to execute this computation on multiple devices. Perhaps this is because the size of our inputs (`N`) is too large to fit on a single device. Or alternatively the function we execute is computationally expensive and we want to leverage the computing power of multiple devices.

Unlike more explicit communication libraries like MPI, the sharding model used by Reactant aims to let you execute a program on multiple devices without significant modifications to the single-device program. In particular, you do not need to write explicit communication calls (e.g. `MPI.Send` or `MPI.Recv`). Instead you write your program as if it executes on a very large single-node and Reactant will automatically determine how to subdivide the data, computation, and required communication.

# TODO describe how arrays are the "global data arrays, even though data is itself only stored on relevant device and computation is performed only devices with the required data (effectively showing under the hood how execution occurs)

# TODO simple case that demonstrates send/recv within (e.g. a simple neighbor add)


# TODO make a simple conway's game of life, or heat equation using sharding simulation example to show how a ``typical MPI'' simulation can be written using sharding.

## Simple 1-Dimensional Heat Equation

::: code-group

```julia [MPI Based Parallelism]
function one_dim_heat_equation_time_step_mpi!(data)
    id = MPI.Comm_rank(MPI.COMM_WORLD)
    last_id = MPI.Comm_size(MPI.COMM_WORLD)

    # Send data right
    if id > 1
        MPI.Send(@view(data[end]), MPI.COMM_WORLD; dest=id + 1)
    end

    # Recv data from left
    if id != last_id
        MPI.Recv(@view(data[1]), MPI.COMM_WORLD; dest=id - 1)
    end

    # 1-D Heat equation x[i, t] = 0.x * [i, t-1] + 0.25 * x[i-1, t-1] + 0.25 * x[i+1, t-1]
    data[2:end-1] .= 0.5 * data[2:end-1] + 0.25 * data[1:end-2] + 0.25 * data[3:end]

    return nothing
end


# Total size of grid we want to simulate
N = 100

# Local size of grid (total size divided by number of MPI devices)
_local = N / MPI.Comm_size(MPI.COMM_WORLD)

# We add two to add a left side padding and right side padding, necessary for storing
# boundaries from other nodes
data = rand(_local + 2)

function simulate(data, time_steps)
    for i in 1:time_steps
        one_dim_heat_equation_time_step_mpi!(data)
    end
end

simulate(data, 100)
```

```julia [Sharded Parallelism]
function one_dim_heat_equation_time_step_sharded!(data)
    # No send recv's required

    # 1-D Heat equation x[i, t] = 0.x * [i, t-1] + 0.25 * x[i-1, t-1] + 0.25 * x[i+1, t-1]
    # Reactant will automatically insert send and recv's
    data[2:end-1] .= 0.5 * data[2:end-1] + 0.25 * data[1:end-2] + 0.25 * data[3:end]

    return nothing
end


# Total size of grid we want to simulate
N = 100

# Reactant's sharding handles distributing the data amongst devices, with each device
# getting a corresponding fraction of the data
data = Reactant.to_rarray(
    rand(N + 2);
    sharding=Sharding.NamedSharding(
        Sharding.Mesh(Reactant.devices(), (:x,)),
        (:x,)
    )
)

function simulate(data, time_steps)
    @traced for i in 1:time_steps
        one_dim_heat_equation_time_step_sharded!(data)
    end
end

@jit simulate(data, 100)
```

:::

# TODO describe generation of distributed array by concatenating local-worker data


# TODO more complex tutorial describing replicated

## Sharding in Neural Networks

### 8-way Batch Parallelism

### 4-way Batch & 2-way Model Parallelism

## Related links

<!-- shardy? https://openxla.org/shardy -->
<!-- https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html -->
<!-- https://colab.research.google.com/drive/1UobcFjfwDI3N2EXvH3KbRS5ZxY9Riy4y#scrollTo=IiR7-0nDLPKK -->
