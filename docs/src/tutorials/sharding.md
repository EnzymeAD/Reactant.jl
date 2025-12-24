# [Automatic Sharding-based Distributed Parallelism](@id sharding)

!!! tip "Use XLA IFRT Runtime"

    While PJRT does support some minimal sharding capabilities on CUDA GPUs, sharding
    support in Reactant is primarily provided via IFRT. Before loading Reactant, set the
    "xla_runtime" preference to be "IFRT". This can be done with:

    ```julia
    using Preferences, UUIDs

    Preferences.set_preferences!(
        UUID("3c362404-f566-11ee-1572-e11a4b42c853"),
        "xla_runtime" => "IFRT"
    )
    ```

## Basics

Sharding is one mechanism supported within Reactant that tries to make it easy to program
for multiple devices (including [multiple nodes](@ref distributed)).

```@example sharding_tutorial
using Reactant

@assert length(Reactant.devices()) > 1 # hide
Reactant.devices()
```

Sharding provides Reactant users a
[PGAS (parallel-global address space)](https://en.wikipedia.org/wiki/Partitioned_global_address_space)
programming model. Let's understand what this means through example.

Suppose we have a function that takes a large input array and computes sin for all elements
of the array.

```@example sharding_tutorial
function big_sin(data)
    data .= sin.(data)
    return nothing
end

N = 1600
x = Reactant.to_rarray(reshape(collect(Float32, 1:N), 40, 40))

compiled_big_sin = @compile big_sin(x)

compiled_big_sin(x)
```

This successfully allocates the array `x` on one device, and executes it on the same device.
However, suppose we want to execute this computation on multiple devices. Perhaps this is
because the size of our inputs (`N`) is too large to fit on a single device. Or
alternatively the function we execute is computationally expensive and we want to leverage
the computing power of multiple devices.

Unlike more explicit communication libraries like MPI, the sharding model used by Reactant
aims to let you execute a program on multiple devices without significant modifications to
the single-device program. In particular, you do not need to write explicit communication
calls (e.g. `MPI.Send` or `MPI.Recv`). Instead you write your program as if it executes on a
very large single-node and Reactant will automatically determine how to subdivide the data,
computation, and required communication.

When using sharding, the one thing you need to change about your code is how arrays are
allocated. In particular, you need to specify how the array is partitioned amongst available
devices. For example, suppose you are on a machine with 4 GPUs. In the example above, we
computed `sin` for all elements of a 40x40 grid. One partitioning we could select is to have
it partitioned along the first axis, such that each GPU has a slice of 10x40 elements. We
could accomplish this as follows. No change is required to the original function. However,
the compiled function is specific to the sharding so we need to compile a new version for
our sharded array.

```@example sharding_tutorial
N = 1600

x_sharded_first = Reactant.to_rarray(
    reshape(collect(1:N), 40, 40),
    sharding=Sharding.NamedSharding(
        Sharding.Mesh(reshape(Reactant.devices()[1:4], 4, 1), (:x, :y)),
        (:x, nothing)
    )
)

compiled_big_sin_sharded_first = @compile big_sin(x_sharded_first)

compiled_big_sin_sharded_first(x_sharded_first)
```

Alternatively, we can parition the data in a different form. In particular, we could
subdivide the data on both axes. As a result each GPU would have a slice of 20x20 elements.
Again no change is required to the original function, but we would change the allocation as
follows:

```@example sharding_tutorial
N = 1600
x_sharded_both = Reactant.to_rarray(
    reshape(collect(1:N), 40, 40),
    sharding=Sharding.NamedSharding(
        Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), (:x, :y)),
        (:x, :y)
    )
)

compiled_big_sin_sharded_both = @compile big_sin(x_sharded_both)

compiled_big_sin_sharded_both(x_sharded_both)
```

Sharding in reactant requires you to specify how the data is sharded across devices on a
mesh. We start by specifying the mesh [`Sharding.Mesh`](@ref) which is a collection of the
devices reshaped into an N-D grid. Additionally, we can specify names for each axis of the
mesh, that are then referenced when specifying how the data is sharded.

1. `Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), (:x, :y))`: Creates a 2D grid of 4
   devices arranged in a 2x2 grid. The first axis is named `:x` and the second axis is named
   `:y`.
2. `Sharding.Mesh(reshape(Reactant.devices()[1:4], 4, 1), (:x, :y))`: Creates a 2D grid of 4
   devices arranged in a 4x1 grid. The first axis is named `:x` and the second axis is
   named `:y`.

Given the mesh, we will specify how the data is sharded across the devices.

## Gradients 

It is also possible to compute gradients of functions that are sharded. Here we show an example using the `Enzyme.jl` package.

```@example sharding_tutorial
using Enzyme

function compute_gradient(x_sharded_both)
    return Enzyme.gradient(Enzyme.ReverseWithPrimal, Enzyme.Const(big_sin), x_sharded_both)
end

@jit compute_gradient(x_sharded_both)
```

<!--
TODO describe how arrays are the "global data arrays, even though data is itself only stored
on relevant device and computation is performed only devices with the required data
(effectively showing under the hood how execution occurs)
-->

<!--
TODO make a simple conway's game of life, or heat equation using sharding simulation example
to show how a ``typical MPI'' simulation can be written using sharding.
-->

## Simple 1-Dimensional Heat Equation

So far we chose a function which was perfectly parallelizable (e.g. each elemnt of the array
only accesses its own data). Let's consider a more realistic example where an updated
element requires data from its neighbors. In the distributed case, this requires
communicating the data along the boundaries.

In particular, let's implement a one-dimensional
[heat equation](https://en.wikipedia.org/wiki/Heat_equation) simulation. In this code you
initialize the temperature of all points of the simulation and over time the code will
simulate how the heat is transfered across space. In particular points of high temperature
will transfer energy to points of low energy.

As an example, here is a visualization of a 2-dimensional heat equation:

![Heat Equation Animation](https://upload.wikimedia.org/wikipedia/commons/a/a9/Heat_eqn.gif)

<!-- TODO we should animate the above -- and even more ideally have one we generate ourselves. -->

To keep things simple, let's implement a 1-dimensional heat equation here. We start off with
an array for the temperature at each point, and will compute the next version of the
temperatures according to the equation
`x[i, t] = 0.x * [i, t-1] + 0.25 * x[i-1, t-1] + 0.25 * x[i+1, t-1]`.

Let's consider how this can be implemented with explicit MPI communication. Each node will
contain a subset of the total data. For example, if we simulate with 100 points, and have 4
devices, each device will contain 25 data points. We're going to allocate some extra room at
each end of the buffer to store the ``halo'', or the data at the boundary. Each time step
that we take will first copy in the data from its neighbors into the halo via an explicit
MPI send and recv call. We'll then compute the updated data for our slice of the data.

With sharding, things are a bit more simple. We can write the code as if we only had one
device. No explicit send or recv's are necessary as they will be added automatically by
Reactant when it deduces they are needed. In fact, Reactant will attempt to optimize the
placement of the communicatinos to minimize total runtime. While Reactant tries to do a
good job (which could be faster than an initial implementation -- especially for complex
codebases), an expert may be able to find a better placement of the communication.

The only difference for the sharded code again occurs during allocation. Here we explicitly
specify that we want to subdivide the initial grid of 100 amongst all devices. Analagously
if we had 4 devices to work with, each device would have 25 elements in its local storage.
From the user's standpoint, however, all arrays give access to the entire dataset.

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
    @trace for i in 1:time_steps
        one_dim_heat_equation_time_step_sharded!(data)
    end
end

@jit simulate(data, 100)
```

:::

MPI to send the data. between computers When using GPUs on different devices, one needs to
copy the data through the network via NCCL instead of the `cuda.

All devices from all nodes are available for use by Reactant. Given the topology of the
devices, Reactant will automatically determine the right type of communication primitive to
use to send data between the relevant nodes. For example, between GPUs on the same host
Reactant may use the faster `cudaMemcpy` whereas for GPUs on different nodes Reactant will
use NCCL.

One nice feature about how Reactant's handling of multiple devices is that you don't need to
specify how the data is transfered. The fact that you doesn't need to specify how the
communication is occuring enables code written with Reactant to be run on a different
topology. For example, when using multiple GPUs on the same host it might be efficient to
copy data using a `cudaMemcpy` to transfer between devices directly.

## Devices

You can query the available devices that Reactant can access as follows using
[`Reactant.devices`](@ref).

```@example sharding_tutorial
Reactant.devices()
```

Not all devices are accessible from each process for [multi-node execution](@ref multihost).
To query the devices accessible from the current process, use
[`Reactant.addressable_devices`](@ref).

```@example sharding_tutorial
Reactant.addressable_devices()
```

You can inspect the type of the device, as well as its properties.

<!-- TODO: Generating Distributed Data by Concatenating Local-Worker Data -->

<!-- TODO: Handling Replicated Tensors -->

<!-- TODO: Sharding in Neural Networks -->

<!-- TODO: 8-way Batch Parallelism -->

<!-- TODO: 4-way Batch & 2-way Model Parallelism -->

## Related links

1. [Shardy Documentation](https://openxla.org/shardy)
2. [Jax Documentation](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
3. [Jax Scaling Book](https://jax-ml.github.io/scaling-book/sharding/)
4. [HuggingFace Ultra Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
