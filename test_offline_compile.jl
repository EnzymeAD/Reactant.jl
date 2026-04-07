## Step 1: Offline compilation for 2 GPUs with sharding
#
# This script compiles a sharded function targeting 2 GPUs. The inputs
# are sharded across the available devices so the IR captures SPMD
# partitioning. The compiled executable is serialized to disk.

using Revise

using Reactant
using Reactant.Sharding: Mesh, NamedSharding
using Reactant.XLA.IFRT: GpuTopology

@show Reactant.devices()

Reactant.set_default_backend("cpu")

@show Reactant.devices()

# Build a mesh over the first 2 devices
devices = Reactant.devices()
@assert length(devices) >= 2 "Need at least 2 devices, got $(length(devices))"
mesh = Mesh(devices[1:2], (:x,))

# Shard the inputs along the first axis across the 2 devices
sharding = NamedSharding(mesh, (:x,))
x = Reactant.to_rarray(Float32[1, 2, 3, 4]; sharding)
y = Reactant.to_rarray(Float32[10, 20, 30, 40]; sharding)

# A simple computation
function add_and_scale(x, y)
    return (x .+ y) .* 2.0f0
end

# Target topology: 2 GPUs on 1 host
topology = GpuTopology(
    Reactant.XLA.client("gpu");
    num_partitions=2,
    num_devices_per_host=2,
    platform_version="12.3",
)

println("Compiling sharded function for 2 GPUs...")
Reactant.save_compiled_executable(
    "offline_2gpu.jls",
    add_and_scale,
    x,
    y;
    topology,
    #use_shardy_partitioner=false,
    #shardy_passes=:to_mhlo_shardings
    xla_debug_options=(; xla_gpu_experimental_aot_compiled_thunks=true),
)

println("Saved to offline_2gpu.jls")

# Verify with @code_hlo that the IR is actually sharded
println("\nIR for reference:")
println(@code_hlo add_and_scale(x, y))
println(@code_xla add_and_scale(x, y))

@jit add_and_scale(x, y)

println("\nDone! Run test_load_executable.jl to load and execute.")
