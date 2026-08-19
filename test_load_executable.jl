## Step 2: Load and run the pre-compiled sharded executable
#
# This script loads the serialized executable from disk and runs it
# directly on real GPUs — no recompilation needed.

using Reactant
using Reactant.Sharding: Mesh, NamedSharding

# Recreate the same sharded inputs
devices = Reactant.devices()
@assert length(devices) >= 2 "Need at least 2 devices, got $(length(devices))"
mesh = Mesh(devices[1:2], (:x,))
sharding = NamedSharding(mesh, (:x,))

x = Reactant.to_rarray(Float32[1, 2, 3, 4]; sharding)
y = Reactant.to_rarray(Float32[10, 20, 30, 40]; sharding)

println("Loading pre-compiled executable from offline_2gpu.jls...")
result = Reactant.load_compiled_executable("offline_2gpu.jls", x, y)

#println("Result: ", result)
#expected = (Float32[1, 2, 3, 4] .+ Float32[10, 20, 30, 40]) .* 2.0f0
#println("Expected: ", expected)
#@assert Array(result[1]) ≈ expected "Mismatch! Got $(Array(result[1]))"
#println("Success!")
