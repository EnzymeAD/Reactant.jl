# [Multi-Host Environments](@ref distributed)

!!! tip "Use XLA IFRT Runtime"

    While PJRT does support some minimal distributed capabilities on CUDA GPUs, distributed
    support in Reactant is primarily provided via IFRT. Before loading Reactant, set the
    "xla_runtime" preference to be "IFRT". This can be done with:

    ```julia
    using Preferences, UUIDs

    Preferences.set_preference!(
        UUID("3c362404-f566-11ee-1572-e11a4b42c853"),
        "xla_runtime" => "IFRT"
    )
    ```

At the top of your code, just after loading Reactant and before running any Reactant related
operations, run `Reactant.Distributed.initialize()`.

!!! tip "Enable debug logging for debugging"

    Reactant emits a lot of useful debugging information when setting up the Distributed
    Runtime. This can be printing by setting the env var `JULIA_DEBUG` to contain
    `Reactant`.

After this simply setup your code with [`Reactant.Sharding`](@ref sharding-api) and the code
will run on multiple devices across multiple nodes.

## Example Slurm Script for Multi-Host Matrix Multiplication

::: code-group

```bash [main.sbatch]
#!/bin/bash -l
#
#SBATCH --job-name=matmul-sharding-reactant
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --account=<account>
#SBATCH --constraint=gpu

export JULIA_DEBUG="Reactant,Reactant_jll"
# Important else XLA might hang indefinitely
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

srun --preserve-env julia --project=. --threads=auto matmul_sharded.jl
```

```julia [matmul_sharded.jl]
using Reactant

Reactant.Distributed.initialize(; single_gpu_per_process=false)

@assert length(Reactant.devices()) >= 2

N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
sharding = Sharding.NamedSharding(mesh, (:x, :y))

x = reshape(collect(Float32, 1:64), 8, 8)
y = reshape(collect(Float32, 1:64), 8, 8)

x_ra = Reactant.to_rarray(x; sharding)
y_ra = Reactant.to_rarray(y; sharding)

res = @jit x_ra * y_ra

display(res)
```

:::
