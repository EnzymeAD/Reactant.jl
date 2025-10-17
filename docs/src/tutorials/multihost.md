# [Multi-Host Environments](@id distributed)

!!! tip "Use XLA IFRT Runtime"

    While PJRT does support some minimal distributed capabilities on CUDA GPUs, distributed
    support in Reactant is primarily provided via IFRT. Before loading Reactant, set the
    "xla_runtime" preference to be "IFRT". This can be done with:

    ```julia
    using Preferences, UUIDs

    Preferences.set_preferences!(
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

srun --preserve-env bash ./matmul.sh
```

```bash [matmul.sh]
#!/bin/bash -l

# Important else XLA might hang indefinitely
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

julia --project=. --threads=auto matmul_sharded.jl
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

## Example Google Cloud Setup for Multi-Host Matrix Multiplication on TPU v6

For more details lookup the details in the
[official Cloud TPU documentation](https://cloud.google.com/tpu/docs). For an introduction
to Cloud TPU MultiSlice refer to the
[official docs](https://cloud.google.com/tpu/docs/multislice-introduction).

!!! tip "Setup a Google Cloud Account"

    As a pre-requisite to this example, users need to setup a Google Cloud Account as
    detailed in
    [Set up the Cloud TPU environment](https://cloud.google.com/tpu/docs/setup-gcp-account).

First setup the Environment Variables based on the configuration that you want to run.

::: code-group

```bash [Single-Slice Multi-Host]
export QR_ID=sharded-single-slice-reactant-test # [!code highlight]
export PROJECT=<project name>
export ZONE=asia-northeast1-b
export RUNTIME_VERSION=v2-alpha-tpuv6e
export ACCELERATOR_TYPE=v6e-16
export SLICE_COUNT=1 # [!code highlight]
```

```bash [Multi-Slice Multi-Host]
export QR_ID=sharded-multi-slice-reactant-test # [!code highlight]
export PROJECT=<project name>
export ZONE=asia-northeast1-b
export RUNTIME_VERSION=v2-alpha-tpuv6e
export ACCELERATOR_TYPE=v6e-16
export SLICE_COUNT=2 # [!code highlight]
```

:::

::: code-group

```julia [Sharded Matrix Multiply]
using Reactant

Reactant.Distributed.initialize()

mesh = Sharding.Mesh(reshape(Reactant.devices(), :, 4), (:x, :y))
sharding = Sharding.NamedSharding(mesh, (:x, :y))

x = reshape(collect(Float32, 1:64), 8, 8)
y = reshape(collect(Float32, 1:64), 8, 8)

x_ra = Reactant.to_rarray(x; sharding)
y_ra = Reactant.to_rarray(y; sharding)
res = @jit x_ra * y_ra

display(res)
```

```bash [Bash Script]
# Remember to set the Environment Variables first

gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE

gcloud compute tpus queued-resources \
    create ${QR_ID} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --runtime-version ${RUNTIME_VERSION} \
    --node-count ${SLICE_COUNT}

# Create a Project.toml file and a LocalPreferences.toml
echo -e '[deps]\nReactant = "3c362404-f566-11ee-1572-e11a4b42c853"' > Project.toml
echo -e '[Reactant]\nxla_runtime = "IFRT"' > LocalPreferences.toml

# Copy these files to all the workers
gcloud compute tpus queued-resources scp ./LocalPreferences.toml ${QR_ID}: \
    --worker=all --node=all \
    --zone=${ZONE} --project=${PROJECT}
gcloud compute tpus queued-resources scp ./Project.toml ${QR_ID}: \
    --worker=all --node=all \
    --zone=${ZONE} --project=${PROJECT}

# Install Julia and Project Dependencies
gcloud compute tpus queued-resources ssh ${QR_ID} \
    --worker=all --node=all \
    --zone=${ZONE} --project=${PROJECT} \
    --command="
       wget --quiet https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz;
                tar xzf julia-1.11.5-linux-x86_64.tar.gz;
                rm julia-1.11.5-linux-x86_64.tar.gz;
                unset LD_PRELOAD;
                ./julia-1.11.5/bin/julia --project=. --threads=auto -e '
                    using Pkg;
                    Pkg.instantiate();
                    Pkg.precompile();'"

# Run the sharding code
gcloud compute tpus queued-resources ssh ${QR_ID} \
    --worker=all --node=all \
    --zone=${ZONE} --project=${PROJECT} \
    --command="LD_PRELOAD='' ./julia-1.11.4/bin/julia --project=. --threads=auto <code>"

# Don't forget to delete the queued resources
gcloud compute tpus queued-resources delete ${QR_ID} \
    --project ${PROJECT} --zone ${ZONE} --force --async
```

:::
