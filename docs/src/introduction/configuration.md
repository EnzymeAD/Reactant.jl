# Configuration

When you [install](@ref Installation) `Reactant.jl`, the library powering the package compatible with your system will be automatically installed for you.
Below are some information about making sure that you are using the right configuration of Reactant for your machine.

## Reactant with CPU

At the moment Reactant supports only Linux (x86-64 and aarch64 architectures) and macOS (x86-64 and aarch64 architectures).
If you are using Julia on any of these systems, then Reactant should always support the CPU backend.
In the same environment where you installed Reactant you can verify it by running the following commands:

```julia-repl
julia> import Pkg

julia> Pkg.add("Reactant_jll")
  [...]

julia> import Reactant_jll

julia> Reactant_jll.is_available()
true
```

If the last command returns `true`, you are good to go, if you get `false` but you think your system is one of the supported ones listed above, [open an issue](https://github.com/EnzymeAD/Reactant.jl/issues/new/choose).

## Reactant with GPU

At the moment Reactant supports only Nvidia GPUs.

### Nvidia GPU

Reactant can accelerate your code using Nvidia GPUs on Linux, with CUDA Driver 12.1+ on x86-64, and CUDA Driver 12.3+ on aarch64.
You can check if Reactant detected the GPU on your system by running the following commands in the environment where you installed Reactant:

```julia-repl
julia> import Pkg

julia> Pkg.add("Reactant_jll")
  [...]

julia> import Reactant_jll

julia> Reactant_jll.is_available()
true

julia> Reactant_jll.host_platform
Linux x86_64 {cuda_version=12.1, cxxstring_abi=cxx11, gpu=cuda, julia_version=1.11.3, libc=glibc, libgfortran_version=5.0.0, libstdcxx_version=3.4.30, mode=opt}
```

Like in the CPU section above, we ran `Reactant_jll.is_available()` to make sure Reactant is available at all, the `Reactant_jll.host_platform` variable then gives us more information about the detected platform.
In particular, if you have an Nvidia GPU you should expect to see `gpu=cuda` and `cuda_version=X.Y`, where `X.Y` should be a version less than or equal to the version of the CUDA Driver present in your system (don't worry if you don't see here exactly the same version as your CUDA Driver, that is expected).

#### Debugging installation with Nvidia GPUs

In some cases you may want to get more verbose information from Reactant during its installation process, to see how it detected CUDA.
To do that, you can force re-installation of `Reactant_jll` with increased verbosity with the commands

```julia-repl
julia> rm(joinpath(Base.DEPOT_PATH[1], "compiled", "v$(VERSION.major).$(VERSION.minor)", "Reactant_jll"); recursive=true, force=true)

julia> ENV["JULIA_DEBUG"] = "Reactant_jll";

julia> import Pkg

julia> Pkg.add("Reactant_jll")
  [...]
  1 dependency had output during precompilation:
┌ Reactant_jll
│  ┌ Debug: Detected CUDA Driver version 12.2.0
│  └ @ Reactant_jll ~/.julia/packages/Reactant_jll/daenT/.pkg/platform_augmentation.jl:60
│  ┌ Debug: Adding include dependency on /lib/x86_64-linux-gnu/libcuda.so.1
│  └ @ Reactant_jll ~/.julia/packages/Reactant_jll/daenT/.pkg/platform_augmentation.jl:108
```

Here you can see that on this system Reactant found the CUDA Driver at `/lib/x86_64-linux-gnu/libcuda.so.1` with version 12.2.0.

#### Installing Reactant on GPU Servers without Internet

If you want to use Reactant on GPU Servers where all packages must be installed on the login nodes and the compute nodes don't have access to internet, add the following to the `Project.toml` and precompile the package:

```toml
[extras]
Reactant_jll = "0192cb87-2b54-54ad-80e0-3be72ad8a3c0"

[preferences.Reactant_jll]
gpu = "cuda"
```

#### Disabling CUDA support

Reactant looks for the CUDA Driver library `libcuda` to determine whether the current system supports Nvidia GPUs.
However in some cases this library may be actually present on the machine even though no GPU is actually attached to it, which would trick Reactant's installation process into believing a GPU is available.
Normally this is not a problem as Reactant will detect that in spite of the CUDA Driver being present there are no GPUs and will default to the CPU backend.
If you do experience issues due to a GPU being detected erroneously, you can force disabling GPU support by creating a file called `LocalPreferences.toml` in the environment where you installed Reactant with the following content:

```toml
[Reactant_jll]
gpu = "none"
```

install the package `Reactant_jll`:

```julia
import Pkg
Pkg.add("Reactant_jll")
```

and then when you restart Julia you should see

```julia-repl
julia> import Reactant_jll

julia> Reactant_jll.is_available()
true

julia> Reactant_jll.host_platform
Linux x86_64 {cuda_version=none, cxxstring_abi=cxx11, gpu=none, julia_version=1.11.3, libc=glibc, libgfortran_version=5.0.0, libstdcxx_version=3.4.30, mode=opt}
```

Reactant is still available for your system, but this time GPU support is disabled.

## Reactant with TPU

Reactant should detect automatically when you are running on a machine with a TPU, and load dynamically the necessary modules.
You can verify a TPU was found correctly with the following commands:

```julia-repl
julia> import Reactant

julia> Reactant.Accelerators.TPU.has_tpu()
true
```

### Memory errors on Google Cloud Platform

If you are running Julia on Google Cloud Platform, you may frequently get scary-looking memory-related error messages like:

```
double free or corruption (out)
```

or

```
free(): invalid pointer
```

This is due to the fact that in this environment a memory allocator incompatible with Julia is forced via the `LD_PRELOAD` environment variable.
Starting Julia with

```sh
LD_PRELOAD='' julia
```

or unsetting the variable

```sh
unset LD_PRELOAD
```

should solve this issue.
