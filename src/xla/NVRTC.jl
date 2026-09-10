# cuDNN's runtime-compiled engines (fused attention / SDPA, runtime fusion) JIT their
# kernels with NVRTC when an execution plan is built. They do not call the NVRTC linked
# statically into libReactantExtra.so: cuDNN's loader dlopens a *shared* libnvrtc named by
# `CUDNN_NVRTC_OVERRIDE_PATH`, and with nothing to point it at every plan build fails with
# `CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED ... compilationResult != NVRTC_SUCCESS`,
# which XLA surfaces as "[cudnn_frontend] Error: No valid execution plans built.".
# The CUDA bundles ship one in `lib/cuda/lib` for exactly this purpose.

const CUDNN_NVRTC_OVERRIDE_ENV = "CUDNN_NVRTC_OVERRIDE_PATH"

# Shared CUDA libraries `Reactant_jll` ships next to its `ptxas` and `libdevice`. Returns
# `nothing` off-CUDA and on bundles predating the shipped NVRTC.
function _reactant_jll_cuda_lib(prefix::AbstractString)
    Reactant_jll.is_available() || return nothing
    dir = joinpath(dirname(Reactant_jll.libReactantExtra_path), "cuda", "lib")
    isdir(dir) || return nothing
    names = filter(n -> startswith(n, prefix), readdir(dir))
    isempty(names) && return nothing
    # Prefer the shortest soname (`libnvrtc.so.13` over `libnvrtc.so.13.1.115`).
    return joinpath(dir, first(sort(names; by=length)))
end

"""
    reactant_jll_libnvrtc()

Path to the shared NVRTC `Reactant_jll` ships in `lib/cuda/lib`, or `nothing`.
"""
reactant_jll_libnvrtc() = _reactant_jll_cuda_lib("libnvrtc.so.")

"""
    reactant_jll_libnvrtc_builtins()

Path to the `libnvrtc-builtins` matching [`reactant_jll_libnvrtc`](@ref), or `nothing`.
"""
reactant_jll_libnvrtc_builtins() = _reactant_jll_cuda_lib("libnvrtc-builtins.so.")

"""
    configure_cudnn_nvrtc_override!()

Point cuDNN's runtime-compiled engines at the shared NVRTC bundled in `Reactant_jll`.
Returns the override path in effect, or `nothing` when the bundle carries no NVRTC.

A `CUDNN_NVRTC_OVERRIDE_PATH` already set in the environment is left alone, so pointing
cuDNN somewhere else is just a matter of exporting that variable. Must run before cuDNN
builds its first runtime-compiled plan; cuDNN reads the variable once.
"""
function configure_cudnn_nvrtc_override!()
    haskey(ENV, CUDNN_NVRTC_OVERRIDE_ENV) && return ENV[CUDNN_NVRTC_OVERRIDE_ENV]
    libnvrtc = reactant_jll_libnvrtc()
    libnvrtc === nothing && return nothing
    builtins = reactant_jll_libnvrtc_builtins()
    if builtins !== nothing
        # libnvrtc has no RUNPATH and dlopens its builtins by soname, which resolves once
        # the library is loaded in the process -- no LD_LIBRARY_PATH entry needed.
        Libdl.dlopen(builtins, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    end
    ENV[CUDNN_NVRTC_OVERRIDE_ENV] = libnvrtc
    return libnvrtc
end
