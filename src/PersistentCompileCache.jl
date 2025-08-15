module PersistentCompileCache

using Scratch: @get_scratch!
using Reactant_jll: Reactant_jll

const CACHE_DIR = Ref{Union{Nothing,String}}(nothing)
const KERNEL_CACHE_ENABLED = Ref(false)
const AUTOTUNE_CACHE_ENABLED = Ref(false)

# TODO: preference to disable persistent caching
# TODO: preference to control what is being cached.
# TODO: preference for cache dir
# TODO: clear cache function

function __init__()
    # We version our cache directory based on Reactant_jll version (technically we need to
    # version according to XLA, but this is a good enough proxy)
    version = pkgversion(Reactant_jll)
    CACHE_DIR[] = @get_scratch!(
        "xla_persistent_cache_$(version.major)_$(version.minor)_$(version.patch)"
    )
    KERNEL_CACHE_ENABLED[] = true
    AUTOTUNE_CACHE_ENABLED[] = true
    return nothing
end

function kernel_cache_enabled()
    return KERNEL_CACHE_ENABLED[] && CACHE_DIR[] !== nothing
end

function get_kernel_cache_path()
    kernel_cache_enabled() || return ""
    return joinpath(CACHE_DIR[], "xla_gpu_kernel_cache_file")
end

function autotune_cache_enabled()
    return AUTOTUNE_CACHE_ENABLED[] && CACHE_DIR[] !== nothing
end

function get_autotune_cache_directory()
    autotune_cache_enabled() || return ""
    dir = joinpath(CACHE_DIR[], "xla_gpu_per_fusion_autotune_cache_dir/")
    mkpath(dir)
    return dir
end

end
