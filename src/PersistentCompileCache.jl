module PersistentCompileCache

using ..Reactant: Reactant

using Preferences: load_preference
using Scratch: @get_scratch!
using Reactant_jll: Reactant_jll

const CACHE_DIR = Ref{Union{Nothing,String}}(nothing)
const KERNEL_CACHE_ENABLED = Ref(false)
const AUTOTUNE_CACHE_ENABLED = Ref(false)

function __init__()
    persistent_cache_enabled = load_preference(Reactant, "persistent_cache_enabled", true)
    persistent_cache_directory = load_preference(Reactant, "persistent_cache_directory", "")

    if persistent_cache_enabled
        if isempty(persistent_cache_directory)
            # We version our cache directory based on Reactant_jll version (technically we
            # need to version according to XLA, but this is a good enough proxy)
            version = pkgversion(Reactant_jll)
            CACHE_DIR[] = @get_scratch!(
                "xla_persistent_cache_$(version.major)_$(version.minor)_$(version.patch)"
            )
        else
            CACHE_DIR[] = persistent_cache_directory
        end
        @debug "Persistent compilation cache enabled. Using base directory: $(CACHE_DIR[])"

        KERNEL_CACHE_ENABLED[] = load_preference(
            Reactant, "persistent_kernel_cache_enabled", false
        )
        @debug "Kernel cache enabled: $(KERNEL_CACHE_ENABLED[])"

        AUTOTUNE_CACHE_ENABLED[] = load_preference(
            Reactant, "persistent_autotune_cache_enabled", true
        )
        @debug "Autotune cache enabled: $(AUTOTUNE_CACHE_ENABLED[])"
    else
        @debug "Persistent compilation cache disabled..."
    end

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

"""
    clear_compilation_cache!()

Deletes the compilation cache directory. This removes all cached compilation artifacts for
all past versions of Reactant_jll.
"""
function clear_compilation_cache!()
    (CACHE_DIR[] !== nothing) && rm(CACHE_DIR[]; recursive=true, force=true)

    for dir in readdir(dirname(@get_scratch!("test_dir")); join=true)
        if isdir(dir) && startswith(basename(dir), "xla_persistent_cache")
            @debug "Removing cache directory: $dir"
            rm(dir; recursive=true, force=true)
        end
    end

    return nothing
end

export clear_compilation_cache!

end

using .PersistentCompileCache

export clear_compilation_cache!
