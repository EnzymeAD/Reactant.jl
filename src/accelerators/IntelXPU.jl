module IntelXPU

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads
using Libdl

const intel_xpu_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)

function __init__()
    @static if Sys.ARCH === :x86_64
        if !Reactant.precompiling()
            setup_intel_xpu_pjrt_plugin!()

            try
                Libdl.dlopen(joinpath(get_intel_xpu_pjrt_plugin_dir(), "sycl_onednn.so");)
            catch e
                @debug "Failed to load sycl_onednn.so: $e"
            end
        end
    end
end

function setup_intel_xpu_pjrt_plugin!()
    path_from_env = get(ENV, "INTEL_XPU_LIBRARY_PATH", nothing)
    if path_from_env !== nothing && ispath(path_from_env)
        intel_xpu_pjrt_plugin_dir[] = path_from_env
    else
        intel_xpu_pjrt_plugin_dir[] = @get_scratch!("pjrt_intel_xpu_plugin")
    end
    download_intel_xpu_pjrt_plugin_if_needed(intel_xpu_pjrt_plugin_dir[])
    return nothing
end

get_intel_xpu_pjrt_plugin_dir() = intel_xpu_pjrt_plugin_dir[]

function get_intel_xpu_pjrt_plugin_path()
    return joinpath(get_intel_xpu_pjrt_plugin_dir(), "pjrt_plugin_xpu.so")
end

function download_intel_xpu_pjrt_plugin_if_needed(path=nothing)
    path === nothing && (path = get_intel_xpu_pjrt_plugin_dir())
    @assert path !== nothing "intel_xpu_pjrt_plugin_dir is not set!"

    intel_xpu_pjrt_plugin_path = joinpath(path, "pjrt_plugin_xpu.so")
    if !isfile(intel_xpu_pjrt_plugin_path)
        zip_file_path = joinpath(path, "pjrt-plugin-intel-xpu.zip")
        tmp_dir = joinpath(path, "tmp")
        Downloads.download(
            if Sys.ARCH === :x86_64
                "https://files.pythonhosted.org/packages/42/28/26564ea0937ec11755e63ab3c85d6d4b96201131a69c6fddf8b985e7f9ae/intel_extension_for_openxla-0.6.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
            else
                error("Unsupported architecture: $(Sys.ARCH)")
            end,
            zip_file_path,
        )
        run(`unzip -qq $(zip_file_path) -d $(tmp_dir)`)
        mv(
            joinpath(
                tmp_dir, "jax_plugins", "intel_extension_for_openxla", "pjrt_plugin_xpu.so"
            ),
            intel_xpu_pjrt_plugin_path;
        )
        mv(
            joinpath(
                tmp_dir,
                "jax_plugins",
                "intel_extension_for_openxla",
                "service",
                "gpu",
                "sycl_onednn.so",
            ),
            joinpath(path, "sycl_onednn.so");
        )
        rm(tmp_dir; recursive=true)
        rm(zip_file_path; recursive=true)
    end
end

end
