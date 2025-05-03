module Metal

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads

const metal_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)

function __init__()
    @static if Sys.isapple()
        Reactant.precompiling() || setup_metal_pjrt_plugin!()
    end
end

function setup_metal_pjrt_plugin!()
    path_from_env = get(ENV, "METAL_LIBRARY_PATH", nothing)
    if path_from_env !== nothing && ispath(path_from_env)
        metal_pjrt_plugin_dir[] = path_from_env
    else
        metal_pjrt_plugin_dir[] = @get_scratch!("pjrt_metal_plugin")
    end
    download_metal_pjrt_plugin_if_needed(metal_pjrt_plugin_dir[])
    return nothing
end

get_metal_pjrt_plugin_dir() = metal_pjrt_plugin_dir[]

function get_metal_pjrt_plugin_path()
    return joinpath(get_metal_pjrt_plugin_dir(), "pjrt_plugin_metal_14.dylib")
end

function download_metal_pjrt_plugin_if_needed(path=nothing)
    path === nothing && (path = get_metal_pjrt_plugin_dir())
    @assert path !== nothing "metal_pjrt_plugin_dir is not set!"

    metal_pjrt_plugin_path = joinpath(path, "pjrt_plugin_metal_14.dylib")
    if !isfile(metal_pjrt_plugin_path)
        zip_file_path = joinpath(path, "pjrt-plugin-metal.zip")
        tmp_dir = joinpath(path, "tmp")
        Downloads.download(
            if Sys.ARCH === :aarch64
                "https://files.pythonhosted.org/packages/80/af/ed482a421a868726e7ca3f51ac19b0c9a8e37f33f54413312c37e9056acc/jax_metal-0.1.0-py3-none-macosx_11_0_arm64.whl"
            elseif Sys.ARCH === :x86_64
                "https://files.pythonhosted.org/packages/51/6a/1c0e2d07d92c6583e874ef2bbf4382662a3469bbb661d885eeaaddca426f/jax_metal-0.1.0-py3-none-macosx_10_14_x86_64.whl"
            else
                error("Unsupported architecture: $(Sys.ARCH)")
            end,
            zip_file_path,
        )
        run(`unzip -qq $(zip_file_path) -d $(tmp_dir)`)
        mv(
            joinpath(tmp_dir, "jax_plugins", "metal_plugin", "pjrt_plugin_metal_14.dylib"),
            metal_pjrt_plugin_path,
        )
        rm(tmp_dir; recursive=true)
        rm(zip_file_path; recursive=true)
    end
end

end
