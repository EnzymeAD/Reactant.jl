module Metal

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads: Downloads
using p7zip_jll: p7zip

const metal_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)

# function __init__()
#     @static if Sys.isapple()
#         Reactant.precompiling() || setup_metal_pjrt_plugin!()
#     end
# end

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
                "https://files.pythonhosted.org/packages/09/dc/6d8fbfc29d902251cf333414cf7dcfaf4b252a9920c881354584ed36270d/jax_metal-0.1.1-py3-none-macosx_13_0_arm64.whl"
            elseif Sys.ARCH === :x86_64
                "https://files.pythonhosted.org/packages/87/ec/9bb7f7f0ffd06c3fb89813126b2f698636ac7a4263ed7bdd1ff7d7c94f8f/jax_metal-0.1.1-py3-none-macosx_10_14_x86_64.whl"
            else
                error("Unsupported architecture: $(Sys.ARCH)")
            end,
            zip_file_path,
        )
        run(pipeline(`$(p7zip()) x -tzip -o$(tmp_dir) -- $(zip_file_path)`, devnull))
        mv(
            joinpath(tmp_dir, "jax_plugins", "metal_plugin", "pjrt_plugin_metal_14.dylib"),
            metal_pjrt_plugin_path,
        )
        rm(tmp_dir; recursive=true)
        rm(zip_file_path; recursive=true)
    end
end

end
