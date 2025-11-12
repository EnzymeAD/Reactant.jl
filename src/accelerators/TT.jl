module TT

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads

const tt_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)

function __init__()
    @static if Sys.islinux()
        Reactant.precompiling() || setup_tt_pjrt_plugin!()
    end
end

has_tt() = true

function setup_tt_pjrt_plugin!()
    path_from_env = get(ENV, "TT_LIBRARY_PATH", nothing)
    if path_from_env !== nothing && ispath(path_from_env)
        tt_pjrt_plugin_dir[] = path_from_env
    else
        tt_pjrt_plugin_dir[] = @get_scratch!("pjrt_tt_plugin")
    end
    # download_tt_pjrt_plugin_if_needed(tt_pjrt_plugin_dir[])
    return nothing
end

get_tt_pjrt_plugin_dir() = tt_pjrt_plugin_dir[]

function get_tt_pjrt_plugin_path()
    return joinpath(get_tt_pjrt_plugin_dir(), "pjrt_plugin_tt.so")
end

# function download_tt_pjrt_plugin_if_needed(path=nothing)
#     path === nothing && (path = get_tt_pjrt_plugin_dir())
#     @assert path !== nothing "tt_pjrt_plugin_dir is not set!"

#     tt_pjrt_plugin_path = joinpath(path, "pjrt_plugin_tt_14.dylib")
#     if !isfile(tt_pjrt_plugin_path)
#         zip_file_path = joinpath(path, "pjrt-plugin-tt.zip")
#         tmp_dir = joinpath(path, "tmp")
#         Downloads.download(
#             if Sys.ARCH === :aarch64
#                 "https://files.pythonhosted.org/packages/09/dc/6d8fbfc29d902251cf333414cf7dcfaf4b252a9920c881354584ed36270d/jax_tt-0.1.1-py3-none-macosx_13_0_arm64.whl"
#             elseif Sys.ARCH === :x86_64
#                 "https://files.pythonhosted.org/packages/87/ec/9bb7f7f0ffd06c3fb89813126b2f698636ac7a4263ed7bdd1ff7d7c94f8f/jax_tt-0.1.1-py3-none-macosx_10_14_x86_64.whl"
#             else
#                 error("Unsupported architecture: $(Sys.ARCH)")
#             end,
#             zip_file_path,
#         )
#         run(`unzip -qq $(zip_file_path) -d $(tmp_dir)`)
#         mv(
#             joinpath(tmp_dir, "jax_plugins", "tt_plugin", "pjrt_plugin_tt_14.dylib"),
#             tt_pjrt_plugin_path,
#         )
#         rm(tmp_dir; recursive=true)
#         rm(zip_file_path; recursive=true)
#     end
# end

end # module TT
