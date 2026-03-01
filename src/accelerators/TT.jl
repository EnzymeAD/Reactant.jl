module TT

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads: Downloads
using p7zip_jll: p7zip

using ..Registration: register_backend

const tt_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)
const tt_pjrt_plugin_name = Ref{String}("pjrt_plugin_tt.so")

function setup_correct_env_vars!()
    # The env var `TT_METAL_RUNTIME_ROOT` must be set before creating the client.
    tt_metal_runtime_root = get(ENV, "TT_METAL_RUNTIME_ROOT", nothing)
    if tt_metal_runtime_root === nothing
        tt_metal_path_in_wheel = joinpath(dirname(get_tt_pjrt_plugin_path()), "tt-metal")
        if ispath(tt_metal_path_in_wheel)
            @debug "Setting environment variable 'TT_METAL_RUNTIME_ROOT' to \
                    '$(tt_metal_path_in_wheel)'"
            ENV["TT_METAL_RUNTIME_ROOT"] = tt_metal_path_in_wheel
        else
            error(
                "`TT_METAL_RUNTIME_ROOT` environment variable not set and we could not automatically determine it",
            )
        end
    else
        @debug "Environment variable 'TT_METAL_RUNTIME_ROOT' already set to to \
                '$(tt_metal_runtime_root)'"
    end
end

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert node_id == 0 "`make_pjrt_client` does not support node_id"
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`make_pjrt_client` does not \
                                                    support distributed_runtime_client"

    return Reactant.XLA.PJRT.MakeClientUsingPluginAPI(get_tt_pjrt_plugin_path(), "tt", "TT")
end

function make_ifrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    return Reactant.XLA.IFRT.MakeIFRTPJRTClientViaPluginAPI(
        get_tt_pjrt_plugin_path(),
        "tt",
        "TT";
        node_id,
        num_nodes,
        distributed_runtime_client,
    )
end

function __init__()
    if Sys.islinux() && has_tt() && !Reactant.precompiling()
        register_backend(
            "tt";
            priority=1000,
            pjrt_initialize_function=make_pjrt_client,
            ifrt_initialize_function=make_ifrt_client,
            preinitialize_setup_function=() -> begin
                setup_tt_pjrt_plugin!()
                setup_correct_env_vars!()
                nothing
            end,
        )
    end
end

force_tt_init() = haskey(ENV, "REACTANT_FORCE_TT_INIT")

function has_tt()
    if force_tt_init()
        return true
    end

    # To find whether we have Tenstorrent devices, we can either
    #
    # * look for devices in `/dev/tenstorrent`, or
    # * look for devices in `/sys/bus/pci/devices` with `vendor` equal to `0x1e52`, something like
    #       any(readchomp(joinpath(dir, "vendor")) == "0x1e52" for dir in readdir("/sys/bus/pci/devices"; join=true))
    #
    # The former is simpler for our current purposes, so we can go that way.
    dev_tt = "/dev/tenstorrent"
    return isdir(dev_tt) && length(readdir(dev_tt)) > 0
end

function setup_tt_pjrt_plugin!()
    plugin_dir_from_env = get(ENV, "TT_PJRT_PLUGIN_DIR", nothing)
    if plugin_dir_from_env !== nothing && ispath(plugin_dir_from_env)
        tt_pjrt_plugin_dir[] = plugin_dir_from_env
    else
        tt_pjrt_plugin_dir[] = @get_scratch!("pjrt_plugin_tt")
    end
    download_tt_pjrt_plugin_if_needed(tt_pjrt_plugin_dir[])
    return nothing
end

get_tt_pjrt_plugin_dir() = tt_pjrt_plugin_dir[]

function get_tt_pjrt_plugin_path()
    return joinpath(get_tt_pjrt_plugin_dir(), tt_pjrt_plugin_name[])
end

function download_tt_pjrt_plugin_if_needed(dir=nothing)
    dir === nothing && (dir = get_tt_pjrt_plugin_dir())
    @assert dir !== nothing "tt_pjrt_plugin_dir is not set!"

    tt_pjrt_plugin_path = joinpath(dir, tt_pjrt_plugin_name[])
    if isfile(tt_pjrt_plugin_path)
        @debug "TT PJRT plugin already found in '$(tt_pjrt_plugin_path)', nothing to do"
    else
        @debug "Will install the TT PJRT plugin to '$(tt_pjrt_plugin_path)'"
        mktempdir() do tmp_dir
            # Index at https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/
            zip_file_path = joinpath(tmp_dir, "pjrt-plugin-tt.zip")
            wheel_url = if Sys.ARCH === :x86_64
                "https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-0.6.0.dev20251202-cp311-cp311-linux_x86_64.whl"
            else
                error("Unsupported architecture for TT PJRT plugin: $(Sys.ARCH)")
            end
            @debug "Downloading TT PJRT plugin from '$(wheel_url)'"
            Downloads.download(wheel_url, zip_file_path)
            run(pipeline(`$(p7zip()) x -tzip -o$(tmp_dir) -- $(zip_file_path)`, devnull))
            data_dir = only(filter!(endswith(".data"), readdir(tmp_dir; join=true)))
            # We need to move the entire `pjrt_plugin_tt` directory to the destination.
            mv(joinpath(data_dir, "purelib", "pjrt_plugin_tt"), dir; force=true)
        end
        @assert isfile(tt_pjrt_plugin_path)
    end
end

end # module TT
