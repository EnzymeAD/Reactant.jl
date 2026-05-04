module Trainium

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads: Downloads
using p7zip_jll: p7zip
using FileWatching: mkpidlock

using Libdl

const TRAINIUM_WHEEL = "libneuronxla-2.2.16408.0%2B50c26cbd-py3-none-linux_x86_64.whl"
const PYTHON_LIB = "/usr/lib/python3.10/config-3.10-x86_64-linux-gnu/libpython3.10.so"

using ..Registration: register_backend

const trainium_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)
const trainium_pjrt_plugin_name = Ref{String}("libneuronpjrt.so")

function setup_correct_env_vars!()
    # From libneuronxla/__init__.py
    if !haskey(ENV, "XLA_IR_SHAPE_CACHE_SIZE")
        ENV["XLA_IR_SHAPE_CACHE_SIZE"] = "20480"
    end
    if haskey(ENV, "XLA_USE_BF16") || haskey(ENV, "XLA_DOWNCAST_BF16")
        if !haskey(ENV, "NEURON_RT_STOCHASTIC_ROUNDING_EN")
            ENV["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"
        end
    end
end

    plugin_dir = get_trainium_pjrt_plugin_dir()
    
    # Create a dummy libneuronxla module with expected attributes
    py_code = """
import sys
import os

# Add path to the scratch directory where libneuronxla is extracted
sys.path.append('$(escape_string(plugin_dir))')

try:
    import libneuronxla
    print('Imported real libneuronxla successfully')
    
    # Call configure_environment and hook as required by the library
    libneuronxla.configure_environment()
    libneuronxla.hook()
    print('Called configure_environment and hook')
except ImportError as e:
    print(f'Failed to import real libneuronxla: {e}. Falling back to dummy module.')
    import types
    import socket
    
    mod = types.ModuleType('libneuronxla')
    
    def dummy_callback(name, addressable_device_index, execution_count):
        return 'inputs'
    mod._dump_hlo_snapshot_callback = dummy_callback
    
    def dummy_configure():
        pass
    mod.configure_environment = dummy_configure
    
    def dummy_hook():
        with socket.socket() as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        os.environ['NEURON_RT_ROOT_COMM_ID'] = f'localhost:{port}'
        print(f'Dummy hook called, set NEURON_RT_ROOT_COMM_ID to localhost:{port}')
        pass
    mod.hook = dummy_hook
    
    def dummy_neuronx_cc(code, code_format, platform_version, file_prefix):
        print('Dummy neuronx_cc called')
        return 0, b''
    mod.neuronx_cc = dummy_neuronx_cc
    
    sys.modules['libneuronxla'] = mod
    print('Dummy libneuronxla module with callbacks, hook, and neuronx_cc created and registered')
"""

    ccall(:PyRun_SimpleString, Cint, (Cstring,), py_code)

    return Reactant.XLA.PJRT.MakeClientUsingPluginAPI(get_trainium_pjrt_plugin_path(), "trainium", "Trainium")
end

function make_ifrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    if allowed_devices !== nothing
        @debug "TrainiumClient doesn't support allowed_devices. Ignoring the kwarg."
    end

    return Reactant.XLA.IFRT.MakeIFRTPJRTClientViaPluginAPI(
        get_trainium_pjrt_plugin_path(),
        "trainium",
        "Trainium";
        node_id,
        num_nodes,
        distributed_runtime_client,
    )
end

function __init__()
    if Sys.islinux() && has_trainium() && !Reactant.precompiling()
        register_backend(
            "trainium";
            priority=1000,
            pjrt_initialize_function=make_pjrt_client,
            ifrt_initialize_function=make_ifrt_client,
            preinitialize_setup_function=() -> begin
                setup_trainium_pjrt_plugin!()
                setup_correct_env_vars!()
                nothing
            end,
        )
    end
end

force_trainium_init() = haskey(ENV, "REACTANT_FORCE_TRAINIUM_INIT")

function has_trainium()
    if force_trainium_init()
        return true
    end
    return Sys.islinux() && any(startswith("neuron"), readdir("/dev"))
end

function setup_trainium_pjrt_plugin!()
    plugin_dir_from_env = get(ENV, "TRAINIUM_PJRT_PLUGIN_DIR", nothing)
    if plugin_dir_from_env !== nothing && ispath(plugin_dir_from_env)
        trainium_pjrt_plugin_dir[] = plugin_dir_from_env
    else
        # Check if we are in the development environment and have the file locally
        dev_path = joinpath(@__DIR__, "../../libneuronxla/libneuronpjrt.so")
        if isfile(dev_path)
            trainium_pjrt_plugin_dir[] = dirname(dev_path)
            return nothing
        end
        trainium_pjrt_plugin_dir[] = @get_scratch!("pjrt_plugin_trainium")
    end
    download_trainium_pjrt_plugin_if_needed(trainium_pjrt_plugin_dir[])
    return nothing
end

get_trainium_pjrt_plugin_dir() = trainium_pjrt_plugin_dir[]

function get_trainium_pjrt_plugin_path()
    dev_path = joinpath(@__DIR__, "../../libneuronxla/libneuronpjrt.so")
    if isfile(dev_path)
        return dev_path
    end
    # Check if it is in the scratch space as a directory
    dir_path = joinpath(get_trainium_pjrt_plugin_dir(), "libneuronxla", trainium_pjrt_plugin_name[])
    if isfile(dir_path)
        return dir_path
    end
    return joinpath(get_trainium_pjrt_plugin_dir(), trainium_pjrt_plugin_name[])
end

function download_trainium_pjrt_plugin_if_needed(dir=nothing)
    dir === nothing && (dir = get_trainium_pjrt_plugin_dir())
    @assert dir !== nothing "trainium_pjrt_plugin_dir is not set!"

    trainium_pjrt_plugin_path = joinpath(dir, trainium_pjrt_plugin_name[])
    if isfile(trainium_pjrt_plugin_path)
        @debug "Trainium PJRT plugin already found in '$(trainium_pjrt_plugin_path)', nothing to do"
    else
        mkpidlock(joinpath(dir, "download_trainium_pjrt_plugin.lock")) do
            if !isfile(trainium_pjrt_plugin_path)
                @debug "Will install the Trainium PJRT plugin to '$(trainium_pjrt_plugin_path)'"
                mktempdir() do tmp_dir
                    zip_file_path = joinpath(tmp_dir, "libneuronxla.zip")
                    wheel_url = "https://pip.repos.neuron.amazonaws.com/libneuronxla/$(TRAINIUM_WHEEL)"
                    @debug "Downloading Trainium PJRT plugin from '$(wheel_url)'"
                    Downloads.download(wheel_url, zip_file_path)
                    run(
                        pipeline(
                            `$(p7zip()) x -tzip -o$(tmp_dir) -- $(zip_file_path)`, devnull
                        ),
                    )
                    # Move the whole libneuronxla directory to dir
                    mv(joinpath(tmp_dir, "libneuronxla"), joinpath(dir, "libneuronxla"); force=true)
                end
            end
        end
        @assert isfile(trainium_pjrt_plugin_path)
    end
end

end # module Trainium
