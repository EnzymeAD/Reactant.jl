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

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert node_id == 0 "`make_pjrt_client` does not support node_id"
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`make_pjrt_client` does not support distributed_runtime_client"

    if allowed_devices !== nothing
        @debug "TrainiumClient doesn't support allowed_devices. Ignoring the kwarg."
    end

    # Load the Python library globally
    Libdl.dlopen(PYTHON_LIB, Libdl.RTLD_GLOBAL)

    # Initialize the Python interpreter
    ccall((:Py_Initialize, PYTHON_LIB), Cvoid, ())

    plugin_dir = get_trainium_pjrt_plugin_dir()
    python_packages_dir = joinpath(plugin_dir, "python_packages")
    
    # Add to system PATH for subprocess calls to neuronx-cc
    if isdir(python_packages_dir)
        ENV["PATH"] = python_packages_dir * ":" * get(ENV, "PATH", "")
    end
    
    # Create a dummy libneuronxla module with expected attributes
    py_code = """
import sys
import os
import runpy

plugin_dir = '$(escape_string(plugin_dir))'
target_dir = '$(escape_string(python_packages_dir))'

sys.path.append(plugin_dir)
sys.path.append(target_dir)

# Ensure boto3 is installed using pip.pyz
try:
    import boto3
    print('boto3 already available')
except ImportError:
    print('boto3 not found, installing using pip.pyz...')
    pip_pyz_path = os.path.join(plugin_dir, 'pip.pyz')
    if not os.path.exists(pip_pyz_path):
        raise FileNotFoundError(f"pip.pyz not found at {pip_pyz_path}")
        
    old_argv = sys.argv
    sys.argv = ['pip', 'install', '--target', target_dir, 'boto3']
    try:
        runpy.run_path(pip_pyz_path, run_name='__main__')
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f'pip failed with code {e.code}')
    finally:
        sys.argv = old_argv
    print('boto3 installed successfully')

try:
    import libneuronxla
    print('Imported real libneuronxla successfully')
    
    # Call configure_environment and hook as required by the library
    libneuronxla.configure_environment()
    libneuronxla.hook()
    print('Called configure_environment and hook')
except ImportError as e:
    print(f'Failed to import real libneuronxla: {e}')
    raise e
"""

    ccall((:PyRun_SimpleString, PYTHON_LIB), Cint, (Cstring,), py_code)

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

    trainium_pjrt_plugin_path = joinpath(dir, "libneuronxla", trainium_pjrt_plugin_name[])
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
                    
                    # Download and unzip neuronx-cc wheel
                    neuronx_cc_url = "https://pip.repos.neuron.amazonaws.com/neuronx-cc/neuronx_cc-2.24.8799.0%2B6f62ff7c-cp310-cp310-linux_x86_64.whl"
                    neuronx_cc_zip = joinpath(tmp_dir, "neuronx_cc.zip")
                    @debug "Downloading neuronx-cc from '$(neuronx_cc_url)'"
                    Downloads.download(neuronx_cc_url, neuronx_cc_zip)
                    run(
                        pipeline(
                            `$(p7zip()) x -tzip -o$(tmp_dir)/neuronx_cc -- $(neuronx_cc_zip)`, devnull
                        ),
                    )
                    # Move content to dir/python_packages
                    python_packages_dir = joinpath(dir, "python_packages")
                    mkpath(python_packages_dir)
                    for f in readdir(joinpath(tmp_dir, "neuronx_cc"); join=true)
                        mv(f, joinpath(python_packages_dir, basename(f)); force=true)
                    end
                    
                    # Download pip.pyz
                    pip_url = "https://bootstrap.pypa.io/pip/pip.pyz"
                    @debug "Downloading pip.pyz from '$(pip_url)'"
                    Downloads.download(pip_url, joinpath(dir, "pip.pyz"))
                end
            end
        end
        @assert isfile(trainium_pjrt_plugin_path)
    end
end

end # module Trainium
