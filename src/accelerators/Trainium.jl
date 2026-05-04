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

    # Create wrapper script for neuronx-cc if it doesn't exist
    wrapper_path = joinpath(python_packages_dir, "neuronx-cc")
    if !isfile(wrapper_path)
        mkpath(dirname(wrapper_path))
        open(wrapper_path, "w") do io
            println(io, "#!/bin/bash")
            println(io, "export PYTHONPATH=\"$(escape_string(python_packages_dir)):\$PYTHONPATH\"")
            println(io, "python3 -m neuronxcc \"\$@\"")
        end
        chmod(wrapper_path, 0o755)
    end
    
    # Create a dummy libneuronxla module with expected attributes
    py_code = """
import sys
import os
import runpy
import subprocess

plugin_dir = '$(escape_string(plugin_dir))'
target_dir = '$(escape_string(python_packages_dir))'

sys.path.append(plugin_dir)
sys.path.append(target_dir)



# Ensure libneuronxla and neuronx-cc are installed using pip.pyz
try:
    import libneuronxla
    print('libneuronxla already available')
except ImportError:
    print('libneuronxla not found, installing using pip.pyz...')
    pip_pyz_path = os.path.join(plugin_dir, 'pip.pyz')
    if not os.path.exists(pip_pyz_path):
        raise FileNotFoundError(f"pip.pyz not found at {pip_pyz_path}")
        
    old_argv = sys.argv
    # Install libneuronxla and neuronx-cc from AWS Neuron repo
    sys.argv = ['pip', 'install', '--target', target_dir, '--extra-index-url', 'https://pip.repos.neuron.amazonaws.com/', 'libneuronxla', 'neuronx-cc']
    try:
        runpy.run_path(pip_pyz_path, run_name='__main__')
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f'pip failed with code {e.code}')
    finally:
        sys.argv = old_argv
    
    import importlib
    importlib.invalidate_caches()
    
    import libneuronxla
    print('libneuronxla installed successfully')

# Configure environment
libneuronxla.configure_environment()
libneuronxla.hook()
print('Called configure_environment and hook')
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
    # Check if it is in the scratch space under python_packages
    pip_path = joinpath(get_trainium_pjrt_plugin_dir(), "python_packages", "libneuronxla", trainium_pjrt_plugin_name[])
    if isfile(pip_path)
        return pip_path
    end
    return joinpath(get_trainium_pjrt_plugin_dir(), trainium_pjrt_plugin_name[])
end

function download_trainium_pjrt_plugin_if_needed(dir=nothing)
    dir === nothing && (dir = get_trainium_pjrt_plugin_dir())
    @assert dir !== nothing "trainium_pjrt_plugin_dir is not set!"

    pip_path = joinpath(dir, "pip.pyz")
    if isfile(pip_path)
        @debug "pip.pyz already found in '$(pip_path)', nothing to do"
    else
        mkpidlock(joinpath(dir, "download_pip.lock")) do
            if !isfile(pip_path)
                @debug "Downloading pip.pyz to '$(pip_path)'"
                pip_url = "https://bootstrap.pypa.io/pip/pip.pyz"
                Downloads.download(pip_url, pip_path)
            end
        end
    end
end

end # module Trainium
