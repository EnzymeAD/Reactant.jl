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
    bin_dir = joinpath(python_packages_dir, "bin")
    ENV["PATH"] = bin_dir * ":" * get(ENV, "PATH", "")
    
    # Create a dummy libneuronxla module with expected attributes
    py_code = """
import sys
import os
import types
import subprocess

plugin_dir = '$(escape_string(plugin_dir))'
target_dir = '$(escape_string(python_packages_dir))'

sys.path.append(plugin_dir)
sys.path.append(target_dir)

# Add bin directory to system PATH in Python environment
bin_dir = os.path.join(target_dir, 'bin')
os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
os.environ['PYTHONPATH'] = target_dir + os.pathsep + os.environ.get('PYTHONPATH', '')

class GlobalCounter:
    _counter = 0
    def __call__(self):
        count = GlobalCounter._counter
        GlobalCounter._counter += 1
        return count

def _neuronx_cc_impl_fast(code, target):
    cmd = [
        'neuronx-cc',
        'compile',
        '--framework=XLA',
        f'--target={target}',
        '--verbose=35',
        '--enable-internal-neff-wrapper',
    ]
    flags = os.environ.get('NEURON_CC_FLAGS', '')
    flags = shlex.split(flags)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dump',
        default=None,
        help='Folder to dump neuronx-cc artifacts')
    args, flags = parser.parse_known_args(flags)
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.dump is not None:
            tmpdir = os.path.abspath(args.dump)
            tmpdir = os.path.join(
                tmpdir, f'pid{os.getpid()}-program{GlobalCounter()()}')
            os.makedirs(tmpdir, exist_ok=True)
            cmd.extend(['--pipeline', 'compile', 'SaveTemps'])
        hlo_module_path = os.path.join(tmpdir, 'file.code')
        with open(hlo_module_path, 'wb') as fp:
            fp.write(code)
        neff_path = os.path.join(tmpdir, 'file.neff')
        cmd.append(f'--output={neff_path}')
        cmd.append(hlo_module_path)
        cmd.extend(flags)
        if args.dump is not None:
            ver_cmd = ['neuronx-cc', '--version']
            ncc_version = subprocess.check_output(
                ver_cmd, stderr=subprocess.STDOUT).decode()
            ncc_version, *_ = ncc_version.split('\n')
            *_, ncc_version = ncc_version.split('version ')
            with open(os.path.join(tmpdir, 'neuronx_cc_metadata.json'), 'w') as fp:
                json.dump([ncc_version, cmd], fp)
        
        env = os.environ.copy()
        ld_preload = env.get('LD_PRELOAD', '')
        if 'libtcmalloc' in ld_preload:
            updated_ld_preload = ':'.join(
                path for path in ld_preload.split(':') if 'libtcmalloc' not in path
            )
            env['LD_PRELOAD'] = updated_ld_preload

        # FIXED: use check_call instead of run to avoid hang
        subprocess.check_call(cmd, cwd=tmpdir, env=env)

        with open(neff_path, 'rb') as fp:
            neff_bytes = fp.read()
        compiled_hlo_bytes = None
        compiled_hlo_path = os.path.join(tmpdir, 'wrapped_neff.hlo')
        if os.path.isfile(compiled_hlo_path):
            with open(compiled_hlo_path, 'rb') as fp:
                compiled_hlo_bytes = fp.read()
    return neff_bytes, compiled_hlo_bytes

# Create dummy libneuronxla module
mod = types.ModuleType('libneuronxla')

def dummy_configure():
    pass
mod.configure_environment = dummy_configure

def dummy_hook():
    pass
mod.hook = dummy_hook

def dummy_callback(name, addressable_device_index, execution_count):
    return 'inputs'
mod._dump_hlo_snapshot_callback = dummy_callback

def my_neuronx_cc(code, code_format, platform_version, file_prefix):
    platform_str = platform_version.decode()
    if platform_str == '2.0':
        target = 'trn1'
    elif platform_str == '2.1':
        target = 'inf2'
    elif platform_str == '1.0':
        target = 'inf1'
    else:
        target = 'trn1'
        
    return _neuronx_cc_impl_fast(code, target)

mod.neuronx_cc = my_neuronx_cc

sys.modules['libneuronxla'] = mod
print('Dummy libneuronxla module with real _neuronx_cc_impl_fast registered')
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
                    # Download and unzip libneuronxla wheel to get libneuronpjrt.so
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
                end
            end
        end
        @assert isfile(trainium_pjrt_plugin_path)
    end
end

end # module Trainium
