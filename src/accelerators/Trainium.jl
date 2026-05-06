module Trainium

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads: Downloads
using p7zip_jll: p7zip
using FileWatching: mkpidlock

using Libdl: Libdl

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
import socket
import argparse
import json
import shlex
import tempfile

plugin_dir = '$(escape_string(plugin_dir))'
target_dir = '$(escape_string(python_packages_dir))'

sys.path.append(plugin_dir)
sys.path.append(target_dir)



# Add bin directory to system PATH in Python environment
bin_dir = os.path.join(target_dir, 'bin')
os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
os.environ['PYTHONPATH'] = target_dir + os.pathsep + os.environ.get('PYTHONPATH', '')

# Ensure neuronxcc is installed using pip.pyz
try:
    import neuronxcc
    print('neuronxcc already available')
except ImportError:
    print('neuronxcc not found, installing using pip.pyz...')
    pip_pyz_path = os.path.join(plugin_dir, 'pip.pyz')
    if not os.path.exists(pip_pyz_path):
        raise FileNotFoundError(f"pip.pyz not found at {pip_pyz_path}")
        
    old_argv = sys.argv
    # Install libneuronxla and neuronx-cc from AWS Neuron repo
    sys.argv = ['pip', 'install', '--target', target_dir, '--extra-index-url', 'https://pip.repos.neuron.amazonaws.com/', 'libneuronxla', 'neuronx-cc']
    try:
        import runpy
        runpy.run_path(pip_pyz_path, run_name='__main__')
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f'pip failed with code {e.code}')
    finally:
        sys.argv = old_argv
    
    import importlib
    importlib.invalidate_caches()
    
    print('Dependencies installed successfully')

# Import hlo_pb2 after installing dependencies
try:
    from libneuronxla.proto import hlo_pb2
    print('Successfully imported hlo_pb2')
except Exception as e:
    print(f'Failed to import hlo_pb2: {e}')

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
            try:
                ver_cmd = ['neuronx-cc', '--version']
                ncc_version = subprocess.check_output(
                    ver_cmd, stderr=subprocess.STDOUT).decode()
                ncc_version, *_ = ncc_version.split('\\n')
                *_, ncc_version = ncc_version.split('version ')
                with open(os.path.join(tmpdir, 'neuronx_cc_metadata.json'), 'w') as fp:
                    json.dump([ncc_version, cmd], fp)
            except Exception as e:
                print(f"Warning: failed to get neuronx-cc version: {e}")
        
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

def _wrap_neff_as_custom_call(code, neff_bytes):
    if not neff_bytes:
        return b''
    hlo_module = hlo_pb2.HloModuleProto()
    hlo_module.ParseFromString(code)
    entry, = [cpt for cpt in hlo_module.computations if cpt.id == hlo_module.entry_computation_id]
    parameters = [None for _ in entry.program_shape.parameters]
    for inst in entry.instructions:
        if inst.opcode == 'parameter':
            parameters[inst.parameter_number] = inst
    root, = [inst for inst in entry.instructions if inst.id == entry.root_id]
    fused_root = hlo_pb2.HloInstructionProto()
    fused_root.CopyFrom(root)
    fused_root.opcode = 'custom-call'
    fused_root.operand_ids[:] = [param.id for param in parameters]
    fused_root.custom_call_target = 'AwsNeuronNeff'
    fused_root.backend_config = neff_bytes
    fused_root.frontend_attributes.map['valid_inputs'] = ','.join(['1' for _ in parameters])
    while entry.instructions:
        entry.instructions.pop()
    entry.instructions.extend(parameters)
    entry.instructions.append(fused_root)
    return hlo_module.SerializeToString()

# Create dummy libneuronxla module
mod = types.ModuleType('libneuronxla')

def dummy_configure():
    pass
mod.configure_environment = dummy_configure

_USED_PORTS = set()

def _find_free_port():
    for _ in range(256):
        with socket.socket() as sock:
            sock.bind(('', 0))
            _, port = sock.getsockname()
            if port not in _USED_PORTS:
                _USED_PORTS.add(port)
                return port
    raise OSError('No free port found!')

def my_hook():
    if 'NEURON_RT_ROOT_COMM_ID' not in os.environ:
        port = _find_free_port()
        os.environ['NEURON_RT_ROOT_COMM_ID'] = f'localhost:{port}'
        print(f"Set NEURON_RT_ROOT_COMM_ID to localhost:{port}")

mod.hook = my_hook

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
        
    neff_bytes, compiled_hlo_bytes = _neuronx_cc_impl_fast(code, target)
    if neff_bytes:
        compiled_hlo_bytes = _wrap_neff_as_custom_call(code, neff_bytes)
        return 0, compiled_hlo_bytes
    else:
        return 0, b''

mod.neuronx_cc = my_neuronx_cc

sys.modules['libneuronxla'] = mod
print('Dummy libneuronxla module with real _neuronx_cc_impl_fast registered')
"""

    ccall((:PyRun_SimpleString, PYTHON_LIB), Cint, (Cstring,), py_code)

    scratch_dir = get_trainium_pjrt_plugin_dir()
    
    # Load custom libibverbs
    libibverbs_path = joinpath(scratch_dir, "libfabric_extracted", "usr", "lib64", "libibverbs.so.1")
    @assert isfile(libibverbs_path) "libibverbs.so.1 not found in scratch space at $libibverbs_path"
    Libdl.dlopen(libibverbs_path, Libdl.RTLD_GLOBAL)
    @debug "Loaded custom libibverbs from $libibverbs_path"

    # Load custom libefa
    libefa_path = joinpath(scratch_dir, "libfabric_extracted", "usr", "lib64", "libefa.so.1")
    @assert isfile(libefa_path) "libefa.so.1 not found in scratch space at $libefa_path"
    Libdl.dlopen(libefa_path, Libdl.RTLD_GLOBAL)
    @debug "Loaded custom libefa from $libefa_path"

    # Load custom libfabric to avoid version mismatch
    libfabric_path = joinpath(scratch_dir, "libfabric_extracted", "opt", "amazon", "efa", "lib", "libfabric.so.1")
    @assert isfile(libfabric_path) "libfabric.so.1 not found in scratch space at $libfabric_path"
    Libdl.dlopen(libfabric_path, Libdl.RTLD_GLOBAL)
    @debug "Loaded custom libfabric from $libfabric_path"

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
    # Check if it is in the scratch space under python_packages (installed via pip)
    pip_path = joinpath(get_trainium_pjrt_plugin_dir(), "python_packages", "libneuronxla", trainium_pjrt_plugin_name[])
    if isfile(pip_path)
        return pip_path
    end
    # Check if it is in the scratch space as a directory (manually unzipped)
    dir_path = joinpath(get_trainium_pjrt_plugin_dir(), "libneuronxla", trainium_pjrt_plugin_name[])
    if isfile(dir_path)
        return dir_path
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
    
    # Download and extract EFA installer for libfabric
    efa_extracted_dir = joinpath(dir, "libfabric_extracted")
    if isdir(efa_extracted_dir)
        @debug "EFA libraries already extracted in '$(efa_extracted_dir)', nothing to do"
    else
        mkpidlock(joinpath(dir, "download_efa.lock")) do
            if !isdir(efa_extracted_dir)
                @debug "Downloading EFA installer..."
                efa_url = "https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz"
                efa_tarball = joinpath(dir, "aws-efa-installer-latest.tar.gz")
                Downloads.download(efa_url, efa_tarball)
                
                @debug "Extracting EFA installer..."
                # Use p7zip to extract .tar.gz to .tar
                run(`$(p7zip()) x -y $(efa_tarball) -o$(dir)`)
                
                tar_file = joinpath(dir, "aws-efa-installer-latest.tar")
                # Extract .tar using p7zip
                run(`$(p7zip()) x -y $(tar_file) -o$(dir)`)
                
                # Clean up tarballs
                rm(efa_tarball; force=true)
                rm(tar_file; force=true)
                
                # Now find and extract libfabric deb
                extracted_efa_dir = joinpath(dir, "aws-efa-installer")
                @assert isdir(extracted_efa_dir) "EFA installer failed to extract to $extracted_efa_dir"
                
                deb_path = joinpath(extracted_efa_dir, "DEBS", "UBUNTU2204", "x86_64")
                @assert isdir(deb_path) "DEB path not found at $deb_path"
                
                debs = readdir(deb_path; join=true)
                libfabric_deb = filter(f -> occursin("libfabric1-aws", f), debs)
                @assert !isempty(libfabric_deb) "libfabric deb not found in $deb_path"
                
                @debug "Extracting libfabric deb: $(libfabric_deb[1])"
                mkpath(efa_extracted_dir)
                
                # Extract deb using dpkg-deb (p7zip fails on .deb files)
                run(`dpkg-deb -x $(libfabric_deb[1]) $(efa_extracted_dir)`)
            
                # Also extract libefa from SUSE RPM
                suse_path = joinpath(extracted_efa_dir, "RPMS", "SUSE", "x86_64", "rdma-core")
                @assert isdir(suse_path) "SUSE RPM path not found at $suse_path"
                
                rpms = readdir(suse_path; join=true)
                libefa_rpm = filter(f -> occursin("libefa1", f), rpms)
                @assert !isempty(libefa_rpm) "libefa RPM not found in $suse_path"
                
                @debug "Extracting libefa rpm: $(libefa_rpm[1])"
                # Extract RPM using p7zip (produces a cpio archive)
                run(`$(p7zip()) x -y $(libefa_rpm[1]) -o$(efa_extracted_dir)`)
                
                # Extract the cpio archive using standard cpio tool
                extracted_file = "libefa1-61.0-0.x86_64"
                cd(efa_extracted_dir) do
                    run(pipeline(`cpio -idmv`, stdin=extracted_file))
                    rm(extracted_file; force=true)
                end
                
                # Also extract libibverbs from SUSE RPM
                libibverbs_rpm = filter(f -> endswith(f, ".rpm") && occursin("libibverbs1", f), rpms)
                @assert !isempty(libibverbs_rpm) "libibverbs RPM not found in $suse_path"
                
                @debug "Extracting libibverbs rpm: $(libibverbs_rpm[1])"
                run(`$(p7zip()) x -y $(libibverbs_rpm[1]) -o$(efa_extracted_dir)`)
                
                extracted_verbs_file = "libibverbs1-61.0-0.x86_64"
                cd(efa_extracted_dir) do
                    run(pipeline(`cpio -idmv`, stdin=extracted_verbs_file))
                    rm(extracted_verbs_file; force=true)
                end
                

                
                # Clean up the large installer directory
                rm(extracted_efa_dir; recursive=true, force=true)
            end
        end
    end
end

end # module Trainium
