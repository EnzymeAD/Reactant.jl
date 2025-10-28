# Invoke with
# `julia --project=deps deps/build_local.jl [--debug] [--backend=auto/cpu/cuda]`

# the pre-built ReactantExtra_jll might not be loadable on this platform
Reactant_jll = Base.UUID("0192cb87-2b54-54ad-80e0-3be72ad8a3c0")

using ArgParse

using Libdl

# adapted from `cudaRuntimeGetVersion` in CUDA_Runtime_jll
function cuDriverGetVersion(library_handle)
    function_handle = Libdl.dlsym(library_handle, "cuDriverGetVersion"; throw_error=false)
    if function_handle === nothing
        @debug "CUDA Driver library seems invalid (does not contain 'cuDriverGetVersion')"
        return nothing
    end
    version_ref = Ref{Cint}()
    status = ccall(function_handle, Cint, (Ptr{Cint},), version_ref)
    if status != 0
        @debug "Call to 'cuDriverGetVersion' failed with status $(status)"
        return nothing
    end
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    version = VersionNumber(major, minor, patch)
    @debug "Detected CUDA Driver version $(version)"
    return version
end

function get_cuda_version()
    cuname = if Sys.iswindows()
        Libdl.find_library("nvcuda")
    else
        Libdl.find_library(["libcuda.so.1", "libcuda.so"])
    end

    if cuname == ""
        return nothing
    end

    handle = Libdl.dlopen(cuname)
    current_cuda_version = cuDriverGetVersion(handle)
    path = Libdl.dlpath(handle)
    Libdl.dlclose(handle)
    return current_cuda_version
end

s = ArgParseSettings()
#! format: off
@add_arg_table! s begin
    "--debug"
        help = "Build with debug mode (-c dbg)."
        action = :store_true
    "--backend"
        help = "Build with the specified backend (auto, cpu, cuda)."
        default = "auto"
        arg_type = String
    "--gcc_host_compiler_path"
        help = "Path to the gcc host compiler."
        default = something(Sys.which("gcc"), "/usr/bin/gcc")
        arg_type = String
    "--cc"
        default = something(Sys.which("clang"), Sys.which("cc"), Sys.which("gcc"), "/usr/bin/cc")
        arg_type = String
    "--hermetic_python_version"
        help = "Hermetic Python version."
        default = "3.12"
        arg_type = String
    "--jobs"
        help = "Number of parallel jobs."
        default = Sys.CPU_THREADS
        arg_type = Int
    "--copt"
        help = "Options to be passed to the C compiler.  Can be used multiple times."
        action = :append_arg
        arg_type = String
    "--cxxopt"
        help = "Options to be passed to the C++ compiler.  Can be used multiple times."
        action = :append_arg
        arg_type = String
    "--extraopt"
        help = "Extra options to be passed to Bazel.  Can be used multiple times."
        action = :append_arg
        arg_type = String
    "--color"
        help = "Set to `yes` to enable color output, or `no` to disable it. Defaults to same color setting as the Julia process."
        default = something(Base.have_color, false) ? "yes" : "no"
        arg_type = String
end
#! format: on
parsed_args = parse_args(ARGS, s)

println("Parsed args:")
for (k, v) in parsed_args
    println("  $k = $v")
end
println()

source_dir = joinpath(@__DIR__, "ReactantExtra")

#--repo_env TF_NEED_ROCM=1
#--define=using_rocm=true --define=using_rocm_hipcc=true
#--action_env TF_ROCM_AMDGPU_TARGETS="gfx900,gfx906,gfx908,gfx90a,gfx1030"

# --repo_env TF_NEED_CUDA=1
# --repo_env TF_NVCC_CLANG=1
# --repo_env TF_NCCL_USE_STUB=1
# --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_50,sm_60,sm_70,sm_80,compute_90"
# --@xla//xla/python:jax_cuda_pip_rpaths=true
# --repo_env=HERMETIC_CUDA_VERSION="12.3.2"
# --repo_env=HERMETIC_CUDNN_VERSION="9.1.1"
# --@local_config_cuda//cuda:include_cuda_libs=true
# --@local_config_cuda//:enable_cuda
# --@local_config_cuda//:cuda_compiler=nvcc
# --crosstool_top="@local_config_cuda//crosstool:toolchain"

abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct CUDABackend <: AbstractBackend
    version::VersionNumber
    CUDABackend(ver::VersionNumber) = new(VersionNumber(ver.major))
end

function parse_build_backend(str::String)::AbstractBackend
    if str == "cpu"
        return CPUBackend()
    elseif str == "cuda12"
        return CUDABackend(v"12")
    elseif str == "cuda13"
        return CUDABackend(v"13")
    end

    if str in ("auto", "cuda")
        cuda_ver = get_cuda_version()
        if isnothing(cuda_ver)
            if str == "cuda"
                throw(
                    AssertionError(
                        "Could not detect cuda version, but requested cuda with auto version build",
                    ),
                )
            end
            return CPUBackend()
        else
            return CUDABackend(get_cuda_version())
        end
    else
        error("Unknown backend '$(str)'")
    end
end

build_kind = parsed_args["debug"] ? "dbg" : "opt"

build_backend = parse_build_backend(parsed_args["backend"])

arg = if build_backend == CUDABackend(v"12")
    "--config=cuda12"
elseif build_backend == CUDABackend(v"13")
    "--config=cuda13"
elseif build_backend == CPUBackend()
    ""
else
    throw(AssertionError("Unknown backend `$build_backend`"))
end

bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

@info "Building JLL with $(bazel_cmd)"

gcc_host_compiler_path = parsed_args["gcc_host_compiler_path"]
cc = parsed_args["cc"]
hermetic_python_version = parsed_args["hermetic_python_version"]

# Try to guess if `cc` is GCC and get its version number.
cc_is_gcc, gcc_version = let
    io = IOBuffer()
    run(pipeline(ignorestatus(`$(cc) --version`); stdout=io))
    version_string = String(take!(io))
    # Detecing GCC is hard, the name "gcc" may not appear anywhere in the
    # version string, but on the second line there should be FSF.
    m = match(
        r"\([^)]+\) (\d+\.\d+\.\d+).*\n.*Free Software Foundation, Inc\.",
        version_string,
    )
    if !isnothing(m)
        true, VersionNumber(m[1])
    else
        false, v"0"
    end
end

build_cmd_list = [bazel_cmd, "build"]
!isempty(arg) && push!(build_cmd_list, arg)
append!(build_cmd_list, ["-c", "$(build_kind)"])
push!(build_cmd_list, "--action_env=JULIA=$(Base.julia_cmd().exec[1])")
push!(build_cmd_list, "--repo_env=HERMETIC_PYTHON_VERSION=$(hermetic_python_version)")
if !isempty(gcc_host_compiler_path)
    push!(build_cmd_list, "--repo_env=GCC_HOST_COMPILER_PATH=$(gcc_host_compiler_path)")
end
push!(build_cmd_list, "--repo_env=CC=$(cc)")
push!(build_cmd_list, "--check_visibility=false")
push!(build_cmd_list, "--verbose_failures")
push!(build_cmd_list, "--jobs=$(parsed_args["jobs"])")
push!(build_cmd_list, "--experimental_ui_max_stdouterr_bytes=-1")
push!(build_cmd_list, "--sandbox_debug")

push!(build_cmd_list, "--linkopt=-fuse-ld=lld")

for opt in parsed_args["copt"]
    push!(build_cmd_list, "--copt=$(opt)")
end
for opt in parsed_args["cxxopt"]
    push!(build_cmd_list, "--cxxopt=$(opt)")
end
for opt in parsed_args["extraopt"]
    push!(build_cmd_list, opt)
end
# Some versions of GCC can't deal with some components of XLA, disable them if necessary.
if cc_is_gcc
    arch = Base.BinaryPlatforms.arch(Base.BinaryPlatforms.HostPlatform())
    if arch == "x86_64"
        if gcc_version < v"13"
            push!(build_cmd_list, "--define=xnn_enable_avxvnniint8=false")
        end
        if gcc_version < v"12"
            push!(build_cmd_list, "--define=xnn_enable_avx512fp16=false")
        end
        if gcc_version < v"11"
            # TODO: this is not sufficient to complete a build with GCC 10.
            push!(build_cmd_list, "--define=xnn_enable_avxvnni=false")
        end
    end
else
    # Assume the compiler is clang if not GCC. `using_clang` is an option
    # introduced by Enzyme-JAX.
    push!(build_cmd_list, "--define=using_clang=true")
    push!(build_cmd_list, "--copt=-Wno-unused-command-line-argument")
end
push!(build_cmd_list, "--copt=-Wno-private-header")
push!(build_cmd_list, "--color=$(parsed_args["color"])")
push!(build_cmd_list, ":libReactantExtra.so")

@info "About to run Bazel" build_cmd_list
run(Cmd(Cmd(build_cmd_list); dir=source_dir))

# Discover built libraries
built_libs = filter(readdir(joinpath(source_dir, "bazel-bin"))) do file
    return endswith(file, "Extra.so") && startswith(file, "lib")
end

lib_path = joinpath(source_dir, "bazel-bin", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

if build_backend isa CUDABackend
    for path in (
        joinpath("bin", "ptxas"),
        joinpath("bin", "fatbinary"),
        joinpath("nvvm", "libdevice", "libdevice.10.bc"),
    )
        full_path = joinpath(source_dir, "bazel-bin", "cuda", path)
        if !Base.Filesystem.ispath(full_path)
            source = joinpath(
                source_dir,
                "bazel-bin",
                "libReactantExtra.so.runfiles",
                # libdevice's directory was moved in CUDA 13, before was in same
                # dir as ptxas and fatbinary
                if contains(basename(path), "libdevice") && build_backend.version >= v"13"
                    "cuda_nvvm"
                else
                    "cuda_nvcc"
                end,
                path,
            )
            if !Base.Filesystem.ispath(source)
                error(
                    "File $(source) does not exist, are you sure it is where you expect it to be?",
                )
            end
            Base.Filesystem.mkpath(dirname(full_path))
            @info "Symlinking $(full_path) -> $(source)"
            Base.Filesystem.symlink(source, full_path)
        end
    end
end

# Tell ReactantExtra_jll to load our library instead of the default artifact one
using Preferences

set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Reactant_jll",
    "libReactantExtra_path" => lib_path;
    force=true,
)
