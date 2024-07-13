# Invoke with
# `julia --project=deps deps/build_local.jl [path-to-enzyme]`

# the pre-built ReactantExtra_jll might not be loadable on this platform
Reactant_jll = Base.UUID("0192cb87-2b54-54ad-80e0-3be72ad8a3c0")

using Pkg, Scratch, Preferences, Libdl

# 1. Get a scratch directory
scratch_dir = get_scratch!(Reactant_jll, "build")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)

source_dir = joinpath(@__DIR__, "ReactantExtra")

# 2. Ensure that an appropriate LLVM_full_jll is installed
Pkg.activate(; temp=true)

# Build!
@info "Building" source_dir scratch_dir
run(`mkdir -p $(scratch_dir)`)
run(Cmd(`$(Base.julia_cmd().exec[1]) --project=. -e "using Pkg; Pkg.instantiate()"`, dir=source_dir))
# --action_env TF_CUDA_COMPUTE_CAPABILITIES="sm_50,sm_60,sm_70,sm_80,compute_90"
run(Cmd(`bazel build -c dbg --action_env=JULIA=$(Base.julia_cmd().exec[1])
--repo_env HERMETIC_PYTHON_VERSION="3.10"
--check_visibility=false --verbose_failures :libReactantExtra.so :Builtin.inc.jl :Arith.inc.jl :Affine.inc.jl :Func.inc.jl :Enzyme.inc.jl :StableHLO.inc.jl :CHLO.inc.jl :VHLO.inc.jl`, dir=source_dir,
env=Dict("HOME"=>ENV["HOME"], "PATH"=>joinpath(source_dir, "..")*":"*ENV["PATH"])))

run(Cmd(`rm -f libReactantExtra.dylib`, dir=joinpath(source_dir, "bazel-bin")))
run(Cmd(`ln -s libReactantExtra.so libReactantExtra.dylib`, dir=joinpath(source_dir, "bazel-bin")))

# Discover built libraries
built_libs = filter(readdir(joinpath(source_dir, "bazel-bin"))) do file
    endswith(file, "Extra.$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(source_dir, "bazel-bin", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

# Tell ReactReactantExtra_jllant_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Reactant_jll",
    "libReactantExtra_path" => lib_path,
    "libReactantDialects_path" => joinpath(source_dir, "bazel-bin"),
    force=true,
)
