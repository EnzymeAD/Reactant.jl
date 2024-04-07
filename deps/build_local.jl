# Invoke with
# `julia --project=deps deps/build_local.jl [path-to-enzyme]`

# the pre-built ReactantExtra_jll might not be loadable on this platform
ReactantExtra_jll = Base.UUID("7cc45869-7501-5eee-bdea-0790c847d4ef")

using Pkg, Scratch, Preferences, Libdl

# 1. Get a scratch directory
scratch_dir = get_scratch!(ReactantExtra_jll, "build")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)

source_dir = joinpath(@__DIR__, "ReactantExtra")

# 2. Ensure that an appropriate LLVM_full_jll is installed
Pkg.activate(; temp=true)

# Build!
@info "Building" source_dir scratch_dir
run(`mkdir -p $(scratch_dir)`)
run(Cmd(`bazel build --check_visibility=false ...`, dir=source_dir))

# Discover built libraries
built_libs = filter(readdir(joinpath(source_dir, "bazel-bin"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(source_dir, "bazel-bin", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

# Tell ReactReactantExtra_jllant_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "ReactantExtra_jll",
    "libReactantExtra_path" => lib_path,
    "libReactantDialects_path" => joinpath(source_dir, "bazel-bin"),
    force=true,
)
