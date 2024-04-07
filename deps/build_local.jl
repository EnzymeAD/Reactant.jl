# Invoke with
# `julia --project=deps deps/build_local.jl [path-to-enzyme]`

# the pre-built Enzyme_jll might not be loadable on this platform
Enzyme_jll = Base.UUID("7cc45869-7501-5eee-bdea-0790c847d4ef")

using Pkg, Scratch, Preferences, Libdl

BUILD_TYPE = "RelWithDebInfo" 

# 1. Get a scratch directory
scratch_dir = get_scratch!(Enzyme_jll, "build")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)

source_dir = nothing
branch = nothing
if length(ARGS) == 2 
    @assert ARGS[1] == "--branch"
    branch = ARGS[2]
    source_dir = nothing
elseif length(ARGS) == 1
    source_dir = ARGS[1]
end

if branch === nothing
    branch = "main"
end

source_dir = (@__DIR__)*"/ReactionExtra"

if source_dir === nothing
    scratch_src_dir = get_scratch!(Enzyme_jll, "src")
    cd(scratch_src_dir) do
        if !isdir("Enzyme")
            run(`git clone https://github.com/wsmoses/Enzyme`)
        end
        run(`git -C Enzyme fetch`)
        run(`git -C Enzyme checkout origin/$(branch)`)
    end
    source_dir = joinpath(scratch_src_dir, "Enzyme", "enzyme")
end

# 2. Ensure that an appropriate LLVM_full_jll is installed
Pkg.activate(; temp=true)

# Build!
@info "Building" source_dir scratch_dir
run(`mkdir -p $(scratch_dir)`)
run(Cmd(`bazel build --check_visibility=false --experimental_repo_remote_exec -s ...`, dir=source_dir))

# Discover built libraries
built_libs = filter(readdir(joinpath(scratch_dir))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(scratch_dir, "Enzyme", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

built_libs = filter(readdir(joinpath(scratch_dir, "BCLoad"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

libBC_path = joinpath(scratch_dir, "BCLoad", only(built_libs))
isfile(libBC_path) || error("Could not find library $libBC_path in build directory")

# Tell Enzyme_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Enzyme_jll",
    "libEnzyme_path" => lib_path,
    "libEnzymeBCLoad_path" => libBC_path;
    force=true,
)
