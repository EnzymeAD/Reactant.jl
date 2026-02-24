# Launch from deps/ReactantExtra

# 1. Install JuliaFormatter
julia --color=yes -e 'import Pkg; Pkg.add(; name="JuliaFormatter", version="1")'

# 2. Set up a temporary depot path
export JULIA_DEPOT_PATH="$HOME/.julia"

# 3. Instantiate dependencies
julia --project=. --color=yes -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using Clang; Clang.JLLEnvs.get_system_includes()'

# 4. Generate MLIR Bindings
julia --project=. --color=yes make-bindings.jl

# 5. Make files writable
chmod -R u+rw ../../src/mlir/Dialects/
chmod u+rw ../../src/mlir/libMLIR_h.jl

# 6. Format the generated code
julia --color=yes -e '
  using JuliaFormatter
  format("../../src/mlir/Dialects/")
  format("../../src/mlir/libMLIR_h.jl")
  # Format twice to work around formatter issue
  format("../../src/mlir/Dialects/")
  format("../../src/mlir/libMLIR_h.jl")
'
