#!/usr/bin/env bash
# Unpack a prebuilt libReactant archive and write Reactant_jll overrides into
# one or more LocalPreferences.toml files.
#
# Usage:
#   install_libreactant.sh <archive_tar_zst> <install_dir> <LocalPreferences.toml> [...]

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <archive_tar_zst> <install_dir> <LocalPreferences.toml> [...]" >&2
  exit 1
fi

ARCHIVE_PATH="$1"
INSTALL_DIR="$2"
shift 2

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "ERROR: Archive not found at ${ARCHIVE_PATH}" >&2
  exit 1
fi

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
tar --zstd -xf "${ARCHIVE_PATH}" -C "${INSTALL_DIR}"

julia --startup-file=no --color=yes -e '
using TOML

bundle_dir = abspath(ARGS[1])
dlext = Sys.isapple() ? "dylib" : (Sys.iswindows() ? "dll" : "so")
lib_path = joinpath(bundle_dir, "libReactantExtra." * dlext)
isfile(lib_path) || error("Missing library at $(lib_path)")

jll = Dict{String,Any}(
    "libReactantExtra_path" => lib_path,
)

ptxas_path = joinpath(bundle_dir, "cuda", "bin", "ptxas")
if isfile(ptxas_path)
    jll["gpu"] = "cuda"
    jll["gpu_version"] = "13.1"
    jll["ptxas_path"] = ptxas_path
    jll["fatbinary_path"] = joinpath(bundle_dir, "cuda", "bin", "fatbinary")
    jll["libdevice_path"] = joinpath(bundle_dir, "cuda", "nvvm", "libdevice", "libdevice.10.bc")
    nccl_path = joinpath(bundle_dir, "libnccl.so.2")
    if isfile(nccl_path)
        jll["libnccl_path"] = nccl_path
    end
elseif Sys.islinux()
    jll["gpu"] = "none"
    jll["gpu_version"] = "none"
end

for prefs_arg in ARGS[2:end]
    prefs_path = abspath(prefs_arg)
    mkpath(dirname(prefs_path))
    prefs = isfile(prefs_path) ? TOML.parsefile(prefs_path) : Dict{String,Any}()
    prefs["Reactant_jll"] = jll
    open(prefs_path, "w") do io
        TOML.print(io, prefs)
    end
    println("Wrote Reactant_jll preferences to $(prefs_path)")
end
' "${INSTALL_DIR}" "$@"
