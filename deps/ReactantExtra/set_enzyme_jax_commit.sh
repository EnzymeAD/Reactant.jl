#!/usr/bin/env bash
# Point ReactantExtra at a given Enzyme-JAX commit: update ENZYMEXLA_COMMIT in
# MODULE.bazel and refresh xla_deps.MODULE.bazel (the JAX / XLA / LLVM stack,
# extracted from the marked region of Enzyme-JAX's MODULE.bazel since bzlmod
# requires every root module to declare the overrides itself) and the copy of
# the patches/ directory it applies (Bazel only accepts override patches from
# the main repository).
#
# Usage: set_enzyme_jax_commit.sh <enzyme_jax_commit> [--from-dir <local Enzyme-JAX checkout>]
#
# Without --from-dir the files are downloaded from GitHub at the given commit.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="${1:?usage: $0 <enzyme_jax_commit> [--from-dir DIR]}"
shift
FROM_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-dir)
            FROM_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

sed -i.bak "s/^ENZYMEXLA_COMMIT = \".*\"/ENZYMEXLA_COMMIT = \"${COMMIT}\"/" "${DIR}/MODULE.bazel"
rm -f "${DIR}/MODULE.bazel.bak"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
if [[ -n "${FROM_DIR}" ]]; then
    cp "${FROM_DIR}/MODULE.bazel" "${TMP}/MODULE.bazel"
    cp -r "${FROM_DIR}/patches" "${TMP}/patches"
else
    curl -fsSL --retry 3 "https://github.com/EnzymeAD/Enzyme-JAX/archive/${COMMIT}.tar.gz" |
        tar -xz -C "${TMP}" --strip-components=1 \
            "Enzyme-JAX-${COMMIT}/MODULE.bazel" \
            "Enzyme-JAX-${COMMIT}/patches"
fi

sed -n '/^# BEGIN: xla_deps.MODULE.bazel/,/^# END: xla_deps.MODULE.bazel/p' "${TMP}/MODULE.bazel" > "${TMP}/xla_deps.MODULE.bazel"
if ! grep -q '^# END: xla_deps.MODULE.bazel' "${TMP}/xla_deps.MODULE.bazel"; then
    echo "Could not find the xla_deps region in Enzyme-JAX's MODULE.bazel at ${COMMIT}" >&2
    exit 1
fi
mv "${TMP}/xla_deps.MODULE.bazel" "${DIR}/xla_deps.MODULE.bazel"
rm -rf "${DIR}/patches"
mv "${TMP}/patches" "${DIR}/patches"

echo "ReactantExtra now uses Enzyme-JAX ${COMMIT}"
