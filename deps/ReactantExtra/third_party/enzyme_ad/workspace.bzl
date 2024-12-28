"""Loads Enzyme-JAX."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ENZYMEXLA_COMMIT = "52e14543628df5e3f806dce096243c4603d3e103"
ENZYMEXLA_SHA256 = ""

def repo():
    http_archive(
        name = "enzyme_ad",
        sha256 = ENZYMEXLA_SHA256,
        strip_prefix = "Enzyme-JAX-" + ENZYMEXLA_COMMIT,
        urls = ["https://github.com/EnzymeAD/Enzyme-JAX/archive/{commit}.tar.gz".format(commit = ENZYMEXLA_COMMIT)],
    )
