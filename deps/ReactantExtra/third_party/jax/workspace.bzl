"""Loads Enzyme-JAX."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@enzyme_ad//:workspace.bzl", "JAX_COMMIT", "JAX_SHA256")

def repo():
    http_archive(
        name = "jax",
        sha256 = JAX_SHA256,
        strip_prefix = "jax-" + JAX_COMMIT,
        urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
        patch_args = ["-p1"],
        patches = ["@enzyme_ad//:patches/jax.patch"],
    )
