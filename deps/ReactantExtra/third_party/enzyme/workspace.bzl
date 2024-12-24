"""Loads Enzyme."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@enzyme_ad//:workspace.bzl", "ENZYME_COMMIT", "ENZYME_SHA256")

def repo():
    http_archive(
        name = "enzyme",
        sha256 = ENZYME_SHA256,
        strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
        urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
    )
