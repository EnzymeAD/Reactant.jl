"""Loads upb."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "UPB_COMMIT", "UPB_SHA256")

def repo():
    http_archive(
        name = "upb",
        sha256 = UPB_SHA256,
        strip_prefix = "upb-" + UPB_COMMIT,
        patch_cmds = [
            "sed -i.bak0 's/@bazel_tools\\/\\/platforms:windows/@platforms\\/\\/os:windows/g' BUILD",
            "sed -i.bak0 's/-Werror//g' BUILD",
        ],
        url = "https://github.com/protocolbuffers/upb/archive/{commit}.tar.gz".format(commit = UPB_COMMIT),
    )
