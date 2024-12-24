"""Loads upb."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "upb",
        sha256 = "61d0417abd60e65ed589c9deee7c124fe76a4106831f6ad39464e1525cef1454",
        strip_prefix = "upb-9effcbcb27f0a665f9f345030188c0b291e32482",
        patch_cmds = [
            "sed -i.bak0 's/@bazel_tools\\/\\/platforms:windows/@platforms\\/\\/os:windows/g' BUILD",
            "sed -i.bak0 's/-Werror//g' BUILD",
        ],
        url = "https://github.com/protocolbuffers/upb/archive/9effcbcb27f0a665f9f345030188c0b291e32482.tar.gz",
    )
