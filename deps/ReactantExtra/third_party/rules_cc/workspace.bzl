"""Loads bazel rules_cc."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "rules_cc",
        sha256 = "85723d827f080c5e927334f1fb18a294c0b3f94fee6d6b45945f5cdae6ea0fd4",
        strip_prefix = "rules_cc-c8c38f8c710cbbf834283e4777916b68261b359c",
        urls = [
            "https://github.com/bazelbuild/rules_cc/archive/c8c38f8c710cbbf834283e4777916b68261b359c.tar.gz",
        ],
    )
