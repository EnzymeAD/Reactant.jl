"""Loads bazel rules_cc."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "RULES_CC_COMMIT", "RULES_CC_SHA256")

def repo():
    http_archive(
        name = "rules_cc",
        sha256 = RULES_CC_SHA256,
        strip_prefix = "rules_cc-" + RULES_CC_COMMIT,
        urls = [
            "https://github.com/bazelbuild/rules_cc/archive/{commit}.tar.gz".format(commit = RULES_CC_COMMIT),
        ],
    )
