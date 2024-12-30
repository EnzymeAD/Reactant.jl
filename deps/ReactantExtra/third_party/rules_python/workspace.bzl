"""Loads rules_python (downgrades over the one used by XLA due to a bug in the latest release)."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "RULES_PYTHON_SHA256", "RULES_PYTHON_VERSION")

def repo():
    http_archive(
        name = "rules_python",
        sha256 = RULES_PYTHON_SHA256,
        strip_prefix = "rules_python-" + RULES_PYTHON_VERSION,
        url = "https://github.com/bazelbuild/rules_python/releases/download/{commit}/rules_python-{commit}.tar.gz".format(commit = RULES_PYTHON_VERSION),
    )
