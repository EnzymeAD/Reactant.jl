"""Loads nsync."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "NSYNC_COMMIT", "NSYNC_SHA256")

def repo():
    http_archive(
        name = "nsync",
        sha256 = NSYNC_SHA256,
        strip_prefix = "nsync-" + NSYNC_COMMIT,
        urls = ["https://github.com/wsmoses/nsync/archive/{commit}.tar.gz".format(commit = NSYNC_COMMIT)],
    )
