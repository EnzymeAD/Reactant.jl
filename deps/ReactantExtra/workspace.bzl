load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def enzyme_jax_repository(local_path, commit, sha256):
    if local_path:
        # NOTE: local_repository does not support patch_cmds. Apply the patch_cmds
        # from the http_archive branch below once to your local checkout before using this.
        native.local_repository(
            name = "enzyme_ad",
            path = local_path,
        )
    else:
        http_archive(
            name = "enzyme_ad",
            patch_cmds = [
                """
sed -i.bak0 "s/\\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/:patches/@enzyme_ad\\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/:patches/g" workspace.bzl
sed -i.bak0 "s,//:patches,@enzyme_ad//:patches,g" third_party/*/workspace.bzl
""",
            ],
            sha256 = sha256,
            strip_prefix = "Enzyme-JAX-" + commit,
            urls = ["https://github.com/EnzymeAD/Enzyme-JAX/archive/{commit}.tar.gz".format(commit = commit)],
        )
