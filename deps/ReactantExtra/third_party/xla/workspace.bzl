"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@enzyme_ad//:workspace.bzl", "XLA_PATCHES")
load("@jax//third_party/xla:workspace.bzl", "XLA_COMMIT", "XLA_SHA256")

def repo():
    http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-" + XLA_COMMIT,
        urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
        patch_cmds = XLA_PATCHES + [
            """sed -i.bak0 "s/__cpp_lib_hardware_interference_size/HW_INTERFERENCE_SIZE/g" xla/backends/cpu/runtime/thunk_executor.h""",
            """sed -i.bak0 "s/__cpp_lib_hardware_interference_size/HW_INTERFERENCE_SIZE/g" xla/stream_executor/host/host_kernel.cc""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_LINK_H=1\\/HAVE_LINK_H=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/LLVM_ENABLE_THREADS=1\\/LLVM_ENABLE_THREADS=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_MALLINFO=1\\/DONT_HAVE_ANY_MALLINFO=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP=1\\/FAKE_HAVE_PTHREAD_GETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP=1\\/FAKE_HAVE_PTHREAD_SETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/ENABLE_CRASH_OVERRIDES 1\\/ENABLE_CRASH_OVERRIDES 0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP\\/FAKE_HAVE_PTHREAD_GETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP\\/FAKE_HAVE_PTHREAD_SETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl""",
            # """sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\['find . -type f -name BUILD.bazel -exec sed -i.bak0 \\\\\\'s\\/\\\"CAPIIR\\\",\\/\\\"CAPIIR\\\",alwayslink=1,\\/g\\\\\\\\' {} +',/g" third_party/llvm/workspace.bzl""",
        ],
    )
