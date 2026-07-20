# [Local build of ReactantExtra](@id local-build)

In the `deps/` subdirectory of the Reactant repository there is a script to do local builds of ReactantExtra, including debug builds.

## Requirements

* Julia.  If you don't have it already, you can obtain it from the [official Julia website](https://julialang.org/downloads/)
* A reasonably recent C/C++ compiler, ideally GCC 12+.
  Older compilers may not work.
* Bazel. If you don't have it already, you can download a build for your platform from [the latest `bazelbuild/bazelisk` release](https://github.com/bazelbuild/bazelisk/releases/latest) and put the `bazel` executable in `PATH`.
* A fast linker like `lld` or `mold`.
  This is strictly speaking not necessary in general, but `lld` is used by default, and for debug builds with CUDA support, you'll need it.
  Binutils `ld` won't work, don't even try using it.
  You can obtain `mold` for your platform from the [latest `rui314/mold` release](https://github.com/rui314/mold/releases/latest) and put the executable in `PATH` with name `ld.mold`.

On MacOS the latter two requirements can be installed with `brew install bazelisk lld`.

## Building

At a high-level, after you `cd` to the `deps/` directory you can run the commands

```bash
julia --project -e 'using Pkg; Pkg.instantiate()' # needed only the first time to install dependencies for this script
julia -O0 --color=yes --project build_local.jl
```

There are a few options you may want to use to tweak the build.
For more information run the command

```bash
julia --project build_local.jl --help
```

which prints the following (note that the output may be out of date; you should run it locally to see the options available to you):

usage: build_local.jl [--debug] [--backend BACKEND]
                      [--gcc_host_compiler_path GCC_HOST_COMPILER_PATH]
                      [--cc CC]
                      [--hermetic_python_version HERMETIC_PYTHON_VERSION]
                      [--jobs JOBS] [--copt COPT] [--cxxopt CXXOPT]
                      [--extraopt EXTRAOPT] [--color COLOR] [--cache]
                      [--push-cache] [-h]

optional arguments:
  --debug               Build with debug mode (-c dbg).
  --backend BACKEND     Build with the specified backend (auto, cpu,
                        cuda). (default: "auto")
  --gcc_host_compiler_path GCC_HOST_COMPILER_PATH
                        Path to the gcc host compiler. (default:
                        "/usr/bin/gcc")
  --cc CC                (default: "/usr/bin/cc")
  --hermetic_python_version HERMETIC_PYTHON_VERSION
                        Hermetic Python version. (default: "3.12")
  --jobs JOBS           Number of parallel jobs. (type: Int64,
                        default: Sys.CPU_THREADS)
  --copt COPT           Options to be passed to the C compiler.  Can
                        be used multiple times.
  --cxxopt CXXOPT       Options to be passed to the C++ compiler.  Can
                        be used multiple times.
  --extraopt EXTRAOPT   Extra options to be passed to Bazel.  Can be
                        used multiple times.
  --color COLOR         Set to `yes` to enable color output, or `no`
                        to disable it. Defaults to same color setting
                        as the Julia process.
  --cache               Build with bazel cache.
  --push-cache          Build and store to bazel cache.
  -h, --help            show this help message and exit
```

## Tips and tricks

### Doing a build on a system with memory or number of processes restrictions

If you try to do the build on certain systems which have restrictions on the number of processes or memory that your user can use (for example, login nodes of clusters), you may have to limit the number of parallel jobs used by Bazel.
By default Bazel tries to use the maximum number of CPUs available on the system.
If you need to reduce that, you can pass the `--jobs JOBS` flag option.

If the Bazel server is terminated abruptly with an error which looks like

```
Server terminated abruptly (error code: 14, error message: 'Socket closed', log file: '/path/to/server/jvm.out')
```

that may be due to the fact that the build process was using too many resources (e.g. if concurrent compiler processes are cumulatively using too much memory or too many threads).
In this case, reducing the number of parallel jobs may also be beneficial.

### CUDA debug build

A CUDA debug build (`--debug --backend=cuda`) requires a recent GCC compiler (at least v12) and also a fast linker (see requirements above).
You can tell GCC to use either `lld` or `mold` with `--extraopt '--linkopt=-fuse-ld=lld'` or `--extraopt '--linkopt=-fuse-ld=mold'` respectively.
NOTE: the option `-fuse-ld=mold` was added in GCC 12, if you're trying to use an older version you can have some luck by making a symlink named `ld` pointing to `mold` in `PATH`, with higher precedence than Binutils `ld`.

### Optimised build with debug symbols

Unoptimised builds of Reactant can be _extremely_ slow at runtime.
You may have more luck by doing an optimised build that retains (i.e., doesn't strip) the debug symbols, which in Bazel can achieved with the options `--strip=never --copt -g -c opt`.
To do that using this `build_local.jl` script pass the options `--extraopt '--strip=never' --copt -g` (optimised builds are the default, unless you use `--debug`).

### Using ccache

To speed up recompilation you can use [ccache](https://ccache.dev/).
You can use ccache on top of Bazel's own caching system, which is very inflexible: Bazel invalidates cache whenever you change any variable, leading to poor cache hit rate when recompiling code with different options, even if they don't actually affect compilation output.
Since [Bazel interacts badly with ccache](https://github.com/ccache/ccache/discussions/1279) when the latter [masquerades itself as the compiler via symlinks](https://ccache.dev/manual/4.11.3.html#_run_modes), the most reliable way to use ccache with Bazel is to create a small shell wrapper like

```bash
#!/bin/bash
/path/to/ccache /path/to/your/compiler "${@}"
```

replacing `/path/to/ccache` with the path where ccache was installed, and `/path/to/your/compiler` with the path of your compiler.
Then inform Bazel to use this script as the compiler (with the `build_local.jl` script add the flag `--cc /path/to/ccache/wrapper`, where `/path/to/ccache/wrapper` is the path where you saved the shell wrapper).
Finally, you may have to add the flag `--extraopt "--sandbox_writable_path=/path/to/ccache/directory"`, otherwise Bazel will not let `ccache` write the cache output to its own directory.

## `LocalPreferences.toml` file

At the end of a successful build, the `build_local.jl` script will create a `LocalPreferences.toml` file (see [`Preferences.jl` documentation](https://juliapackaging.github.io/Preferences.jl/stable/) for more information) in the top-level of the Reactant repository, pointing `libReactantExtra` to the new local build. 
If you instantiate this environment, Reactant will automatically use the new local build.

You can check that the local build is being used by running

```julia
using Reactant_jll; Reactant_jll.libReactantExtra_path
```

which should point to the locally built library.

If you want to use the local build in a different environment:

1. copy the `LocalPreferences.toml` file (or its contents, if you already have a `LocalPreferences.toml` file) to the directory of that environment, and
2. ensure that `Reactant_jll` is a direct dependency of that environment (i.e., add `Reactant_jll` to your `Project.toml` directly).

## Troubleshooting macOS builds

### Abseil module dependency

Depending on your compiler toolchain and Bazel version you may encounter an error like:

```console
external/com_google_absl/absl/status/statusor.h:58:10: error: module com_google_absl//absl/status:statusor does not depend on a module exporting 'absl/types/span.h'
```

This is caused by Bazel performing a strict version of dependency checking between C++ modules, which can be disabled with

```console
julia [...] --extraopt "--features=-layering_check" --extraopt "--host_features=-layering_check"
```

### Toolchain for Intel macOS

One of Reactant's transitive dependencies, grpc, has a Bazel build configuration which [instructs it to build universal binaries on macOS](https://github.com/grpc/grpc/blob/8542e01ff47eb07247ff6cfbd545f3b6f4e9b5d3/bazel/grpc_build_system.bzl#L215-L225) (i.e., binaries which work on both Intel and Apple Silicon Macs).
If your toolchain is not set up to do this, you may encounter a fairly cryptic error like:

```console
ERROR: /path/to/...:72:19: in cc_toolchain_suite rule @@local_config_cc//:toolchain: cc_toolchain_suite '@@local_config_cc//:toolchain' does not contain a toolchain for cpu 'darwin_x86_64'
```

The direct solution is to disable the universal build by changing `_universal` to `_native` in that configuration.
The least intrusive way to do this is to edit `ReactantExtra/WORKSPACE` to add a new patch to the XLA dependency.
Replace the line which reads

```
NEW_XLA_PATCHES = []
```

with

```
NEW_XLA_PATCHES = [
    r"""cat >> third_party/grpc/grpc.patch << 'GRPC_PATCH'

diff --git a/bazel/grpc_build_system.bzl b/bazel/grpc_build_system.bzl
--- a/bazel/grpc_build_system.bzl
+++ b/bazel/grpc_build_system.bzl
@@ -212,7 +212,7 @@
     native.genrule(
         name = name,
         srcs = select({
-            "@platforms//os:macos": [name + "_universal"],
+            "@platforms//os:macos": [name + "_native"],
             "//conditions:default": [name + "_native"],
         }),
         outs = [name + "_binary"],
GRPC_PATCH""",
]
```

(or if `NEW_XLA_PATCHES` already contains some patches, just add the `cat >> ...` command to the list).
Don't be confused by the `-` and `+`'s here: copy the text above *verbatim* into the `WORKSPACE` file.
The text above represents a *command that modifies a patch file*, it's not itself a patch.

What's going on here?
Because grpc isn't a direct dependency, we can't easily add a patch to it in Reactant.
However, it turns out that XLA itself [contains a file](https://github.com/openxla/xla/blob/3ede8358116e1795f21e4a8833920ae5ce9f8fdc/third_party/grpc/grpc.patch) that describes the patches *it* wants to make to grpc.
We are piggybacking on this file to add *our* own patch to grpc.

Note that this is pretty fragile and can break if new versions of XLA or grpc are used.
However, the description here should hopefully be enough to help you figure out how to fix it if that happens.
