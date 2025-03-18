## Local build of ReactantExtra

The script in this directory can be used to do local builds of ReactantExtra, including debug builds.

### Requirements

* Julia.  If you don't have it already, you can obtain it from the [official Julia website](https://julialang.org/downloads/)
* A reasonably recent C/C++ compiler, ideally GCC 12+.
  Older compilers may not work.
* Bazel. If you don't have it already, you can download a build for your platform from [the latest `bazelbuild/bazelisk` release](https://github.com/bazelbuild/bazelisk/releases/latest) and put the `bazel` executable in `PATH`
* not necessary in general, but for debug builds with CUDA support, you'll need a fast linker, like `lld` or `mold`
  Binutils `ld` won't work, don't even try using it.
  You can obtain `mold` for your platform from the [latest `rui314/mold` release](https://github.com/rui314/mold/releases/latest) and put the `mold` executable in `PATH`

### Building

At a high-level, you can run the commands

```
julia --project -e 'using Pkg; Pkg.instantiate()' # needed only the first time to install Julia's deps
julia -O0 --color=yes --project build_local.jl
```

There are a few of options you may want to use.
For more information run the command (what's show below may not be up to date, run the command locally to see the options available to you):

```console
% julia --project build_local.jl --help
usage: build_local.jl [--debug] [--backend BACKEND]
                      [--gcc_host_compiler_path GCC_HOST_COMPILER_PATH]
                      [--cc CC]
                      [--hermetic_python_version HERMETIC_PYTHON_VERSION]
                      [--jobs JOBS] [--copt COPT] [--cxxopt CXXOPT]
                      [--extraopt EXTRAOPT] [--color COLOR] [-h]

optional arguments:
  --debug               Build with debug mode (-c dbg).
  --backend BACKEND     Build with the specified backend (auto, cpu,
                        cuda). (default: "auto")
  --gcc_host_compiler_path GCC_HOST_COMPILER_PATH
                        Path to the gcc host compiler. (default:
                        "/usr/bin/gcc")
  --cc CC                (default: "/usr/bin/cc")
  --hermetic_python_version HERMETIC_PYTHON_VERSION
                        Hermetic Python version. (default: "3.10")
  --jobs JOBS           Number of parallel jobs. (type: Int64,
                        default: <MAXIMUM NUMBER OF CPUs>)
  --copt COPT           Options to be passed to the C compiler.  Can
                        be used multiple times.
  --cxxopt CXXOPT       Options to be passed to the C++ compiler.  Can
                        be used multiple times.
  --extraopt EXTRAOPT   Extra options to be passed to Bazel.  Can be
                        used multiple times.
  --color COLOR         Set to `yes` to enable color output, or `no`
                        to disable it. Defaults to same color setting
                        as the Julia process. (default: "no")
  -h, --help            show this help message and exit
```

### Doing a build on a system with memoryor number of processes restrictions

If you try to do the build on certain systems where there are in place restrictions on the number of processes or memory that your user can use (for example login node of clusters), you may have to limit the number of parallel jobs used by Bazel.
By default Bazel would try to use the maximum number of CPUs available on the system, if you need reduce that pass the `--jobs JOBS` flag option.
The Bazel server may be terminated abruptly if using too much memory (e.g. if concurrent compiler processes are cumulatively using too much memory), also in this case reducing the number of parallel jobs may be beneficial.

### CUDA debug build

A CUDA debug build (`--debug --backend=cuda`) requires a recent GCC compiler and also a fast linker (see requirements above).
You can tell GCC to use either `lld` or `mold` with `--extraopt '--linkopt=-fuse-ld=Lld'` or `--extraopt '--linkopt=-fuse-ld=mold'` respectively.
NOTE: the option `-fuse-ld=mold` was added in GCC 12, if you're trying to use an older version you can have some luck by making a symlink `ld` pointing to `mold` in `PATH`, with higher precendce than Binutils `ld`.

### Using ccache

If you want to use `ccache` as your compiler, you may have to add the flag `--extraopt "--sandbox_writable_path=/path/to/ccache/directory"` to let `ccache` write to its own directory.
