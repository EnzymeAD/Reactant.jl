"""Declare the beast toolchain."""

load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")

def beast_cc_toolchain():
    cc_toolchain_suite(
        name = "beast_toolchain_suite",
        toolchains = {
            "k8": ":beast_toolchain",
        },
    )

    cc_toolchain(
        name = "beast_x86_toolchain",
        all_files = ":empty",
        compiler_files = ":empty",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 1,
        toolchain_config = ":beast_x86_toolchain_config",
        toolchain_identifier = "beast_x86_toolchain",
    )

    cc_toolchain_config(
        name = "beast_x86_toolchain_config",
        abi_libc_version = "local",
        abi_version = "local",
        compile_flags = [
            "-I/usr/include/c++/11",
        ],
        compiler = "clang",
        coverage_compile_flags = ["--coverage"],
        coverage_link_flags = ["--coverage"],
        cpu = "k8",
        cxx_builtin_include_directories = [
            "/home/wmoses/llvms/llvm16/install/lib/clang/include",
            "/home/wmoses/llvms/llvm16/install/lib/clang/16/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include",
            "/usr/include/c++/11",
            "/usr/include/x86_64-linux-gnu/c++/11",
        ],
        dbg_compile_flags = ["-g",
            "-I/usr/include/c++/11",
        ],
        host_system_name = "linux",
        link_flags = [
            "-fuse-ld=lld",
            "--ld-path=/home/wmoses/llvms/llvm16/install/bin/ld.lld",
            "-stdlib=libstdc++",
            "-static-libstdc++"
        ],
        link_libs = [
            "-lstdc++",
            "-lm",
        ],
        opt_compile_flags = [
            "-g0",
            "-O2",
            "-D_FORTIFY_SOURCE=1",
            "-DNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-stdlib=libstdc++",
            "-I/usr/include/c++/11",
        ],
        opt_link_flags = ["-Wl,--gc-sections"],
        supports_start_end_lib = True,
        target_libc = "",
        target_system_name = "x86_64-unknown-linux-gnu",
        tool_paths = {
            "/usr/bin/gcc": "/home/wmoses/llvms/llvm16/install/bin/clang",
            # "/usr/bin/gcc": "/home/wmoses/git/Reactant.jl/deps/clang",
            # "gcc": "/home/wmoses/git/Reactant.jl/deps/clang",
            "gcc": "/home/wmoses/llvms/llvm16/install/bin/clang",
            "g++": "/home/wmoses/llvms/llvm16/install/bin/clang++",
            "cpp": "/home/wmoses/llvms/llvm16/install/bin/clang++",
            "ld": "/home/wmoses/llvms/llvm16/install/bin/ld.lld",
        },
        toolchain_identifier = "beast_x86_toolchain",
        unfiltered_compile_flags = [
            "-no-canonical-prefixes",
            "-Wno-builtin-macro-redefined",
            "-D__DATE__=\"redacted\"",
            "-D__TIMESTAMP__=\"redacted\"",
            "-D__TIME__=\"redacted\"",
            "-Wno-unused-command-line-argument",
            "-Wno-gnu-offsetof-extensions",
        ],
    )
