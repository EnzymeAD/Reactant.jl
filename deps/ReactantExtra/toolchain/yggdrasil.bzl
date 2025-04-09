load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")

# based on https://github.com/bazelbuild/examples/blob/main/bzlmod/04-local_config_and_register_toolchains/local_config_sh.bzl
# TODO move this configuration to BinaryBuilderBase
ygg_config_repository_rule = repository_rule(
    environ = ["bb_target", "bb_full_target", "bb_cpu"],
    local = True,
    implementation = _ygg_cc_toolchain_impl,
)

def ygg_configure(name = "my_ygg_config"):
    """Detect the Yggdrasil toolchain and register its toolchain."""
    ygg_config_repository_rule(name)

def ygg_cc_toolchain():
    bb_target = "aarch64-linux-gnu"
    bb_full_target = "aarch64-linux-gnu-libgfortran5-cxx11-gpu+none-mode+opt"
    cpu = "aarch64"
    toolchain_identifier = "ygg_toolchain"
    target_system_name = ""
    supports_start_end_lib = False

    cc_toolchain(
        name = "ygg_target_toolchain",
        all_files = ":empty",
        compiler_files = ":empty",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 1,
        toolchain_config = ":ygg_target_toolchain_config",
        toolchain_identifier = "ygg_toolchain",
    )

    # TODO distinguish between clang and gcc toolchains?
    cc_toolchain_config(
        name = "ygg_target_toolchain_config",
        cpu = cpu,
        compiler = "compiler",
        toolchain_identifier = toolchain_identifier,
        target_system_name = target_system_name,
        target_libc = "",
        abi_libc_version = "local",
        abi_version = "local",
        cxx_builtin_include_directories = [
            "/opt/{bb_target}/lib/gcc/{bb_target}/10.2.0/include".format(bb_target = bb_target),
            "/opt/{bb_target}/lib/gcc/{bb_target}/10.2.0/include-fixed".format(bb_target = bb_target),
            "/opt/{bb_target}/{bb_target}/include".format(bb_target = bb_target),
            "/opt/{bb_target}/{bb_target}/sys-root/usr/include".format(bb_target = bb_target),
            "/opt/{bb_target}/{bb_target}/include/c++/10.2.0".format(bb_target = bb_target),
            "/opt/{bb_target}/{bb_target}/include/c++/10.2.0/{bb_target}".format(bb_target = bb_target),
            "/opt/{bb_target}/{bb_target}/include/c++/10.2.0/backward".format(bb_target = bb_target),
            "/opt/{bb_target}/{bb_target}/include/c++/10.2.0/parallel".format(bb_target = bb_target),
        ],
        tool_paths = {
            "ar": "/opt/bin/{bb_full_target}/ar".format(bb_full_target = bb_full_target),
            "as": "/opt/bin/{bb_full_target}/as".format(bb_full_target = bb_full_target),
            "c++": "/opt/bin/{bb_full_target}/c++".format(bb_full_target = bb_full_target),
            "c++filt": "/opt/bin/{bb_full_target}/c++filt".format(bb_full_target = bb_full_target),
            "cc": "/opt/bin/{bb_full_target}/cc".format(bb_full_target = bb_full_target),
            "clang": "/opt/bin/{bb_full_target}/clang".format(bb_full_target = bb_full_target),
            "clang++": "/opt/bin/{bb_full_target}/clang++".format(bb_full_target = bb_full_target),
            "cpp": "/opt/bin/{bb_full_target}/cpp".format(bb_full_target = bb_full_target),
            "f77": "/opt/bin/{bb_full_target}/f77".format(bb_full_target = bb_full_target),
            # WARN we force to use clang instead of gcc
            "g++": "/opt/bin/{bb_full_target}/clang++".format(bb_full_target = bb_full_target),
            "gcc": "/opt/bin/{bb_full_target}/clang".format(bb_full_target = bb_full_target),
            "gfortran": "/opt/bin/{bb_full_target}/gfortran".format(bb_full_target = bb_full_target),
            "ld": "/opt/bin/{bb_full_target}/ld".format(bb_full_target = bb_full_target),
            "ld.lld": "/opt/bin/{bb_full_target}/ld.lld".format(bb_full_target = bb_full_target),
            "libtool": "/opt/bin/{bb_full_target}/libtool".format(bb_full_target = bb_full_target),
            "lld": "/opt/bin/{bb_full_target}/lld".format(bb_full_target = bb_full_target),
            "nm": "/opt/bin/{bb_full_target}/nm".format(bb_full_target = bb_full_target),
            "objcopy": "/opt/bin/{bb_full_target}/objcopy".format(bb_full_target = bb_full_target),
            "patchelf": "/opt/bin/{bb_full_target}/patchelf".format(bb_full_target = bb_full_target),
            "ranlib": "/opt/bin/{bb_full_target}/ranlib".format(bb_full_target = bb_full_target),
            "readelf": "/opt/bin/{bb_full_target}/readelf".format(bb_full_target = bb_full_target),
            "strip": "/opt/bin/{bb_full_target}/strip".format(bb_full_target = bb_full_target),
            # from host
            "llvm-cov": "/opt/x86_64-linux-musl/bin/llvm-cov",
            "llvm-profdata": "/opt/x86_64-linux-musl/bin/llvm-profdata",
            "objdump": "/usr/bin/objdump",
        },
        compile_flags = [
            "-fstack-protector",
            "-Wall",
            "-Wunused-but-set-parameter",
            "-Wno-free-nonheap-object",
            "-fno-omit-frame-pointer",
            # TODO cxx_builtin_include_directories doesn't seem to be working, so we add the INCLUDE_PATHs manually
            "-isystem /opt/{bb_target}/lib/gcc/{bb_target}/10.2.0/include".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/lib/gcc/{bb_target}/10.2.0/include-fixed".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/{bb_target}/include".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/{bb_target}/sys-root/usr/include".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0/{bb_target}".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0/backward".format(bb_target = bb_target),
            "-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0/parallel".format(bb_target = bb_target),
        ],
        opt_compile_flags = [
            "-g0",
            "-O2",
            "-D_FORTIFY_SOURCE=1",
            "-DNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            # "-stdlib=libstdc++",
        ],
        dbg_compile_flags = ["-g"],
        link_flags = [],
        link_libs = [
            "-lstdc++",
            "-lm",
        ],
        opt_link_flags = ["-Wl,--gc-sections"],
        unfiltered_compile_flags = [
            "-no-canonical-prefixes",
            "-Wno-builtin-macro-redefined",
            "-D__DATE__=\"redacted\"",
            "-D__TIMESTAMP__=\"redacted\"",
            "-D__TIME__=\"redacted\"",
            "-Wno-unused-command-line-argument",
            "-Wno-gnu-offsetof-extensions",
        ],
        builtin_sysroot = "/opt/{bb_target}/{bb_target}/sys-root/".format(bb_target = bb_target),
        coverage_compile_flags = ["--coverage"],
        coverage_link_flags = ["--coverage"],
        host_system_name = "linux",
        # TODO gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
        # supports_start_end_lib = supports_start_end_lib,
    )
