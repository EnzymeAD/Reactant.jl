load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")

# name = "ygg_aarch64_toolchain_config"
# cpu = "aarch64"
# toolchain_identifier = "ygg_aarch64"
# target_system_name = "aarch64-unknown-linux-gnu"
def _ygg_cc_toolchain_impl(ctx):
    bb_target = ctx.configuration.default_shell_env["bb_target"]
    bb_full_target = ctx.configuration.default_shell_env["bb_full_target"]
    cpu = ctx.configuration.default_shell_env["bb_cpu"]
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
        compiler = "clang",
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
            "ar": f"/opt/bin/{bb_full_target}/ar",
            "as": f"/opt/bin/{bb_full_target}/as",
            "c++": f"/opt/bin/{bb_full_target}/c++",
            "c++filt": f"/opt/bin/{bb_full_target}/c++filt",
            "cc": f"/opt/bin/{bb_full_target}/cc",
            "clang": f"/opt/bin/{bb_full_target}/clang",
            "clang++": f"/opt/bin/{bb_full_target}/clang++",
            "cpp": f"/opt/bin/{bb_full_target}/cpp",
            "f77": f"/opt/bin/{bb_full_target}/f77",
            # WARN we force to use clang instead of gcc
            "g++": f"/opt/bin/{bb_full_target}/clang++",
            "gcc": f"/opt/bin/{bb_full_target}/clang",
            "gfortran": f"/opt/bin/{bb_full_target}/gfortran",
            "ld": f"/opt/bin/{bb_full_target}/ld",
            "ld.lld": f"/opt/bin/{bb_full_target}/ld.lld",
            "libtool": f"/opt/bin/{bb_full_target}/libtool",
            "lld": f"/opt/bin/{bb_full_target}/lld",
            "nm": f"/opt/bin/{bb_full_target}/nm",
            "objcopy": f"/opt/bin/{bb_full_target}/objcopy",
            "patchelf": f"/opt/bin/{bb_full_target}/patchelf",
            "ranlib": f"/opt/bin/{bb_full_target}/ranlib",
            "readelf": f"/opt/bin/{bb_full_target}/readelf",
            "strip": f"/opt/bin/{bb_full_target}/strip",
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
            f"-isystem /opt/{bb_target}/lib/gcc/{bb_target}/10.2.0/include",
            f"-isystem /opt/{bb_target}/lib/gcc/{bb_target}/10.2.0/include-fixed",
            f"-isystem /opt/{bb_target}/{bb_target}/include",
            f"-isystem /opt/{bb_target}/{bb_target}/sys-root/usr/include",
            f"-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0",
            f"-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0/{bb_target}",
            f"-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0/backward",
            f"-isystem /opt/{bb_target}/{bb_target}/include/c++/10.2.0/parallel",
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
        builtin_sysroot = f"/opt/{bb_target}/{bb_target}/sys-root/",
        coverage_compile_flags = ["--coverage"],
        coverage_link_flags = ["--coverage"],
        host_system_name = "linux",
        # TODO gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
        # supports_start_end_lib = supports_start_end_lib,
    )

ygg_cc_toolchain = rule(
    implementation = _ygg_cc_toolchain_impl,
    attrs = {},
    executable = False,
    test = False,
)
