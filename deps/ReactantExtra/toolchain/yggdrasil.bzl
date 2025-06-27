load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")

# name = "ygg_aarch64_toolchain_config"
# cpu = "aarch64"
# toolchain_identifier = "ygg_aarch64"
# target_system_name = "aarch64-unknown-linux-gnu"
def ygg_cc_toolchain(cpu, toolchain_identifier, target_system_name, bb_target, bb_full_target, supports_start_end_lib = False):
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
            "/opt/" + bb_target + "/lib/gcc/" + bb_target + "/10.2.0/include",
            "/opt/" + bb_target + "/lib/gcc/" + bb_target + "/10.2.0/include-fixed",
            "/opt/" + bb_target + "/" + bb_target + "/include",
            "/opt/" + bb_target + "/" + bb_target + "/sys-root/usr/include",
            "/opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0",
            "/opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0/" + bb_target,
            "/opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0/backward",
            "/opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0/parallel",
        ],
        tool_paths = {
            "ar": "/opt/bin/" + bb_full_target + "/ar",
            "as": "/opt/bin/" + bb_full_target + "/as",
            "c++": "/opt/bin/" + bb_full_target + "/c++",
            "c++filt": "/opt/bin/" + bb_full_target + "/c++filt",
            "cc": "/opt/bin/" + bb_full_target + "/cc",
            "clang": "/opt/bin/" + bb_full_target + "/clang",
            "clang++": "/opt/bin/" + bb_full_target + "/clang++",
            "cpp": "/opt/bin/" + bb_full_target + "/cpp",
            "f77": "/opt/bin/" + bb_full_target + "/f77",
            "g++": "/opt/bin/" + bb_full_target + "/g++",
            "gcc": "/opt/bin/" + bb_full_target + "/gcc",
            "gfortran": "/opt/bin/" + bb_full_target + "/gfortran",
            "ld": "/opt/bin/" + bb_full_target + "/ld",
            "ld.lld": "/opt/bin/" + bb_full_target + "/ld.lld",
            "libtool": "/opt/bin/" + bb_full_target + "/libtool",
            "lld": "/opt/bin/" + bb_full_target + "/lld",
            "nm": "/opt/bin/" + bb_full_target + "/nm",
            "objcopy": "/opt/bin/" + bb_full_target + "/objcopy",
            "patchelf": "/opt/bin/" + bb_full_target + "/patchelf",
            "ranlib": "/opt/bin/" + bb_full_target + "/ranlib",
            "readelf": "/opt/bin/" + bb_full_target + "/readelf",
            "strip": "/opt/bin/" + bb_full_target + "/strip",
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
            "-isystem /opt/" + bb_target + "/lib/gcc/" + bb_target + "/10.2.0/include",
            "-isystem /opt/" + bb_target + "/lib/gcc/" + bb_target + "/10.2.0/include-fixed",
            "-isystem /opt/" + bb_target + "/" + bb_target + "/include",
            "-isystem /opt/" + bb_target + "/" + bb_target + "/sys-root/usr/include",
            "-isystem /opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0",
            "-isystem /opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0/" + bb_target,
            "-isystem /opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0/backward",
            "-isystem /opt/" + bb_target + "/" + bb_target + "/include/c++/10.2.0/parallel",
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
        builtin_sysroot = "/opt/" + bb_target + "/" + bb_target + "/sys-root/",
        coverage_compile_flags = ["--coverage"],
        coverage_link_flags = ["--coverage"],
        host_system_name = "linux",
        # TODO gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
        supports_start_end_lib = supports_start_end_lib,
    )
