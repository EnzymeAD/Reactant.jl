load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")

# load("//toolchain:yggdrasil.bzl", "ygg_cc_toolchain")
licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)
# =============================================================================
# Proto file collection for Julia bindings generation
# =============================================================================
#
# This section defines rules to collect proto files from XLA/TSL dependencies
# and stage them for Julia binding generation using ProtoBuf.jl.
#
# Usage:
#   bazel build //:proto_files
#   julia --project=. make-proto-bindings.jl bazel-bin/proto_files_staging
#
# The proto files are exported via exports_files() in the external repos.
# We reference them using the tf_proto_library source pattern.

# Collect all XLA proto source files into a tar archive
# The Julia script will extract this to get the proto files with proper structure
genrule(
    name = "proto_files",
    srcs = [
        # XLA core protos
        "@xla//xla:xla.proto",
        "@xla//xla:xla_data.proto",
        "@xla//xla:autotuning.proto",
        "@xla//xla:autotune_results.proto",
        "@xla//xla/service:hlo.proto",
        "@xla//xla/service:metrics.proto",
        "@xla//xla/pjrt/proto:execute_options.proto",
        "@xla//xla/pjrt/proto:compile_options.proto",
        "@xla//xla/stream_executor:device_description.proto",
        "@xla//xla/stream_executor/cuda:cuda_compute_capability.proto",
        "@xla//xla/tsl/protobuf:dnn.proto",
        # Xprof protos - main files
        "@org_xprof//plugin/xprof/protobuf:op_metrics.proto",
        "@org_xprof//plugin/xprof/protobuf:op_stats.proto",
        "@org_xprof//plugin/xprof/protobuf:op_profile.proto",
        "@org_xprof//plugin/xprof/protobuf:hlo_stats.proto",
        "@org_xprof//plugin/xprof/protobuf:kernel_stats.proto",
        "@org_xprof//plugin/xprof/protobuf:memory_profile.proto",
        "@org_xprof//plugin/xprof/protobuf:overview_page.proto",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline.proto",
        "@org_xprof//plugin/xprof/protobuf:roofline_model.proto",
        "@org_xprof//plugin/xprof/protobuf:diagnostics.proto",
        "@org_xprof//plugin/xprof/protobuf:hardware_types.proto",
        "@org_xprof//plugin/xprof/protobuf:steps_db.proto",
        "@org_xprof//plugin/xprof/protobuf:trace_events.proto",
        # Xprof protos - dependencies
        "@org_xprof//plugin/xprof/protobuf:topology.proto",
        "@org_xprof//plugin/xprof/protobuf:source_info.proto",
        "@org_xprof//plugin/xprof/protobuf:task.proto",
        "@org_xprof//plugin/xprof/protobuf:power_metrics.proto",
        "@org_xprof//plugin/xprof/protobuf:source_stats.proto",
        "@org_xprof//plugin/xprof/protobuf:tf_function.proto",
        # TSL profiler protos
        "@tsl//tsl/profiler/protobuf:xplane.proto",
        "@tsl//tsl/profiler/protobuf:profiler_options.proto",
        "@tsl//tsl/profiler/protobuf:profile.proto",
        "@tsl//tsl/profiler/protobuf:trace_events.proto",
    ],
    outs = ["proto_files.tar"],
    cmd = """
        # Create a temporary staging directory
        STAGING=$$(mktemp -d)
        mkdir -p $$STAGING/xla
        mkdir -p $$STAGING/xla/pjrt/proto
        mkdir -p $$STAGING/xla/service
        mkdir -p $$STAGING/xla/stream_executor/cuda
        mkdir -p $$STAGING/xla/tsl/protobuf
        mkdir -p $$STAGING/tsl/profiler/protobuf
        mkdir -p $$STAGING/google/protobuf
        mkdir -p $$STAGING/plugin/xprof/protobuf

        # Process each source file, preserving directory structure
        for src in $(SRCS); do
            # Route to correct directory based on path patterns
            if echo "$$src" | grep -q "xla/service"; then
                cp $$src $$STAGING/xla/service/
            elif echo "$$src" | grep -q "xla/pjrt/proto"; then
                cp $$src $$STAGING/xla/pjrt/proto/
            elif echo "$$src" | grep -q "xla/stream_executor/cuda"; then
                cp $$src $$STAGING/xla/stream_executor/cuda/
            elif echo "$$src" | grep -q "xla/stream_executor"; then
                cp $$src $$STAGING/xla/stream_executor/
            elif echo "$$src" | grep -q "xla/tsl/protobuf"; then
                cp $$src $$STAGING/xla/tsl/protobuf/
            elif echo "$$src" | grep -q "xla/xla"; then
                cp $$src $$STAGING/xla/
            elif echo "$$src" | grep -q "/xla:"; then
                cp $$src $$STAGING/xla/
            elif echo "$$src" | grep -q "tsl/profiler"; then
                cp $$src $$STAGING/tsl/profiler/protobuf/
            elif echo "$$src" | grep -q "xprof/protobuf"; then
                cp $$src $$STAGING/plugin/xprof/protobuf/
            else
                # Fallback: put in root
                cp $$src $$STAGING/
            fi
        done

        # Patch xla/autotune_results.proto to be package autotune_results
        sed -i 's/package xla;/package xla_autotuning;/g' $$STAGING/xla/autotune_results.proto
        sed -i 's/package xla;/package xla_autotuning;/g' $$STAGING/xla/autotuning.proto
        sed -i 's/package stream_executor.dnn;/package xla_tsl_dnn;/g' $$STAGING/xla/tsl/protobuf/dnn.proto

        # Update usages of stream_executor.dnn to xla_tsl_dnn
        find $$STAGING -name "*.proto" -type f -exec sed -i 's/stream_executor\\.dnn/xla_tsl_dnn/g' {} +

        # Update usages of xla.AutotuneResults to autotune_results.AutotuneResults
        find $$STAGING -name "*.proto" -type f -exec sed -i \
            -e 's/xla\\.AutotuneResults/xla_autotuning.AutotuneResults/g' \
            -e 's/xla\\.AutotuneLogs/xla_autotuning.AutotuneLogs/g' \
            -e 's/xla\\.AutotuneResult/xla_autotuning.AutotuneResult/g' \
            -e 's/xla\\.AutotuningLog/xla_autotuning.AutotuningLog/g' \
            -e 's/xla\\.CudnnVersion/xla_autotuning.CudnnVersion/g' \
            -e 's/xla\\.ComputeCapability/xla_autotuning.ComputeCapability/g' \
            -e 's/xla\\.TritonGemmConfigsProto/xla_autotuning.TritonGemmConfigsProto/g' \
            {} +

        # Copy google well-known types from the protobuf external repo
        # These are needed for imports like google/protobuf/any.proto
        if [ -d "external/com_google_protobuf/src/google/protobuf" ]; then
            cp external/com_google_protobuf/src/google/protobuf/any.proto $$STAGING/google/protobuf/ 2>/dev/null || true
            cp external/com_google_protobuf/src/google/protobuf/duration.proto $$STAGING/google/protobuf/ 2>/dev/null || true
            cp external/com_google_protobuf/src/google/protobuf/timestamp.proto $$STAGING/google/protobuf/ 2>/dev/null || true
            cp external/com_google_protobuf/src/google/protobuf/wrappers.proto $$STAGING/google/protobuf/ 2>/dev/null || true
            cp external/com_google_protobuf/src/google/protobuf/empty.proto $$STAGING/google/protobuf/ 2>/dev/null || true
            cp external/com_google_protobuf/src/google/protobuf/descriptor.proto $$STAGING/google/protobuf/ 2>/dev/null || true
        fi

        # Create tar archive
        tar -cf $@ -C $$STAGING .

        # Cleanup
        rm -rf $$STAGING
    """,
    tags = ["proto"],
    visibility = ["//visibility:public"],
)

# Julia proto bindings generation rule
# This invokes ProtoBuf.jl to generate Julia code from the staged protos
genrule(
    name = "julia_proto_bindings",
    srcs = [
        ":proto_files",
        "//:Project.toml",
        "//:Manifest.toml",
        "//:make-proto-bindings.jl",
    ],
    outs = ["proto_bindings.tar"],
    cmd = """
        # Create temp directories
        PROTO_DIR=$$(mktemp -d)
        OUT_DIR=$$(mktemp -d)
        
        # Extract proto files
        tar -xf $(location :proto_files) -C $$PROTO_DIR
        
        # Run Julia to generate bindings
        $$JULIA --color=yes --project=$(location //:Project.toml) \
            $(location //:make-proto-bindings.jl) \
            $$PROTO_DIR $$OUT_DIR
        
        # Create output tar
        tar -cf $@ -C $$OUT_DIR .
        
        # Cleanup
        rm -rf $$PROTO_DIR $$OUT_DIR
    """,
    tags = [
        "jlrule",
        "proto",
    ],
    visibility = ["//visibility:public"],
)

cc_toolchain_suite(
    name = "ygg_host_toolchain_suite",
    toolchains = {
        "k8": ":ygg_host_toolchain",
    },
)

cc_toolchain_suite(
    name = "ygg_cross_compile_toolchain_suite",
    toolchains = {
        "k8": ":ygg_x86_toolchain",
        "x64_windows": ":ygg_win_x86_toolchain",
        "darwin": ":ygg_darwin_x86_toolchain",
        "darwin_arm64": ":ygg_darwin_arm64_toolchain",
        "aarch64": ":ygg_aarch64_toolchain",
    },
)

cc_toolchain_suite(
    name = "beast_toolchain_suite",
    toolchains = {
        "k8": ":beast_toolchain",
    },
)

filegroup(name = "empty")

cc_toolchain(
    name = "ygg_host_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":ygg_host_toolchain_config",
    toolchain_identifier = "ygg_host_toolchain",
)

toolchain(
    name = "cc_toolchain_for_ygg_host",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    toolchain = ":ygg_host_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_config(
    name = "ygg_host_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/x86_64-linux-musl/bin/../x86_64-linux-musl/sys-root",
    compile_flags = [
        "-stdlib=libstdc++",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "k8",
    cxx_builtin_include_directories = [
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include/c++/GCC_VERSION",
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include/c++/GCC_VERSION/x86_64-linux-musl",
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include/c++/GCC_VERSION/backward",
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include",
        "/opt/x86_64-linux-musl/bin/../include/x86_64-unknown-linux-musl/c++/v1",
        "/opt/x86_64-linux-musl/bin/../include/c++/v1",
        "/opt/x86_64-linux-musl/x86_64-linux-musl/sys-root/usr/include",
        "/opt/x86_64-linux-musl/lib/clang/16/include",
        "/opt/x86_64-linux-musl/lib/clang/17/include",
        "/opt/x86_64-linux-musl/lib/clang/18/include",
        #         "/opt/x86_64-linux-musl/x86_64-linux-musl/include",
        #         "/opt/x86_64-linux-musl/lib/gcc/x86_64-linux-musl/10.2.0/include",
        #         "/opt/x86_64-linux-musl/x86_64-linux-musl/sys-root/usr/include",
        #         "/opt/x86_64-linux-musl/lib/gcc/x86_64-linux-musl/12.1.0/include",
        #         "/opt/x86_64-linux-musl/include",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "linux",
    link_flags = [
        "-fuse-ld=lld",
        #"--ld-path=/opt/x86_64-linux-musl/bin/ld.lld",
        "--ld-path=/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-ld.lld",
        "-stdlib=libstdc++",
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
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "",
    target_system_name = "x86_64-unknown-linux-gnu",
    tool_paths = {
        #"gcc": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-gcc",
        #"cpp": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-g++",
        #"ld": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-ld",
        "gcc": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-clang",
        "g++": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-clang++",
        "cpp": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-clang++",
        "ld": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-ld.lld",
        #"gcc": "/opt/x86_64-linux-musl/bin/clang",
        #"ld": "/opt/x86_64-linux-musl/bin/ld.lld",
        #"cpp": "/opt/x86_64-linux-musl/bin/clang++",
        "ar": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-ar",
        "llvm-cov": "/usr/lib/llvm-18/bin/llvm-cov",
        "nm": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-nm",
        "objdump": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-objdump",
        "strip": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-strip",
    },
    toolchain_identifier = "ygg_host",
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

cc_toolchain(
    name = "ygg_x86_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":ygg_x86_toolchain_config",
    toolchain_identifier = "ygg_x86_toolchain",
)

toolchain(
    name = "cc_toolchain_for_ygg_x86",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    toolchain = ":ygg_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# TODO distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        # TODO cxx_builtin_include_directories doesn't seem to be working, so we add the INCLUDE_PATHs manually
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "-isystem /opt/BB_TARGET/BB_TARGET/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "-stdlib=libstdc++",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "k8",
    cxx_builtin_include_directories = [
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "/opt/BB_TARGET/BB_TARGET/include",
        "/opt/BB_TARGET/BB_TARGET/sys-root",
        "/opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "/opt/x86_64-linux-musl/lib/clang/16/include",
        "/opt/x86_64-linux-musl/lib/clang/17/include",
        "/opt/x86_64-linux-musl/lib/clang/18/include",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "linux",
    link_flags = [
        "-stdlib=libstdc++",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    # TODO gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
    supports_start_end_lib = False,
    target_libc = "",
    target_system_name = "x86-unknown-linux-gnu",
    tool_paths = {
        "ar": "/opt/bin/BB_FULL_TARGET/ar",
        "as": "/opt/bin/BB_FULL_TARGET/as",
        "c++": "/opt/bin/BB_FULL_TARGET/c++",
        "c++filt": "/opt/bin/BB_FULL_TARGET/c++filt",
        "cc": "/opt/bin/BB_FULL_TARGET/clang",
        "clang": "/opt/bin/BB_FULL_TARGET/clang",
        "clang++": "/opt/bin/BB_FULL_TARGET/clang++",
        "cpp": "/opt/bin/BB_FULL_TARGET/cpp",
        "f77": "/opt/bin/BB_FULL_TARGET/f77",
        "g++": "/opt/bin/BB_FULL_TARGET/clang++",
        "gcc": "/opt/bin/BB_FULL_TARGET/clang",
        "gfortran": "/opt/bin/BB_FULL_TARGET/gfortran",
        "ld": "/opt/bin/BB_FULL_TARGET/ld",
        "ld.lld": "/opt/bin/BB_FULL_TARGET/ld.lld",
        "libtool": "/opt/bin/BB_FULL_TARGET/libtool",
        "lld": "/opt/bin/BB_FULL_TARGET/lld",
        "nm": "/opt/bin/BB_FULL_TARGET/nm",
        "objcopy": "/opt/bin/BB_FULL_TARGET/objcopy",
        "patchelf": "/opt/bin/BB_FULL_TARGET/patchelf",
        "ranlib": "/opt/bin/BB_FULL_TARGET/ranlib",
        "readelf": "/opt/bin/BB_FULL_TARGET/readelf",
        "strip": "/opt/bin/BB_FULL_TARGET/strip",
        # from host
        "llvm-cov": "/opt/x86_64-linux-musl/bin/llvm-cov",
        "llvm-profdata": "/opt/x86_64-linux-musl/bin/llvm-profdata",
        "objdump": "/usr/bin/objdump",
    },
    toolchain_identifier = "ygg_x86",
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

cc_toolchain(
    name = "ygg_aarch64_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":ygg_aarch64_toolchain_config",
    toolchain_identifier = "ygg_aarch64_toolchain",
)

toolchain(
    name = "cc_toolchain_for_ygg_aarch64",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    toolchain = ":ygg_aarch64_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# TODO distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_aarch64_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        # TODO cxx_builtin_include_directories doesn't seem to be working, so we add the INCLUDE_PATHs manually
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "-isystem /opt/BB_TARGET/BB_TARGET/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "aarch64",
    cxx_builtin_include_directories = [
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "/opt/BB_TARGET/BB_TARGET/include",
        "/opt/BB_TARGET/BB_TARGET/sys-root",
        "/opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "/opt/x86_64-linux-musl/lib/clang/16/include",
        "/opt/x86_64-linux-musl/lib/clang/17/include",
        "/opt/x86_64-linux-musl/lib/clang/18/include",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "linux",
    link_flags = [
        "-stdlib=libstdc++",
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
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    # TODO gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
    supports_start_end_lib = False,
    target_libc = "",
    target_system_name = "aarch64-unknown-linux-gnu",
    tool_paths = {
        "ar": "/opt/bin/BB_FULL_TARGET/ar",
        "as": "/opt/bin/BB_FULL_TARGET/as",
        "c++": "/opt/bin/BB_FULL_TARGET/c++",
        "c++filt": "/opt/bin/BB_FULL_TARGET/c++filt",
        "cc": "/opt/bin/BB_FULL_TARGET/clang",
        "clang": "/opt/bin/BB_FULL_TARGET/clang",
        "clang++": "/opt/bin/BB_FULL_TARGET/clang++",
        "cpp": "/opt/bin/BB_FULL_TARGET/cpp",
        "f77": "/opt/bin/BB_FULL_TARGET/f77",
        "g++": "/opt/bin/BB_FULL_TARGET/clang++",
        "gcc": "/opt/bin/BB_FULL_TARGET/clang",
        "gfortran": "/opt/bin/BB_FULL_TARGET/gfortran",
        "ld": "/opt/bin/BB_FULL_TARGET/ld",
        "ld.lld": "/opt/bin/BB_FULL_TARGET/ld.lld",
        "libtool": "/opt/bin/BB_FULL_TARGET/libtool",
        "lld": "/opt/bin/BB_FULL_TARGET/lld",
        "nm": "/opt/bin/BB_FULL_TARGET/nm",
        "objcopy": "/opt/bin/BB_FULL_TARGET/objcopy",
        "patchelf": "/opt/bin/BB_FULL_TARGET/patchelf",
        "ranlib": "/opt/bin/BB_FULL_TARGET/ranlib",
        "readelf": "/opt/bin/BB_FULL_TARGET/readelf",
        "strip": "/opt/bin/BB_FULL_TARGET/strip",
        # from host
        "llvm-cov": "/opt/x86_64-linux-musl/bin/llvm-cov",
        "llvm-profdata": "/opt/x86_64-linux-musl/bin/llvm-profdata",
        "objdump": "/usr/bin/objdump",
    },
    toolchain_identifier = "ygg_aarch64",
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

cc_toolchain(
    name = "ygg_darwin_x86_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":ygg_darwin_x86_toolchain_config",
    toolchain_identifier = "ygg_darwin_x86_toolchain",
)

toolchain(
    name = "cc_toolchain_for_ygg_darwin_x86",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:macos",
    ],
    toolchain = ":ygg_darwin_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# TODO distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_darwin_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-stdlib=libc++",
    ],
    compiler = "compiler",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "k8",
    cxx_builtin_include_directories = [
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "/opt/BB_TARGET/BB_TARGET/include",
        "/opt/BB_TARGET/BB_TARGET/sys-root",
        "/opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "/opt/x86_64-linux-musl/lib/clang/16/include",
        "/opt/x86_64-linux-musl/lib/clang/17/include",
        "/opt/x86_64-linux-musl/lib/clang/18/include",
    ],
    dbg_compile_flags = [
        "-g",
    ],
    host_system_name = "linux",
    link_flags = [
        "-fuse-ld=lld",
        "-stdlib=libc++",
    ],
    link_libs = [
        "-lc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "",
    target_system_name = "x86_64-apple-macos",
    tool_paths = {
        "ar": "/opt/bin/BB_FULL_TARGET/ar",
        "as": "/opt/bin/BB_FULL_TARGET/as",
        "c++": "/opt/bin/BB_FULL_TARGET/c++",
        "c++filt": "/opt/bin/BB_FULL_TARGET/c++filt",
        "cc": "/opt/bin/BB_FULL_TARGET/cc",
        "clang": "/opt/bin/BB_FULL_TARGET/clang",
        "clang++": "/opt/bin/BB_FULL_TARGET/clang++",
        "cpp": "/opt/bin/BB_FULL_TARGET/cpp",
        "f77": "/opt/bin/BB_FULL_TARGET/f77",
        "gcc": "/opt/bin/BB_FULL_TARGET/clang",
        "g++": "/opt/bin/BB_FULL_TARGET/clang++",
        # "g++": "/opt/bin/BB_FULL_TARGET/g++",
        # "gcc": "/opt/bin/BB_FULL_TARGET/gcc",
        "gfortran": "/opt/bin/BB_FULL_TARGET/gfortran",
        "ld": "/opt/bin/BB_FULL_TARGET/ld",
        "ld.lld": "/opt/bin/BB_FULL_TARGET/ld.lld",
        "libtool": "/opt/bin/BB_FULL_TARGET/libtool",
        "lld": "/opt/bin/BB_FULL_TARGET/lld",
        "nm": "/opt/bin/BB_FULL_TARGET/nm",
        "objcopy": "/opt/bin/BB_FULL_TARGET/objcopy",
        "patchelf": "/opt/bin/BB_FULL_TARGET/patchelf",
        "ranlib": "/opt/bin/BB_FULL_TARGET/ranlib",
        "readelf": "/opt/bin/BB_FULL_TARGET/readelf",
        "strip": "/opt/bin/BB_FULL_TARGET/strip",
        # from host
        "llvm-cov": "/opt/x86_64-linux-musl/bin/llvm-cov",
        "llvm-profdata": "/opt/x86_64-linux-musl/bin/llvm-profdata",
        "objdump": "/usr/bin/objdump",
    },
    toolchain_identifier = "ygg_darwin_x86",
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

cc_toolchain(
    name = "ygg_darwin_arm64_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":ygg_darwin_arm64_toolchain_config",
    toolchain_identifier = "ygg_darwin_arm64_toolchain",
)

toolchain(
    name = "cc_toolchain_for_ygg_darwin_arm64",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:arm64",
        "@platforms//os:macos",
    ],
    toolchain = ":ygg_darwin_arm64_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# TODO distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_darwin_arm64_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-stdlib=libc++",
    ],
    compiler = "compiler",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "arm64",
    cxx_builtin_include_directories = [
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "/opt/BB_TARGET/BB_TARGET/include",
        "/opt/BB_TARGET/BB_TARGET/sys-root",
        "/opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "/opt/x86_64-linux-musl/lib/clang/16/include",
        "/opt/x86_64-linux-musl/lib/clang/17/include",
        "/opt/x86_64-linux-musl/lib/clang/18/include",
    ],
    dbg_compile_flags = [
        "-g",
        "-stdlib=libc++",
    ],
    host_system_name = "linux",
    link_flags = [
        "-stdlib=libc++",
    ],
    link_libs = [
        "-lc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
        "-stdlib=libc++",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "",
    target_system_name = "arm64-apple-macos",
    tool_paths = {
        "ar": "/opt/bin/BB_FULL_TARGET/ar",
        "as": "/opt/bin/BB_FULL_TARGET/as",
        "c++": "/opt/bin/BB_FULL_TARGET/c++",
        "c++filt": "/opt/bin/BB_FULL_TARGET/c++filt",
        "cc": "/opt/bin/BB_FULL_TARGET/cc",
        "clang": "/opt/bin/BB_FULL_TARGET/clang",
        "clang++": "/opt/bin/BB_FULL_TARGET/clang++",
        "cpp": "/opt/bin/BB_FULL_TARGET/cpp",
        "f77": "/opt/bin/BB_FULL_TARGET/f77",
        "gcc": "/opt/bin/BB_FULL_TARGET/clang",
        "g++": "/opt/bin/BB_FULL_TARGET/clang++",
        # "g++": "/opt/bin/BB_FULL_TARGET/g++",
        # "gcc": "/opt/bin/BB_FULL_TARGET/gcc",
        "gfortran": "/opt/bin/BB_FULL_TARGET/gfortran",
        "ld": "/opt/bin/BB_FULL_TARGET/ld",
        "ld.lld": "/opt/bin/BB_FULL_TARGET/ld.lld",
        "libtool": "/opt/bin/BB_FULL_TARGET/libtool",
        "lld": "/opt/bin/BB_FULL_TARGET/lld",
        "nm": "/opt/bin/BB_FULL_TARGET/nm",
        "objcopy": "/opt/bin/BB_FULL_TARGET/objcopy",
        "patchelf": "/opt/bin/BB_FULL_TARGET/patchelf",
        "ranlib": "/opt/bin/BB_FULL_TARGET/ranlib",
        "readelf": "/opt/bin/BB_FULL_TARGET/readelf",
        "strip": "/opt/bin/BB_FULL_TARGET/strip",
        # from host
        "llvm-cov": "/opt/x86_64-linux-musl/bin/llvm-cov",
        "llvm-profdata": "/opt/x86_64-linux-musl/bin/llvm-profdata",
        "objdump": "/usr/bin/objdump",
    },
    toolchain_identifier = "ygg_darwin_arm64",
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

cc_toolchain(
    name = "ygg_win_x86_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":ygg_win_x86_toolchain_config",
    toolchain_identifier = "ygg_win_x86_toolchain",
)

toolchain(
    name = "cc_toolchain_for_ygg_win_x86",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":ygg_win_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# TODO distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_win_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-stdlib=libstdc++",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "/opt/BB_TARGET/BB_TARGET/include",
        "/opt/BB_TARGET/BB_TARGET/sys-root",
        "/opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "/opt/x86_64-linux-musl/lib/clang/16/include",
        "/opt/x86_64-linux-musl/lib/clang/17/include",
        "/opt/x86_64-linux-musl/lib/clang/18/include",
    ],
    dbg_compile_flags = [
        "-g",
    ],
    host_system_name = "linux",
    link_flags = [
        "-fuse-ld=lld",
        "-stdlib=libstdc++",
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
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = False,
    target_libc = "",
    target_system_name = "x86_64-apple-macos",
    tool_paths = {
        "ar": "/opt/bin/BB_FULL_TARGET/ar",
        "as": "/opt/bin/BB_FULL_TARGET/as",
        "c++": "/opt/bin/BB_FULL_TARGET/c++",
        "c++filt": "/opt/bin/BB_FULL_TARGET/c++filt",
        "cc": "/opt/bin/BB_FULL_TARGET/cc",
        "clang": "/opt/bin/BB_FULL_TARGET/clang",
        "clang++": "/opt/bin/BB_FULL_TARGET/clang++",
        "cpp": "/opt/bin/BB_FULL_TARGET/cpp",
        "f77": "/opt/bin/BB_FULL_TARGET/f77",
        "gcc": "/opt/bin/BB_FULL_TARGET/clang",
        "g++": "/opt/bin/BB_FULL_TARGET/clang++",
        # "g++": "/opt/bin/BB_FULL_TARGET/g++",
        # "gcc": "/opt/bin/BB_FULL_TARGET/gcc",
        "gfortran": "/opt/bin/BB_FULL_TARGET/gfortran",
        "ld": "/opt/bin/BB_FULL_TARGET/ld",
        "ld.lld": "/opt/bin/BB_FULL_TARGET/ld.lld",
        "libtool": "/opt/bin/BB_FULL_TARGET/libtool",
        "lld": "/opt/bin/BB_FULL_TARGET/lld",
        "nm": "/opt/bin/BB_FULL_TARGET/nm",
        "objcopy": "/opt/bin/BB_FULL_TARGET/objcopy",
        "patchelf": "/opt/bin/BB_FULL_TARGET/patchelf",
        "ranlib": "/opt/bin/BB_FULL_TARGET/ranlib",
        "readelf": "/opt/bin/BB_FULL_TARGET/readelf",
        "strip": "/opt/bin/BB_FULL_TARGET/strip",
        # from host
        "llvm-cov": "/opt/x86_64-linux-musl/bin/llvm-cov",
        "llvm-profdata": "/opt/x86_64-linux-musl/bin/llvm-profdata",
        "objdump": "/usr/bin/objdump",
    },
    toolchain_identifier = "ygg_win_x86",
    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
        "-Wno-unused-command-line-argument",
        "-Wno-gnu-offsetof-extensions",
        "-mxsave",
    ],
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
        "-stdlib=libstdc++",
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
    dbg_compile_flags = [
        "-g",
        "-I/usr/include/c++/11",
    ],
    host_system_name = "linux",
    link_flags = [
        "-fuse-ld=lld",
        "--ld-path=/home/wmoses/llvms/llvm16/install/bin/ld.lld",
        "-stdlib=libstdc++",
        "-static-libstdc++",
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

platform(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

platform(
    name = "darwin_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

platform(
    name = "win_x86_64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
)

platform(
    name = "win_aarch64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:aarch64",
    ],
)

cc_library(
    name = "ReactantExtraLib",
    srcs = glob(
        [
            "*.cpp",
        ],
    ) + [
        "@enzyme_ad//src/enzyme_ad/jax:gpu.cc",
        "@enzyme_ad//src/enzyme_ad/jax:cpu.cc",
        # "@xla//xla/service/gpu:backend_configs.pb.cc",
        # "@xla//xla:autotuning.pb.cc",
        # "@xla//xla:autotune_results.pb.cc",
        # "@xla//xla/service:buffer_assignment.pb.cc",
    ],
    hdrs = glob([
        "*.h",
    ]) + [
        "@enzyme_ad//src/enzyme_ad/jax:RegistryUtils.h",
    ],
    copts = [
        "-Werror=unused-variable",
        "-Werror=unused-but-set-variable",
        "-Werror=return-type",
        "-Werror=unused-result",
        "-Wno-error=stringop-truncation",
    ] + if_cuda(["-DREACTANT_CUDA=1"]) + if_rocm(["-DREACTANT_ROCM=1"]),
    linkopts = select({
        "//conditions:default": [],
        "@bazel_tools//src/conditions:darwin": [
            "-Wl,-exported_symbol,_stablehlo*",
            "-Wl,-exported_symbol,_enzymexla*",
            "-Wl,-exported_symbol,_mlir*",
            "-Wl,-exported_symbol,_sdy*",
            "-Wl,-exported_symbol,_EnzymeJaXMapSymbol",
            "-Wl,-exported_symbol,_InitializeLogs",
            "-Wl,-exported_symbol,_SetLogLevel",
            "-Wl,-exported_symbol,_SetModuleLogLevel",
            "-Wl,-exported_symbol,_GetDefaultTargetTriple",
            "-Wl,-exported_symbol,_enzymeActivityAttrGet",
            "-Wl,-exported_symbol,_UninitPJRTBuffer",
            "-Wl,-exported_symbol,_CopyToBuffer",
            "-Wl,-exported_symbol,_CopyFromBuffer",
            "-Wl,-exported_symbol,_MakeCPUClient",
            "-Wl,-exported_symbol,_MakeGPUClient",
            "-Wl,-exported_symbol,_MakeTPUClient",
            "-Wl,-exported_symbol,_LoadPjrtPlugin",
            "-Wl,-exported_symbol,_InitializePjrtPlugin",
            "-Wl,-exported_symbol,_MakeClientUsingPluginAPI",
            "-Wl,-exported_symbol,_GetCApiClient",
            "-Wl,-exported_symbol,_ClientNumDevices",
            "-Wl,-exported_symbol,_ClientNumAddressableDevices",
            "-Wl,-exported_symbol,_ClientProcessIndex",
            "-Wl,-exported_symbol,_ClientGetDevice",
            "-Wl,-exported_symbol,_ClientGetAddressableDevice",
            "-Wl,-exported_symbol,_PjRtDeviceGetAllocatorStats",
            "-Wl,-exported_symbol,_ExecutableFree",
            "-Wl,-exported_symbol,_BufferToDevice",
            "-Wl,-exported_symbol,_BufferToClient",
            "-Wl,-exported_symbol,_DeviceToClient",
            "-Wl,-exported_symbol,_PjRtBufferFree",
            "-Wl,-exported_symbol,_UnsafeBufferPointer",
            "-Wl,-exported_symbol,_ArrayFromHostBuffer",
            "-Wl,-exported_symbol,_BufferOnCPU",
            "-Wl,-exported_symbol,_CopyBufferToDevice",
            "-Wl,-exported_symbol,_BufferToHost",
            "-Wl,-exported_symbol,_FreeClient",
            "-Wl,-exported_symbol,_ClientCompile",
            "-Wl,-exported_symbol,_ClientCompileWithProto",
            "-Wl,-exported_symbol,_ConvertLLVMStrToMLIR",
            "-Wl,-exported_symbol,_LinkInModule",
            "-Wl,-exported_symbol,_FreeFuture",
            "-Wl,-exported_symbol,_FutureIsReady",
            "-Wl,-exported_symbol,_FutureAwait",
            "-Wl,-exported_symbol,_XLAExecute",
            "-Wl,-exported_symbol,_RegisterDialects",
            "-Wl,-exported_symbol,_InitializeRegistry",
            "-Wl,-exported_symbol,_InitializePasses",
            "-Wl,-exported_symbol,_RegisterCustomCallTarget",
            "-Wl,-exported_symbol,_ConvertLLVMToMLIR",
            "-Wl,-exported_symbol,_RegisterEnzymeXLAGPUHandler",
            "-Wl,-exported_symbol,_ReactantThrowError",
            "-Wl,-exported_symbol,_ReactantHandleCuResult",
            "-Wl,-exported_symbol,_CreateProfilerSession",
            "-Wl,-exported_symbol,_ProfilerSessionCollectData",
            "-Wl,-exported_symbol,_ProfilerSessionDelete",
            "-Wl,-exported_symbol,_ProfilerServerStart",
            "-Wl,-exported_symbol,_ProfilerServerStop",
            "-Wl,-exported_symbol,_ProfilerActivityStart",
            "-Wl,-exported_symbol,_ProfilerActivityEnd",
            "-Wl,-exported_symbol,_ReactantFuncSetArgAttr",
            "-Wl,-exported_symbol,_ReactantHermeticCudaGetVersion",
            "-Wl,-exported_symbol,_ReactantCudaDriverGetVersion",
            "-Wl,-exported_symbol,_ReactantLLVMParseCommandLineOptions",
            "-Wl,-exported_symbol,_ReactantCudaDeviceGetComputeCapalilityMajor",
            "-Wl,-exported_symbol,_ReactantCudaDeviceGetComputeCapalilityMinor",
            "-Wl,-exported_symbol,_ReactantCudaDeviceGetWarpSizeInThreads",
            "-Wl,-exported_symbol,_ReactantCudaDeviceGetProperties",
            "-Wl,-exported_symbol,_ReactantCudaGetRegsSpillsMaxThreadsFromBinary",
            "-Wl,-exported_symbol,_PjRtDeviceGetLocalDeviceId",
            "-Wl,-exported_symbol,_PjRtDeviceGetGlobalDeviceId",
            "-Wl,-exported_symbol,_PjRtDeviceGetLocalHardwareId",
            "-Wl,-exported_symbol,_XLAExecuteSharded",
            "-Wl,-exported_symbol,_ClientGetPlatformName",
            "-Wl,-exported_symbol,_RegisterEnzymeXLACPUHandler",
            "-Wl,-exported_symbol,_PjRtLoadedExecutableGetClient",
            "-Wl,-exported_symbol,_ReactantFuncSetResultAttr",
            "-Wl,-exported_symbol,_BufferShape",
            "-Wl,-exported_symbol,_BufferNDimensions",
            "-Wl,-exported_symbol,_BufferPrimitiveType",
            "-Wl,-exported_symbol,_PjRtLoadedExecutableGetOuputShardings",
            "-Wl,-exported_symbol,_PjRtLoadedExecutableGetParameterShardings",
            "-Wl,-exported_symbol,_PjRtLoadedExecutableGetHloModules",
            "-Wl,-exported_symbol,_HloModuleToString",
            "-Wl,-exported_symbol,_FreeHloModule",
            "-Wl,-exported_symbol,_PjRtLoadedExecutableNumReplicas",
            "-Wl,-exported_symbol,_PjRtLoadedExecutableNumPartitions",
            "-Wl,-exported_symbol,_ifrt_*",
            "-Wl,-exported_symbol,_pjrt_*",
            "-Wl,-exported_symbol,_reactant_*",
            "-Wl,-exported_symbol,_free_op_sharding",
            "-Wl,-exported_symbol,_free_hlo_sharding",
            "-Wl,-exported_symbol,_hlo_sharding_from_op_sharding",
            "-Wl,-exported_symbol,_hlo_sharding_to_op_sharding",
            "-Wl,-exported_symbol,_hlo_sharding_to_string",
            "-Wl,-exported_symbol,_DeviceGetKind",
            "-Wl,-exported_symbol,_GetDistributedRuntimeClient",
            "-Wl,-exported_symbol,_free_distributed_runtime_client",
            "-Wl,-exported_symbol,_distributed_runtime_client_connect",
            "-Wl,-exported_symbol,_distributed_runtime_client_shutdown",
            "-Wl,-exported_symbol,_GetDistributedRuntimeService",
            "-Wl,-exported_symbol,_free_distributed_runtime_service",
            "-Wl,-exported_symbol,_distributed_runtime_service_shutdown",
            "-Wl,-exported_symbol,_ClientGetDevices",
            "-Wl,-exported_symbol,_ClientGetAddressableDevices",
            "-Wl,-exported_symbol,_hloShardingFromTensorShardingAttr",
            "-Wl,-exported_symbol,_op_sharding_*",
            "-Wl,-exported_symbol,_hloShardingToTensorShardingAttr",
            "-Wl,-exported_symbol,_dump_operation",
            "-Wl,-exported_symbol,_hlo_sharding_*",
            "-Wl,-exported_symbol,_free_ifrt_sharding",
            "-Wl,-exported_symbol,_addSdyPropagationPipeline",
            "-Wl,-exported_symbol,_mlirGetFunctionTypeFromOperation",
            "-Wl,-exported_symbol,_mlirIsFunctionOpInterface",
            "-Wl,-exported_symbol,_convertMlirModuleToHloModule",
            "-Wl,-exported_symbol,_parseAndReturnUnverifiedHloModule",
            "-Wl,-exported_symbol,_hloModuleGetEntryComputation",
            "-Wl,-exported_symbol,_freeHloComputation",
            "-Wl,-exported_symbol,_hloComputationToString",
            "-Wl,-exported_symbol,_hloComputationGetInstructionsPostOrder",
            "-Wl,-exported_symbol,_freeHloInstruction",
            "-Wl,-exported_symbol,_hloInstructionToString",
            "-Wl,-exported_symbol,_hloInstructionGetOpcode",
            "-Wl,-exported_symbol,_hloInstructionIsFusion",
            "-Wl,-exported_symbol,_hloInstructionGetFusionKind",
            "-Wl,-exported_symbol,_hloOpcodeToString",
            "-Wl,-exported_symbol,_hloFusionKindToString",
            "-Wl,-exported_symbol,_hloInstructionFusedInstructionsComputation",
            "-Wl,-exported_symbol,_deviceDescriptionToString",
            "-Wl,-exported_symbol,_CudaGetStreamExecutorDeviceDescription",
            "-Wl,-exported_symbol,_CreateGPUPerformanceModel",
            "-Wl,-exported_symbol,_RunAnalysisOnHloModule",
            "-Wl,-exported_symbol,_EstimateRunTimeForInstruction",
            "-Wl,-exported_symbol,_registerReactantXLAFFI",
            "-Wl,-exported_symbol,_InitializeXProfStubs",
            "-Wl,-exported_symbol,_StartGrpcServer",
            "-Wl,-exported_symbol,_XSpaceToToolsData",
        ],
    }),
    linkstatic = True,
    deps = [
        "@enzyme_ad//src/enzyme_ad/jax:RegistryUtils",
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FromLLVMIRTranslation",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        # "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:LLVMIRToLLVMTranslation",
        "@llvm-project//mlir:LLVMIRToNVVMTranslation",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@enzyme_ad//src/enzyme_ad/jax:TransformOps",
        "@enzyme_ad//src/enzyme_ad/jax:XLADerivatives",
        "@enzyme_ad//src/enzyme_ad/jax:CInterface",
        # "@enzyme_ad//src/enzyme_ad/jax:gpu",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/ffi:ffi_api",
        "@stablehlo//:chlo_ops",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/pjrt/distributed",
        "@xla//xla/pjrt/distributed:client",
        "@xla//xla/pjrt/distributed:service",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",

        # "@xla//xla:xla_proto_cc",
        # "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/stream_executor:device_description_proto_cc_impl",
        "@xla//xla/service:metrics_proto_cc",
        "@xla//xla/service:metrics_proto_cc_impl",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service/cpu:cpu_compiler",
        "@xla//xla/stream_executor/tpu:tpu_on_demand_compiler",
        "@xla//xla/stream_executor/tpu:tpu_executor",
        "@xla//xla/stream_executor/tpu:tpu_transfer_manager",
        "@xla//xla/service/cpu:cpu_transfer_manager",
        "@xla//xla/tsl/protobuf:protos_all_cc_impl",
        "@xla//xla/tsl/framework:allocator_registry_impl",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/python/ifrt_proxy/server:grpc_server",
        "@xla//xla/python/ifrt_proxy/client:grpc_client",
        "@xla//xla/python/ifrt_proxy/client:registry",
        # "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        # "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/ifrt/ir:ifrt_ir_program",
        "@xla//xla/ffi:call_frame",
        # "@com_google_protobuf//:protobuf",
        "@tsl//tsl/profiler/lib:profiler_session_impl",
        "@tsl//tsl/profiler/lib:profiler_factory_impl",
        "@tsl//tsl/profiler/lib:profiler_controller",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
        "@xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client",
        "@xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@xla//xla/tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profile_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:trace_events_proto_cc_impl",
        "@xla//xla/backends/profiler/cpu:host_tracer",
        "@xla//xla/backends/profiler/cpu:host_tracer_impl",
        "@xla//xla/backends/profiler/cpu:metadata_collector",
        "@xla//xla/backends/profiler/cpu:metadata_utils",
        "@xla//xla/backends/profiler/tpu:tpu_tracer",
        "@xla//xla/python:profiler_utils",
        "@org_xprof//xprof/pywrap:profiler_plugin_impl",
        "@org_xprof//plugin/xprof/worker:stub_factory",
        "@org_xprof//xprof/convert:tool_options",
        "@tsl//tsl/platform:env_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/mlir/utils:type_util",
        "@stablehlo//:stablehlo_capi_objects",
        "@stablehlo//:chlo_capi_objects",
        "@shardy//shardy/integrations/c:sdy_capi_objects",
        "@shardy//shardy/dialect/sdy/transforms:passes",
        "@shardy//shardy/dialect/sdy/transforms/propagation:passes",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log:globals",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:CAPILLVMObjects",

        # Broken upstream x/ref https://github.com/jax-ml/jax/issues/33344
        # "@jax//jaxlib/mosaic:tpu_dialect_capi_objects",
        "@jax//jaxlib/triton:triton_dialect_capi_objects",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@stablehlo//:linalg_passes",
        "@stablehlo//:tosa_passes",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_passes_optimization",
        "@stablehlo//stablehlo/tests:check_ops",
        "@tsl//tsl/platform:env",
        "@xla//xla/tsl/protobuf:dnn_proto_cc_impl",
        "@xla//xla/tsl/protobuf:histogram_proto_cc",
        "@xla//xla/tsl/protobuf:histogram_proto_cc_impl",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/pjrt/proto:compile_options_proto_cc_impl",
        "@com_google_absl//absl/status:statusor",
    ] + if_cuda([
        "@xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl",
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@jax//jaxlib/cuda:cuda_gpu_kernels",
        "@xla//xla/backends/profiler:profiler_backends",
        "@xla//xla/backends/profiler/gpu:device_tracer",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_internal",
        "@xla//xla/service/gpu:gpu_transfer_manager",
        "@xla//xla/service/gpu:nvptx_compiler",
        "@xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
        "@xla//xla/service/gpu/model:hlo_op_profiles",
        "@xla//xla/stream_executor:cuda_platform",
        "@xla//xla/stream_executor:kernel",
        "@xla//xla/stream_executor/cuda:all_runtime",
    ]) + if_rocm([
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/stream_executor:rocm_platform",
        "@xla//xla/service/gpu:amdgpu_compiler",
        "@xla//xla/backends/profiler/gpu:device_tracer",
    ]) + select({
        # gloo tcp transport only builds on linux
        "@xla//xla/tsl:macos": [
            "@gloo//:transport_uv",
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
            "@xla//xla/backends/cpu/collectives:mpi_collectives",
            "@xla//xla/tsl/platform/default:platform_port",
        ],
        "@xla//xla/tsl:windows": [
            "@xla//xla/tsl/platform/windows:platform_port",
        ],
        "//conditions:default": [
            "@gloo//:transport_tcp",
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
            "@xla//xla/backends/cpu/collectives:mpi_collectives",
            "@xla//xla/tsl/platform/default:platform_port",
        ],
    }),
    alwayslink = True,
)

# cc_shared_library(
cc_binary(
    name = "libReactantExtra.so",
    linkshared = 1,  ## important
    linkstatic = 1,  ## important
    deps = [":ReactantExtraLib"],
)

cc_binary(
    name = "mlir-jl-tblgen",
    srcs = [
        "//tblgen:jl-generators.cc",
        "//tblgen:mlir-jl-tblgen.cc",
    ],
    tags = [
        "optional",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl_cc_library(
    name = "BuiltinJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Builtin.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/IR/BuiltinOps.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "ArithJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Arith.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Arith/IR/ArithOps.td",
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "AffineJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Affine.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Affine/IR/AffineOps.td",
    deps = [
        "@llvm-project//mlir:AffineOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "FuncJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Func.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Func/IR/FuncOps.td",
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

gentbl_cc_library(
    name = "LlvmJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Llvm.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/LLVMIR/LLVMOps.td",
    deps = [
        "@llvm-project//mlir:LLVMOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "MemRefJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "MemRef.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/MemRef/IR/MemRefOps.td",
    deps = [
        "@llvm-project//mlir:MemRefOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "NvvmIncJLGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Nvvm.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/LLVMIR/NVVMOps.td",
    deps = [
        "@llvm-project//mlir:NVVMOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "GpuIncJLGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Gpu.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/GPU/IR/GPUOps.td",
    deps = [
        "@llvm-project//mlir:GPUOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "MPIIncJLGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "MPI.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/MPI/IR/MPIOps.td",
    deps = [
        "@llvm-project//mlir:MPITdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Enzyme.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@enzyme//:Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [
        "@enzyme//:EnzymeDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeXLAJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "EnzymeXLA.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@enzyme_ad//src/enzyme_ad/jax:Dialect/EnzymeXLAOps.td",
    deps = [
        "@enzyme//:EnzymeDialectTdFiles",
        "@enzyme_ad//src/enzyme_ad/jax:EnzymeXLADialectTdFiles",
        "@llvm-project//mlir:GPUOpsTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "TritonExtJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "TritonExt.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@enzyme_ad//src/enzyme_ad/jax:Dialect/TritonExt/Ops.td",
    deps = [
        "@enzyme_ad//src/enzyme_ad/jax:TritonExtDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TPUJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "TPU.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@jax//jaxlib/mosaic:dialect/tpu/tpu.td",
    deps = [
        "@jax//jaxlib/mosaic:tpu_td_files",
    ],
)

gentbl_cc_library(
    name = "MosaicGPUJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "MosaicGPU.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@jax//jaxlib/mosaic/dialect/gpu:mosaic_gpu.td",
    deps = [
        "@jax//jaxlib/mosaic/dialect/gpu:mosaic_gpu_td_files",
    ],
)

gentbl_cc_library(
    name = "TritonJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Triton.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@jax//jaxlib/triton:triton.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@triton//:td_files",
    ],
)

gentbl_cc_library(
    name = "StableHLOJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "StableHLO.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@stablehlo//:stablehlo/dialect/StablehloOps.td",
    deps = [
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "CHLOJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "CHLO.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@stablehlo//:stablehlo/dialect/ChloOps.td",
    deps = [
        "@stablehlo//:chlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "VHLOJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "VHLO.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@stablehlo//:stablehlo/dialect/VhloOps.td",
    deps = [
        "@stablehlo//:vhlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "ShardyJLIncGen",
    includes = ["external/shardy"],
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Shardy.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@shardy//shardy/dialect/sdy/ir:ops.td",
    deps = [
        "@shardy//shardy/dialect/sdy/ir:sdy_td_files",
    ],
)

gentbl_cc_library(
    name = "SparseTensorJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "SparseTensor.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.td",
    deps = [
        "@llvm-project//mlir:SparseTensorTdFiles",
    ],
)

genrule(
    name = "libMLIR_h.jl",
    srcs = [
        "@llvm-project//mlir:include/mlir-c/Bindings/Python/Interop.h",
        "@llvm-project//llvm:include/llvm-c/Support.h",
        "@llvm-project//llvm:include/llvm-c/DataTypes.h",
        "@llvm-project//llvm:include/llvm-c/ExternC.h",
        "@llvm-project//llvm:include/llvm-c/Types.h",
        "@llvm-project//mlir:c_headers",
        "@llvm-project//mlir:ConversionPassIncGen_filegroup",
        "@llvm-project//mlir:TransformsPassIncGen_filegroup",
        "@llvm-project//mlir:SparseTensorPassIncGen_filegroup",
        "@llvm-project//mlir:LinalgPassIncGen_filegroup",
        "@llvm-project//mlir:AsyncPassIncGen_filegroup",
        "@llvm-project//mlir:GPUPassIncGen_filegroup",
        "@stablehlo//:stablehlo/integrations/c/StablehloAttributes.h",
        "@stablehlo//:stablehlo/integrations/c/StablehloDialect.h",
        "@stablehlo//:stablehlo/integrations/c/StablehloDialectApi.h",
        "@stablehlo//:stablehlo/integrations/c/StablehloTypes.h",
        "@shardy//shardy/integrations/c:attributes.h",
        "@jax//jaxlib/triton:triton_dialect_capi.h",
        "@jax//jaxlib/mosaic:dialect/tpu/integrations/c/tpu_dialect.h",
        "@jax//jaxlib/mosaic:dialect/tpu/integrations/c/tpu_passes.capi.h.inc",
        "@jax//jaxlib/mosaic/dialect/gpu:integrations/c/attributes.h",
        "@jax//jaxlib/mosaic/dialect/gpu:integrations/c/gpu_dialect.h",
        "@enzyme_ad//src/enzyme_ad/jax:Integrations/c/EnzymeXLA.h",
        "//:Project.toml",
        "//:Manifest.toml",
        "//:wrap.toml",
        "//:missing_defs.jl",
        "//:make.jl",
    ],
    outs = ["libMLIR_h.jl"],
    cmd = "$$JULIA \"--color=yes\" \"--project=$(location //:Project.toml)\" \"$(location //:make.jl)\" \"$(location @llvm-project//mlir:include/mlir-c/Bindings/Python/Interop.h)\" \"$(location @llvm-project//llvm:include/llvm-c/Support.h)\" \"$(locations @llvm-project//mlir:ConversionPassIncGen_filegroup)\" \"$(location @stablehlo//:stablehlo/integrations/c/StablehloAttributes.h)\" \"$(location @shardy//shardy/integrations/c:attributes.h)\" \"$(location @jax//jaxlib/triton:triton_dialect_capi.h)\" \"$(location @jax//jaxlib/mosaic:dialect/tpu/integrations/c/tpu_dialect.h)\" \"$(location @jax//jaxlib/mosaic/dialect/gpu:integrations/c/gpu_dialect.h)\" \"$(location @enzyme_ad//src/enzyme_ad/jax:Integrations/c/EnzymeXLA.h)\" \"$@\"",
    tags = [
        "jlrule",
    ],
)
