load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
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
        "@xla//xla/service:buffer_assignment.proto",
        "@xla//xla/service:gpu_topology.proto",
        "@xla//xla/service:hlo.proto",
        "@xla//xla/service:metrics.proto",
        "@xla//xla/service:shaped_slice.proto",
        "@xla//xla/service/cpu:executable.proto",
        "@xla//xla/backends/cpu:xnn_fusion_options.proto",
        "@xla//xla/backends/cpu:ynn_fusion_options.proto",
        "@xla//xla/backends/cpu/runtime:thunk.proto",
        "@xla//xla/pjrt/proto:execute_options.proto",
        "@xla//xla/pjrt/proto:compile_options.proto",
        "@xla//xla/stream_executor:device_description.proto",
        "@xla//xla/stream_executor/cuda:cublas_lt_matmul_algorithm.proto",
        "@xla//xla/stream_executor/cuda:cuda_compute_capability.proto",
        "@xla//xla/stream_executor/gpu:blas_lt_matmul_algorithm.proto",
        "@xla//xla/stream_executor/sycl:oneapi_compute_capability.proto",
        "@xla//xla/backends/autotuner:backends.proto",
        "@xla//xla/tsl/protobuf:dnn.proto",
        # Xprof protos - main files
        "@org_xprof//plugin/xprof/protobuf:op_metrics.proto",
        "@org_xprof//plugin/xprof/protobuf:op_stats.proto",
        "@org_xprof//plugin/xprof/protobuf:tf_stats.proto",
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
        "@org_xprof//plugin/xprof/protobuf:flat_op_metrics.proto",
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
        # Google well-known types
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:wrappers_proto",
        "@com_google_protobuf//:empty_proto",
    ],
    outs = ["proto_files.tar"],
    cmd = """
        # Create a temporary staging directory
        STAGING=$$(mktemp -d)
        mkdir -p $$STAGING/xla
        mkdir -p $$STAGING/xla/pjrt/proto
        mkdir -p $$STAGING/xla/service
        mkdir -p $$STAGING/xla/service/cpu
        mkdir -p $$STAGING/xla/backends/cpu/runtime
        mkdir -p $$STAGING/xla/stream_executor/cuda
        mkdir -p $$STAGING/xla/stream_executor/gpu
        mkdir -p $$STAGING/xla/stream_executor/sycl
        mkdir -p $$STAGING/xla/backends/autotuner
        mkdir -p $$STAGING/xla/tsl/protobuf
        mkdir -p $$STAGING/tsl/profiler/protobuf
        mkdir -p $$STAGING/google/protobuf
        mkdir -p $$STAGING/plugin/xprof/protobuf

        # Process each source file, preserving directory structure
        for src in $(SRCS); do
            # Route to correct directory based on path patterns
            if echo "$$src" | grep -q "google/protobuf/"; then
                # Handle well-known types from @com_google_protobuf
                mkdir -p $$STAGING/google/protobuf
                cp $$src $$STAGING/google/protobuf/
            elif echo "$$src" | grep -q "xla/service/cpu/"; then
                cp $$src $$STAGING/xla/service/cpu/
            elif echo "$$src" | grep -q "xla/service/"; then
                cp $$src $$STAGING/xla/service/
            elif echo "$$src" | grep -q "xla/backends/cpu/runtime/"; then
                cp $$src $$STAGING/xla/backends/cpu/runtime/
            elif echo "$$src" | grep -q "xla/backends/cpu/"; then
                cp $$src $$STAGING/xla/backends/cpu/
            elif echo "$$src" | grep -q "xla/pjrt/proto/"; then
                cp $$src $$STAGING/xla/pjrt/proto/
            elif echo "$$src" | grep -q "xla/stream_executor/cuda/"; then
                cp $$src $$STAGING/xla/stream_executor/cuda/
            elif echo "$$src" | grep -q "xla/stream_executor/gpu/"; then
                cp $$src $$STAGING/xla/stream_executor/gpu/
            elif echo "$$src" | grep -q "xla/stream_executor/sycl/"; then
                cp $$src $$STAGING/xla/stream_executor/sycl/
            elif echo "$$src" | grep -q "xla/backends/autotuner/"; then
                cp $$src $$STAGING/xla/backends/autotuner/
            elif echo "$$src" | grep -q "xla/stream_executor/"; then
                cp $$src $$STAGING/xla/stream_executor/
            elif echo "$$src" | grep -q "xla/tsl/protobuf/"; then
                cp $$src $$STAGING/xla/tsl/protobuf/
            elif echo "$$src" | grep -q "/xla/xla/"; then
                cp $$src $$STAGING/xla/
            elif echo "$$src" | grep -q "/xla:"; then
                cp $$src $$STAGING/xla/
            elif echo "$$src" | grep -q "tsl/profiler/"; then
                cp $$src $$STAGING/tsl/profiler/protobuf/
            elif echo "$$src" | grep -q "xprof/protobuf/"; then
                cp $$src $$STAGING/plugin/xprof/protobuf/
            else
                # Fallback: put in root
                cp $$src $$STAGING/
            fi
        done

        # Patch xla/autotune_results.proto to be package autotune_results
        [ -f $$STAGING/xla/autotune_results.proto ] && sed -i 's/package xla;/package xla_autotuning;/g' $$STAGING/xla/autotune_results.proto
        [ -f $$STAGING/xla/autotuning.proto ] && sed -i 's/package xla;/package xla_autotuning;/g' $$STAGING/xla/autotuning.proto
        [ -f $$STAGING/xla/tsl/protobuf/dnn.proto ] && sed -i 's/package stream_executor.dnn;/package xla_tsl_dnn;/g' $$STAGING/xla/tsl/protobuf/dnn.proto
        [ -f $$STAGING/xla/stream_executor/gpu/blas_lt_matmul_algorithm.proto ] && sed -i 's/package stream_executor.gpu;/package stream_executor_gpu;/g' $$STAGING/xla/stream_executor/gpu/blas_lt_matmul_algorithm.proto
        [ -f $$STAGING/xla/stream_executor/cuda/cublas_lt_matmul_algorithm.proto ] && sed -i 's/package stream_executor.cuda;/package stream_executor_cuda;/g' $$STAGING/xla/stream_executor/cuda/cublas_lt_matmul_algorithm.proto

        # Update usages of stream_executor.{dnn,gpu,cuda} to avoid cyclic top-level
        # package dependencies between stream_executor and xla_autotuning
        find $$STAGING -name "*.proto" -type f -exec sed -i \
            -e 's/stream_executor\\.dnn/xla_tsl_dnn/g' \
            -e 's/stream_executor\\.gpu/stream_executor_gpu/g' \
            -e 's/stream_executor\\.cuda/stream_executor_cuda/g' \
            {} +

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

        # Create tar archive
        tar -C $$STAGING -cf $@ .

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
        "-DREACTANT_TOOLCHAIN_IS_BB",
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

# TODO(#2247) distinguish between clang and gcc toolchains?
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
        # TODO(#2247) cxx_builtin_include_directories doesn't seem to be working, so we add the INCLUDE_PATHs manually
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "-isystem /opt/BB_TARGET/BB_TARGET/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "-stdlib=libstdc++",
        "-DREACTANT_TOOLCHAIN_IS_BB",
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
    # TODO(#2247) gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
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

# TODO(#2247) distinguish between clang and gcc toolchains?
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
        # TODO(#2247) cxx_builtin_include_directories doesn't seem to be working, so we add the INCLUDE_PATHs manually
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "-isystem /opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "-isystem /opt/BB_TARGET/BB_TARGET/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "-isystem /opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
        "-DREACTANT_TOOLCHAIN_IS_BB",
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
    # TODO(#2247) gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
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

# TODO(#2247) distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_darwin_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-stdlib=libc++",
        "-DREACTANT_TOOLCHAIN_IS_BB",
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

# TODO(#2247) distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_darwin_arm64_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-stdlib=libc++",
        "-DREACTANT_TOOLCHAIN_IS_BB",
    ],
    compiler = "clang",
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

# TODO(#2247) distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_win_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    compile_flags = [
        "-stdlib=libstdc++",
        "-DREACTANT_TOOLCHAIN_IS_BB",
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

alias(
    name = "libReactantExtra.so",
    actual = "@enzyme_ad//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so",
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
    name = "ComplexJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Complex.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Complex/IR/ComplexOps.td",
    deps = [
        "@llvm-project//mlir:ComplexOpsTdFiles",
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
    name = "ImpulseJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Impulse.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@enzyme//:Enzyme/MLIR/Dialect/Impulse/ImpulseOps.td",
    deps = [
        "@enzyme//:ImpulseDialectTdFiles",
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
    name = "CUDATileJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "CUDATile.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@cuda_tile//:include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
    deps = [
        "@cuda_tile//:CudaTileTdFiles",
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
    td_file = "@xla//xla/mosaic:dialect/tpu/tpu.td",
    deps = [
        "@xla//xla/mosaic:tpu_td_files",
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
    td_file = "@triton//:include/triton/Dialect/Triton/IR/TritonOps.td",
    deps = [
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

gentbl_cc_library(
    name = "TensorJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Tensor.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Tensor/IR/TensorOps.td",
    deps = [
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "ShapeJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Shape.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeOps.td",
    deps = [
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "CommJLIncGen",
    tbl_outs = [
        (
            [
                "--generator=jl-op-defs",
                "--disable-module-wrap=0",
            ],
            "Comm.jl",
        ),
    ],
    tblgen = "//:mlir-jl-tblgen",
    td_file = "@enzyme_ad//src/enzyme_ad/jax/Dialect/Comm:Ops.td",
    deps = [
        "@enzyme_ad//src/enzyme_ad/jax/Dialect/Comm:CommDialectTdFiles",
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
        "@enzyme_ad//src/enzyme_ad/jax:Integrations/c/triton_dialect_capi.h",
        "@xla//xla/mosaic:dialect/tpu/integrations/c/tpu_dialect.h",
        # "@xla//xla/mosaic:dialect/tpu/integrations/c/tpu_passes.capi.h.inc",
        "@jax//jaxlib/mosaic/dialect/gpu:integrations/c/attributes.h",
        "@jax//jaxlib/mosaic/dialect/gpu:integrations/c/gpu_dialect.h",
        "@enzyme_ad//src/enzyme_ad/jax:Integrations/c/EnzymeXLA.h",
        "@enzyme//:Enzyme/MLIR/Integrations/c/EnzymeMLIR.h",
        "@enzyme_ad//src/enzyme_ad/jax/ReactantExtra:API.cpp",
        "@enzyme_ad//src/enzyme_ad/jax/ReactantExtra:xla_ffi.cpp",
        "//:Project.toml",
        "//:Manifest.toml",
        "//:wrap.toml",
        "//:missing_defs.jl",
        "//:make.jl",
    ],
    outs = ["libMLIR_h.jl"],
    cmd = "$$JULIA \"--color=yes\" \"--project=$(location //:Project.toml)\" \"$(location //:make.jl)\" \"--extract-api\" \"$(location @llvm-project//mlir:include/mlir-c/Bindings/Python/Interop.h)\" \"$(location @llvm-project//llvm:include/llvm-c/Support.h)\" \"$(locations @llvm-project//mlir:ConversionPassIncGen_filegroup)\" \"$(location @stablehlo//:stablehlo/integrations/c/StablehloAttributes.h)\" \"$(location @shardy//shardy/integrations/c:attributes.h)\" \"$(location @enzyme_ad//src/enzyme_ad/jax:Integrations/c/triton_dialect_capi.h)\" \"$(location @xla//xla/mosaic:dialect/tpu/integrations/c/tpu_dialect.h)\" \"$(location @jax//jaxlib/mosaic/dialect/gpu:integrations/c/gpu_dialect.h)\" \"$(location @enzyme_ad//src/enzyme_ad/jax:Integrations/c/EnzymeXLA.h)\" \"$(location @enzyme//:Enzyme/MLIR/Integrations/c/EnzymeMLIR.h)\" \"$@\"",
    tags = [
        "jlrule",
    ],
)
