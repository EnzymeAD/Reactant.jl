load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")
# load("//toolchain:yggdrasil.bzl", "ygg_cc_toolchain")
licenses(["notice"])

load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_toolchain_suite(
    name = "ygg_cross_compile_toolchain_suite",
    toolchains = {
        "k8": ":ygg_x86_toolchain",
        "darwin": ":ygg_x86_toolchain",
        "darwin_arm64": ":ygg_x86_toolchain",
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

cc_toolchain_config(
    name = "ygg_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/opt/x86_64-linux-musl/bin/../x86_64-linux-musl/sys-root",
    compile_flags = [
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
        "-stdlib=libstdc++"
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
    toolchain_identifier = "ygg_x86_toolchain",
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

# TODO change it to more generic yggdrasil toolchain once we have it
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
    toolchain_identifier = "ygg_toolchain",
)

# TODO distinguish between clang and gcc toolchains?
cc_toolchain_config(
    name = "ygg_aarch64_toolchain_config",
    cpu = "aarch64",
    compiler = "compiler",
    toolchain_identifier = "ygg_aarch64",
    target_system_name = "aarch64-unknown-linux-gnu",
    target_libc = "",
    abi_libc_version = "local",
    abi_version = "local",
    cxx_builtin_include_directories = [
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include",
        "/opt/BB_TARGET/lib/gcc/BB_TARGET/GCC_VERSION/include-fixed",
        "/opt/BB_TARGET/BB_TARGET/include",
        "/opt/BB_TARGET/BB_TARGET/sys-root/usr/include",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/BB_TARGET",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/backward",
        "/opt/BB_TARGET/BB_TARGET/include/c++/GCC_VERSION/parallel",
    ],
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
        "g++": "/opt/bin/BB_FULL_TARGET/g++",
        "gcc": "/opt/bin/BB_FULL_TARGET/gcc",
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
    builtin_sysroot = "/opt/BB_TARGET/BB_TARGET/sys-root/",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    host_system_name = "linux",
    # TODO gcc doesn't support it, only put it on clang (maybe even only for clang on aarch64-darwin?)
    supports_start_end_lib = False,
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

cc_library(
    name = "ReactantExtraLib",
    srcs = glob(
        [
            "*.cpp",
        ],

    ) + [
        "@enzyme_ad//src/enzyme_ad/jax:RegistryUtils.cpp",
        "@enzyme_ad//src/enzyme_ad/jax:gpu.cc",
        "@enzyme_ad//src/enzyme_ad/jax:cpu.cc",
        # "@com_google_protobuf//:src/google/protobuf/io/coded_stream.cc",
        # "@xla//xla:xla.pb.cc",
        "@xla//xla:xla_data.pb.cc",
        # "@xla//xla/stream_executor:device_description.pb.cc",
        "@xla//xla/service:hlo.pb.cc",
        #  # "@tsl//tsl/protobuf:dnn.pb.cc",
        #"@tsl//tsl/protobuf:histogram.pb.cc",
        #"@tsl//tsl/protobuf:bfc_memory_map_proto",bfc_memory_map.pb.cc",
        "@xla//xla/service/gpu:backend_configs.pb.cc",
        "@xla//xla:autotuning.pb.cc",
        "@xla//xla:autotune_results.pb.cc",
        "@xla//xla/service:buffer_assignment.pb.cc",
        ],
    hdrs = glob([
        "*.h",
    ]),
    copts = [
        "-Werror=unused-variable",
        "-Werror=unused-but-set-variable",
        "-Werror=return-type",
        "-Werror=unused-result",
        "-Wno-error=stringop-truncation"
    ] + select({
    "@xla//xla/tsl:is_cuda_enabled_and_oss":[
        "-DREACTANT_CUDA=1",
    ],
    "//conditions:default": [],
    }),
    alwayslink = True,
    linkstatic = True,
    linkopts = select({
    "//conditions:default": [],
    "@bazel_tools//src/conditions:darwin": [
"-Wl,-exported_symbol,_stablehlo*",
"-Wl,-exported_symbol,_mlir*",
"-Wl,-exported_symbol,_sdy*",
"-Wl,-exported_symbol,_EnzymeJaXMapSymbol",
"-Wl,-exported_symbol,_InitializeLogs",
"-Wl,-exported_symbol,_SetLogLevel",
"-Wl,-exported_symbol,_SetModuleLogLevel",
"-Wl,-exported_symbol,_GetDefaultTargetTriple",
"-Wl,-exported_symbol,_enzymeActivityAttrGet",
"-Wl,-exported_symbol,_MakeCPUClient",
"-Wl,-exported_symbol,_MakeGPUClient",
"-Wl,-exported_symbol,_MakeTPUClient",
"-Wl,-exported_symbol,_LoadPjrtPlugin",
"-Wl,-exported_symbol,_InitializePjrtPlugin",
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
"-Wl,-exported_symbol,_free_ifrt_hlo_sharding",
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
    ]}),
    deps = [
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
        # "@enzyme_ad//src/enzyme_ad/jax:gpu",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/ffi:ffi_api",
        "@stablehlo//:chlo_ops",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/pjrt/distributed:distributed",
        "@xla//xla/pjrt/distributed:client",
        "@xla//xla/pjrt/distributed:service",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",

        "@xla//xla:xla_proto_cc",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/stream_executor:device_description_proto_cc_impl",

        "@xla//xla/tsl/platform/default:platform_port",

        "@xla//xla/service:metrics_proto_cc",
        "@xla//xla/service:metrics_proto_cc_impl",

        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service/cpu:cpu_compiler",
        "@xla//xla/stream_executor/tpu:tpu_on_demand_compiler",
        "@xla//xla/stream_executor/tpu:tpu_executor",
        "@xla//xla/stream_executor/tpu:tpu_transfer_manager",

        "@xla//xla/service/cpu:cpu_transfer_manager",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",

        "@xla//xla/tsl/protobuf:protos_all_cc_impl",
        "@xla//xla/tsl/framework:allocator_registry_impl",

        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python/ifrt:ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_ifrt",
        "@xla//xla/python/ifrt_proxy/server:grpc_server",
        "@xla//xla/python/ifrt_proxy/client:grpc_client",
        "@xla//xla/python/ifrt_proxy/client:registry",
        # "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        # "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/ifrt/ir:ifrt_ir_program",
        "@xla//xla/ffi:call_frame",
        "@com_google_protobuf//:protobuf",

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
        "@xla//xla/backends/cpu/collectives:mpi_collectives",

        "@tsl//tsl/platform:env_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/mlir/utils:type_util",
        "@stablehlo//:stablehlo_capi_objects",
        "@stablehlo//:chlo_capi_objects",
        "@shardy//shardy/integrations/c:sdy_capi_objects",
        "@com_google_absl//absl/hash:hash",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log:globals",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:CAPILLVMObjects",
        "@jax//jaxlib/mosaic:tpu_dialect_capi_objects",
        "@jax//jaxlib/triton:triton_dialect_capi_objects",
        "@xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl",
    ] + select({
        "@xla//xla/tsl:is_cuda_enabled_and_oss":[
            "@xla//xla/stream_executor/cuda:all_runtime",
            "@xla//xla/service/gpu/model:hlo_op_profiles",
            "@xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
            "@xla//xla/service/gpu:nvptx_compiler",
            "@xla//xla/service/gpu:gpu_transfer_manager",
            "@xla//xla/stream_executor:kernel",
            "@xla//xla/backends/profiler/gpu:device_tracer",
        ],
        "//conditions:default": [
        ],
    }) + if_rocm([
        "@xla//xla/service/gpu:amdgpu_compiler",
        "@xla//xla/backends/profiler/gpu:device_tracer",
    ]) + select({
        # gloo tcp transport only builds on linux
        "@xla//xla/tsl:macos": [
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
            "@gloo//:transport_uv",
        ],
        "@xla//xla/tsl:windows": [],
        "//conditions:default": [
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
            "@gloo//:transport_tcp",
        ],
    }),
)

# cc_shared_library(
cc_binary(
    name = "libReactantExtra.so",
    linkshared = 1,     ## important
    linkstatic = 1,     ## important
    deps = [":ReactantExtraLib"],
)

cc_binary(
    name = "mlir-jl-tblgen",
    srcs = ["//tblgen:mlir-jl-tblgen.cc", "//tblgen:jl-generators.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//mlir:TableGen",
    ],
    tags = [
        "optional"
    ],
)

gentbl_cc_library(
    name = "BuiltinJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Builtin.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/IR/BuiltinOps.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "ArithJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Arith.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Arith/IR/ArithOps.td",
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "AffineJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Affine.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Affine/IR/AffineOps.td",
    deps = [
        "@llvm-project//mlir:AffineOpsTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "FuncJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Func.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Func/IR/FuncOps.td",
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "LlvmJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Llvm.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/LLVMIR/LLVMOps.td",
    deps = [
        "@llvm-project//mlir:LLVMOpsTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "MemRefJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "MemRef.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/MemRef/IR/MemRefOps.td",
    deps = [
        "@llvm-project//mlir:MemRefOpsTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "NvvmIncJLGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Nvvm.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/LLVMIR/NVVMOps.td",
    deps = [
        "@llvm-project//mlir:NVVMOpsTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "GpuIncJLGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Gpu.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/GPU/IR/GPUOps.td",
    deps = [
        "@llvm-project//mlir:GPUOpsTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "MPIIncJLGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "MPI.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/MPI/IR/MPIOps.td",
    deps = [
        "@llvm-project//mlir:MPITdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "EnzymeJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Enzyme.jl"
        )
    ],
    td_file = "@enzyme//:Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [
        "@enzyme//:EnzymeDialectTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "EnzymeXLAJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "EnzymeXLA.jl"
        )
    ],
    td_file = "@enzyme_ad//src/enzyme_ad/jax:Dialect/EnzymeXLAOps.td",
    deps = [
        "@enzyme//:EnzymeDialectTdFiles",
        "@enzyme_ad//src/enzyme_ad/jax:EnzymeXLADialectTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "TPUJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "TPU.jl"
        )
    ],
    td_file = "@jax//jaxlib/mosaic:dialect/tpu/tpu.td",
    deps = [
        "@jax//jaxlib/mosaic:tpu_td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "TritonJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Triton.jl"
        )
    ],
    td_file = "@jax//jaxlib/triton:triton.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@triton//:td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "StableHLOJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "StableHLO.jl"
        )
    ],
    td_file = "@stablehlo//:stablehlo/dialect/StablehloOps.td",
    deps = [
        "@stablehlo//:stablehlo_ops_td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "CHLOJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "CHLO.jl"
        )
    ],
    td_file = "@stablehlo//:stablehlo/dialect/ChloOps.td",
    deps = [
        "@stablehlo//:chlo_ops_td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "VHLOJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "VHLO.jl"
        )
    ],
    td_file = "@stablehlo//:stablehlo/dialect/VhloOps.td",
    deps = [
        "@stablehlo//:vhlo_ops_td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "ShardyJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Shardy.jl"
        )
    ],
    td_file = "@shardy//shardy/dialect/sdy/ir:ops.td",
    deps = [
        "@shardy//shardy/dialect/sdy/ir:sdy_td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
    includes = ["external/shardy"],
)

genrule(
    name = "libMLIR_h.jl",
    tags = [
        "jlrule"
    ],
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
        "@shardy//shardy/integrations/c:attributes.h",
        "//:Project.toml",
        "//:Manifest.toml",
        "//:wrap.toml",
        "//:missing_defs.jl",
        "//:make.jl"
    ],
    outs = ["libMLIR_h.jl"],
    cmd = "$$JULIA \"--project=$(location //:Project.toml)\" \"$(location //:make.jl)\" \"$(location @llvm-project//mlir:include/mlir-c/Bindings/Python/Interop.h)\" \"$(location @llvm-project//llvm:include/llvm-c/Support.h)\" \"$(locations @llvm-project//mlir:ConversionPassIncGen_filegroup)\" \"$(location @stablehlo//:stablehlo/integrations/c/StablehloAttributes.h)\" \"$(location @shardy//shardy/integrations/c:attributes.h)\" \"$@\"",
)
