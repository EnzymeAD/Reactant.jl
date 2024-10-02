load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@xla//tools/toolchains/cross_compile/cc:cc_toolchain_config.bzl", "cc_toolchain_config")
licenses(["notice"])

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
        "linux_arm64": ":ygg_x86_toolchain",
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
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include/c++/10.2.0",
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include/c++/10.2.0/x86_64-linux-musl",
        "/opt/x86_64-linux-musl/x86_64-linux-musl/include/c++/10.2.0/backward",
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
    name = "darwin_aarch64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
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
        # "@com_google_protobuf//:src/google/protobuf/io/coded_stream.cc",
        "@xla//xla:xla.pb.cc",
        "@xla//xla:xla_data.pb.cc",
        "@xla//xla/stream_executor:device_description.pb.cc",
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
    ],
    alwayslink = True,
    linkstatic = True,
    linkopts = select({
    "//conditions:default": [],
    "@bazel_tools//src/conditions:darwin": [
"-Wl,-exported_symbol,_stablehlo*",
"-Wl,-exported_symbol,_mlir*",
"-Wl,-exported_symbol,_InitializeLogs",
"-Wl,-exported_symbol,_SetLogLevel",
"-Wl,-exported_symbol,_SetModuleLogLevel",
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
"-Wl,-exported_symbol,_FreeFuture",
"-Wl,-exported_symbol,_FutureIsReady",
"-Wl,-exported_symbol,_FutureAwait",
"-Wl,-exported_symbol,_XLAExecute",
"-Wl,-exported_symbol,_RegisterDialects",
"-Wl,-exported_symbol,_InitializeRegistryAndPasses",
    ]}),
    deps = [
                "@enzyme//:EnzymeMLIR",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@enzyme_ad//src/enzyme_ad/jax:TransformOps",
        "@enzyme_ad//src/enzyme_ad/jax:XLADerivatives",
        "@stablehlo//:chlo_ops",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt/cpu:cpu_client",

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
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/ffi:call_frame",
        "@com_google_protobuf//:protobuf",
	"@xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@xla//xla/tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/platform:env_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/mlir/utils:type_util",
        "@stablehlo//:stablehlo_capi_objects",
        "@stablehlo//:chlo_capi_objects",
        "@com_google_absl//absl/hash:hash",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log:globals",
        "@llvm-project//mlir:CAPIIRObjects",
    ] + select({
    "@xla//xla/tsl:is_cuda_enabled_and_oss":[
        "@xla//xla/stream_executor/cuda:all_runtime", 
        "@xla//xla/stream_executor/rocm:all_runtime",
        "@xla//xla/service/gpu/model:hlo_op_profiles",
        "@xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
        "@xla//xla/service/gpu:nvptx_compiler",
        "@xla//xla/service/gpu:amdgpu_compiler",
        "@xla//xla/service/gpu:gpu_transfer_manager",
        "@xla//xla/stream_executor:kernel", 
    ],
    "//conditions:default": [],
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
            "Builtin.inc.jl"
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
            "Arith.inc.jl"
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
            "Affine.inc.jl"
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
            "Func.inc.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Func/IR/FuncOps.td",
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "EnzymeJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "Enzyme.inc.jl"
        )
    ],
    td_file = "@enzyme//:Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [
        "@enzyme//:EnzymeDialectTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "StableHLOJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap=0"],
            "StableHLO.inc.jl"
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
            "CHLO.inc.jl"
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
            "VHLO.inc.jl"
        )
    ],
    td_file = "@stablehlo//:stablehlo/dialect/VhloOps.td",
    deps = [
        "@stablehlo//:vhlo_ops_td_files",
    ],
    tblgen = "//:mlir-jl-tblgen",
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
        "//:Project.toml",
        "//:Manifest.toml",
        "//:wrap.toml",
        "//:missing_defs.jl",
        "//:make.jl"
    ],
    outs = ["libMLIR_h.jl"],
    cmd = "$$JULIA \"--project=$(location //:Project.toml)\" \"$(location //:make.jl)\" \"$(location @llvm-project//mlir:include/mlir-c/Bindings/Python/Interop.h)\" \"$(location @llvm-project//llvm:include/llvm-c/Support.h)\" \"$(locations @llvm-project//mlir:ConversionPassIncGen_filegroup)\" \"$(location @stablehlo//:stablehlo/integrations/c/StablehloAttributes.h)\" \"$@\"",
)
