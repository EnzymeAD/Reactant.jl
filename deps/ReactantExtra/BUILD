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
		"gcc": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-clang++",
		"cpp": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-clang++",
		"ld": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-ld.lld",
        #"gcc": "/opt/x86_64-linux-musl/bin/clang",
        #"ld": "/opt/x86_64-linux-musl/bin/ld.lld",
        #"cpp": "/opt/x86_64-linux-musl/bin/clang++",
        "ar": "/opt/bin/x86_64-linux-musl-cxx11/x86_64-linux-musl-ar",
        "llvm-cov": "/usr/lib/llvm-17/bin/llvm-cov",
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
        # "@tsl//tsl/protobuf:dnn.pb.cc",
        "@tsl//tsl/protobuf:histogram.pb.cc",
        "@tsl//tsl/protobuf:bfc_memory_map.pb.cc",
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
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:Transforms",
        "@enzyme_ad//src/enzyme_ad/jax:TransformOps",
        "@enzyme_ad//src/enzyme_ad/jax:XLADerivatives",
        "@stablehlo//:chlo_ops",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python/ifrt:ifrt",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/ffi:call_frame",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/platform:env_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/mlir/utils:type_util",
        "@stablehlo//:stablehlo_capi",
        "@com_google_absl//absl/hash:hash"
    ],
)

cc_shared_library(
    name = "ReactantExtra",
    deps = [":ReactantExtraLib",
        "@enzyme_ad//src/enzyme_ad/jax:XLADerivatives",
        "@llvm-project//mlir:CAPIIR",
        "@stablehlo//:stablehlo_capi",
    ],
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
