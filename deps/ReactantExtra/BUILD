load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)


cc_library(
    name = "ReactantExtraLib",
    srcs = glob(
        [
            "*.cpp",
        ],
    ),
    hdrs = glob([
        "*.h",
    ]),
    copts = [
        "-Werror=unused-variable",
        "-Werror=unused-but-set-variable",
        "-Werror=return-type",
        "-Werror=unused-result",
    ],
    deps = [
        "@enzyme_ad//src/enzyme_ad/jax:XLADerivatives",
        "@llvm-project//mlir:CAPIIR",
    ],
    alwayslink = 1,
)

cc_shared_library(
    name = "ReactantExtra",
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
)

gentbl_cc_library(
    name = "BuiltinJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap"],
            "Buitin.inc.jl"
        )
    ],
    td_file = "@llvm-project//mlir:include/mlir/IR/BuiltinOps.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
    tblgen = "//:mlir-jl-tblgen",
)

gentbl_cc_library(
    name = "AffineJLIncGen",
    tbl_outs = [(
            ["--generator=jl-op-defs", "--disable-module-wrap"],
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
            ["--generator=jl-op-defs", "--disable-module-wrap"],
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
            ["--generator=jl-op-defs", "--disable-module-wrap"],
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
            ["--generator=jl-op-defs", "--disable-module-wrap"],
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
            ["--generator=jl-op-defs", "--disable-module-wrap"],
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
            ["--generator=jl-op-defs", "--disable-module-wrap"],
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
        "//:Project.toml",
        "//:Manifest.toml",
        "//:wrap.toml",
        "//:missing_defs.jl",
        "//:make.jl"
    ],
    outs = ["libMLIR_h.jl"],
    cmd = "$$JULIA \"--project=$(location //:Project.toml)\" \"$(location //:make.jl)\" \"$(location @llvm-project//mlir:include/mlir-c/Bindings/Python/Interop.h)\" \"$(location @llvm-project//llvm:include/llvm-c/Support.h)\" \"$(locations @llvm-project//mlir:ConversionPassIncGen_filegroup)\" \"$@\"",
)