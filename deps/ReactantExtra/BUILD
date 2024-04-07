load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)


cc_library(
    name = "ReactantExtra",
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
        "@enzyme_ad//src/enzyme_ad/jax:XLADerivatives"
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
