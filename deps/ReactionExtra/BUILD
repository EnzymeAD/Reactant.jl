licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)


cc_library(
    name = "ReactionExtra",
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