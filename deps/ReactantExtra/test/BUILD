load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

cc_test(
    name = "address_reservation_test",
    srcs = [
        "address_reservation_test.cpp",
        "//:libReactantExtra.so",
    ],
    data = ["//:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": "."},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "linkable_runtime_test",
    srcs = [
        "linkable_runtime_test.cpp",
        "reactant_extra_test.h",
        "//:libReactantExtra.so",
    ],
    data = ["//:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": "."},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "pjrt_ifrt_cpu_test",
    srcs = [
        "pjrt_ifrt_cpu_test.cpp",
        "reactant_extra_test.h",
        "//:libReactantExtra.so",
    ],
    data = ["//:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": "."},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "xla_ffi_test",
    srcs = [
        "reactant_extra_test.h",
        "xla_ffi_test.cpp",
        "//:libReactantExtra.so",
    ],
    data = ["//:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": "."},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "mlir_hlo_utils_test",
    srcs = [
        "mlir_hlo_utils_test.cpp",
        "reactant_extra_test.h",
        "//:libReactantExtra.so",
    ],
    data = ["//:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": "."},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)
