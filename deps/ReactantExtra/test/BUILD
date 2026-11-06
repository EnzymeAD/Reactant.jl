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
