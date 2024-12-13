# TODO libcxxwrap_julia has LICENSE.md file in share/licenses/libcxxwrap_julia/LICENSE.md
licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "libcxxwrap_julia",
    hdrs = glob(["include/jlcxx/*.hpp"]),
    shared_library = "lib/libcxxwrap_julia.dylib",
    visibility = ["//visibility:public"],
)
