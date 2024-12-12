"""Loads the libcxxwrap_julia library."""

def libcxxwrap_julia_deps():
    # TODO change this to download the real artifacts or build them from source
    native.new_local_repository(
        name = "libcxxwrap_julia",
        path = "/Users/mofeing/.julia/artifacts/6a1f8b0d254a485be750499b732b476ddbee44c5/",
        build_file = "//third_party/libcxxwrap_julia:libcxxwrap_julia.BUILD",
    )
