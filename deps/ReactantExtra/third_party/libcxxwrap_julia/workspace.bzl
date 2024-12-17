"""Loads the libcxxwrap_julia library."""

def repo():
    # TODO change this to download the real artifacts or build them from source
    native.new_local_repository(
        name = "libcxxwrap_julia",
        path = "/Users/mofeing/.julia/artifacts/4997cdb1f8db7f55d750afcad5db88e3bb4a7819/",
        build_file = "//third_party/libcxxwrap_julia:libcxxwrap_julia.BUILD",
    )
