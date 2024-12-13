"""Loads julia."""

def repo():
    # TODO change this to download the real artifacts or build them from source?
    native.new_local_repository(
        name = "julia",
        path = "/Users/mofeing/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/",
        build_file = "//third_party/julia:julia.BUILD",
    )
