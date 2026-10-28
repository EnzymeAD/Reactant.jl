# Copy of rules_ml_toolchain's gpu/cuda/build_templates/cuda_cudart.BUILD.tpl
# (at ML_TOOLCHAIN_COMMIT 30ef4a9096f9490e8f198faa5ce5bbddd1b72fdb), always
# linking the shared cudart. Used through the `cuda_cudart` override in WORKSPACE.
licenses(["restricted"])  # NVIDIA proprietary license
load(
     "@local_config_cuda//cuda:build_defs.bzl",
     "if_cuda_newer_than",
)
load(
    "@rules_ml_toolchain//gpu:nvidia_common_rules.bzl",
    "cuda_rpath_flags",
)

exports_files([
    "include/cuda.h",
])

filegroup(
    name = "static",
    srcs = ["lib/libcudart_static.a"],
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
%{multiline_comment}
cc_import(
    name = "cuda_stub",
    interface_library = "lib/stubs/libcuda.so",
    system_provided = 1,
)

cc_import(
    name = "cudart_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcudart.so.%{libcudart_version}",
)

cc_import(
    name = "cudart_static_library",
    hdrs = [":headers"],
    static_library = "lib/libcudart_static.a",
)

cc_import(
    name = "culibos_static_library",
    hdrs = [":headers"],
    static_library = if_cuda_newer_than("13_0", None, "lib/libculibos.a"),
)

cc_import(
    name = "cudadevrt_static_library",
    hdrs = [":headers"],
    static_library = "lib/libcudadevrt.a",
)
%{multiline_comment}
cc_library(
    name = "cuda_driver",
    %{comment}deps = [":cuda_stub"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart",
    %{comment}deps = select({
        %{comment}"@cuda_driver//:cuda_umd_libs": [
            %{comment}"@cuda_driver//:nvidia_driver",
            %{comment}"@cuda_driver//:nvidia_ptxjitcompiler",
        %{comment}],
        %{comment}"//conditions:default": [":cuda_driver"],
    # Always link the shared cudart (Reactant.jl): kernels jitted by lower-jit
    # register their fat binaries with the CUDA runtime, which does not work
    # with the static cudart.
    %{comment}}) + [":cudart_shared_library"],
    %{comment}linkopts = if_cuda_newer_than(
        %{comment}"13_0",
        %{comment}if_true = cuda_rpath_flags("nvidia/cu13/lib"),
        %{comment}if_false = cuda_rpath_flags("nvidia/cuda_runtime/lib"),
    %{comment}),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "header_list",
    %{comment}srcs = glob([
        %{comment}"include/builtin_types.h",
        %{comment}"include/channel_descriptor.h",
        %{comment}"include/common_functions.h",
        %{comment}"include/cooperative_groups/**",
        %{comment}"include/cooperative_groups.h",
        %{comment}"include/cuComplex.h",
        %{comment}"include/cuda.h",
        %{comment}"include/cudaEGL.h",
        %{comment}"include/cudaEGLTypedefs.h",
        %{comment}"include/cudaGL.h",
        %{comment}"include/cudaGLTypedefs.h",
        %{comment}"include/cudaProfilerTypedefs.h",
        %{comment}"include/cudaTypedefs.h",
        %{comment}"include/cudaVDPAU.h",
        %{comment}"include/cudaVDPAUTypedefs.h",
        %{comment}"include/cuda_awbarrier.h",
        %{comment}"include/cuda_awbarrier_helpers.h",
        %{comment}"include/cuda_awbarrier_primitives.h",
        %{comment}"include/cuda_bf16.h",
        %{comment}"include/cuda_bf16.hpp",
        %{comment}"include/cuda_device_runtime_api.h",
        %{comment}"include/cuda_egl_interop.h",
        %{comment}"include/cuda_fp16.h",
        %{comment}"include/cuda_fp16.hpp",
        %{comment}"include/cuda_fp4.h",
        %{comment}"include/cuda_fp4.hpp",
        %{comment}"include/cuda_fp6.h",
        %{comment}"include/cuda_fp6.hpp",
        %{comment}"include/cuda_fp8.h",
        %{comment}"include/cuda_fp8.hpp",
        %{comment}"include/cuda_gl_interop.h",
        %{comment}"include/cuda_occupancy.h",
        %{comment}"include/cuda_pipeline.h",
        %{comment}"include/cuda_pipeline_helpers.h",
        %{comment}"include/cuda_pipeline_primitives.h",
        %{comment}"include/cuda_runtime.h",
        %{comment}"include/cuda_runtime_api.h",
        %{comment}"include/cuda_surface_types.h",
        %{comment}"include/cuda_texture_types.h",
        %{comment}"include/cuda_vdpau_interop.h",
        %{comment}"include/cudart_platform.h",
        %{comment}"include/device_atomic_functions.h",
        %{comment}"include/device_atomic_functions.hpp",
        %{comment}"include/device_double_functions.h",
        %{comment}"include/device_functions.h",
        %{comment}"include/device_launch_parameters.h",
        %{comment}"include/device_types.h",
        %{comment}"include/driver_functions.h",
        %{comment}"include/driver_types.h",
        %{comment}"include/host_config.h",
        %{comment}"include/host_defines.h",
        %{comment}"include/library_types.h",
        %{comment}"include/math_constants.h",
        %{comment}"include/math_functions.h",
        %{comment}"include/mma.h",
        %{comment}"include/nvfunctional",
        %{comment}"include/sm_20_atomic_functions.h",
        %{comment}"include/sm_20_atomic_functions.hpp",
        %{comment}"include/sm_20_intrinsics.h",
        %{comment}"include/sm_20_intrinsics.hpp",
        %{comment}"include/sm_30_intrinsics.h",
        %{comment}"include/sm_30_intrinsics.hpp",
        %{comment}"include/sm_32_atomic_functions.h",
        %{comment}"include/sm_32_atomic_functions.hpp",
        %{comment}"include/sm_32_intrinsics.h",
        %{comment}"include/sm_32_intrinsics.hpp",
        %{comment}"include/sm_35_atomic_functions.h",
        %{comment}"include/sm_35_intrinsics.h",
        %{comment}"include/sm_60_atomic_functions.h",
        %{comment}"include/sm_60_atomic_functions.hpp",
        %{comment}"include/sm_61_intrinsics.h",
        %{comment}"include/sm_61_intrinsics.hpp",
        %{comment}"include/surface_functions.h",
        %{comment}"include/surface_indirect_functions.h",
        %{comment}"include/surface_types.h",
        %{comment}"include/texture_fetch_functions.h",
        %{comment}"include/texture_indirect_functions.h",
        %{comment}"include/texture_types.h",
        %{comment}"include/vector_functions.h",
        %{comment}"include/vector_functions.hpp",
        %{comment}"include/vector_types.h",
    %{comment}]),
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)

cc_library(
    name = "headers",
    hdrs = [ ":header_list" ],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
