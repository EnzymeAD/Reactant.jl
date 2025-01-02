#include "../type_conversion.hpp"
#include "xla/python/ifrt/array.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::DType* ifrt_array_dtype(ifrt::Array* array)
{
    return new ifrt::DType(array->dtype());
}

extern "C" const ifrt::Shape* ifrt_array_shape(ifrt::Array* array)
{
    return &(array->shape());
}

extern "C" const ifrt::Sharding* ifrt_array_sharding(ifrt::Array* array)
{
    return &(array->sharding());
}

extern "C" PjRtLayout* ifrt_array_layout(ifrt::Array* array)
{
    return MyValueOrThrow(array->layout()).release();
}

// TODO xla::ifrt::Array::DisassembleIntoSingleDeviceArrays
// TODO xla::ifrt::Array::FullyReplicatedShard

extern "C" ifrt::Future<>
ifrt_array_copy_to_host_buffer(ifrt::Array* array, void* data,
    const int64_t* byte_strides, int semantics)
{
    return array->CopyToHostBuffer(
        data,
        absl::Span<const int64_t>(byte_strides, array->shape().num_elements()),
        ifrt::ArrayCopySemantics(semantics));
}
