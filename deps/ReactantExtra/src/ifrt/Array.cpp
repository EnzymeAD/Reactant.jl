#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/array.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" DType* ifrt_array_dtype(Array* array)
{
    return new DType(array->dtype());
}

extern "C" const Shape* ifrt_array_shape(Array* array)
{
    return &(array->shape());
}

extern "C" const Sharding* ifrt_array_sharding(Array* array)
{
    return &(array->sharding());
}

// TODO now it returns a `shared_ptr<PjRtLayout>`
// extern "C" xla::PjRtLayout* ifrt_array_layout(Array* array)
// {
//     return MyValueOrThrow(array->layout()).release();
// }

// TODO xla::Array::DisassembleIntoSingleDeviceArrays
// TODO xla::Array::FullyReplicatedShard

extern "C" Future<> ifrt_array_copy_to_host_buffer(Array* array, void* data, const int64_t* byte_strides, int semantics)
{
    return array->CopyToHostBuffer(
        data,
        absl::Span<const int64_t>(byte_strides, array->shape().num_elements()),
        ArrayCopySemantics(semantics));
}
