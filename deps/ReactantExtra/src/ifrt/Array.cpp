#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/ifrt/array.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" DType* ifrt_array_dtype(Array* array)
{
    return new DType(array->dtype());
}

extern "C" const Shape* ifrt_array_shape(Array* array)
{
    return new Shape(array->shape());
}

extern "C" const Sharding* ifrt_array_sharding(Array* array)
{
    return capture_shared(array->shared_ptr_sharding());
}

extern "C" xla::PjRtLayout* ifrt_array_layout(Array* array)
{
    return capture_shared(MyValueOrThrow(array->layout()));
}

extern "C" span<Array*> ifrt_array_disassemble_into_single_device_arrays(Array* array, ArrayCopySemantics copy_semantics, SingleDeviceShardSemantics shard_semantics)
{
    auto arrays = MyValueOrThrow(array->DisassembleIntoSingleDeviceArrays(copy_semantics, shard_semantics));
    return convert(Type<span<Array*>>(), arrays);
}

extern "C" Array* ifrt_array_fully_replicated_shard(Array* array, ArrayCopySemantics copy_semantics)
{
    return MyValueOrThrow(array->FullyReplicatedShard(copy_semantics)).release();
}

extern "C" Future<>* ifrt_array_copy_to_host_buffer(Array* array, void* data, const int64_t* byte_strides, int semantics)
{
    return new Future<>(array->CopyToHostBuffer(
        data,
        absl::Span<const int64_t>(byte_strides, array->shape().num_elements()),
        ArrayCopySemantics(semantics)));
}
