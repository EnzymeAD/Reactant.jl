#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/ifrt/array.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" DType* ifrt_array_dtype(Holded<tsl::RCReference<xla::ifrt::Array>>* array)
{
    return new DType((*array)->dtype());
}

extern "C" const Shape* ifrt_array_shape(Holded<tsl::RCReference<xla::ifrt::Array>>* array)
{
    return new Shape((*array)->shape());
}

extern "C" Holded<std::shared_ptr<const Sharding>>* ifrt_array_sharding(Holded<tsl::RCReference<xla::ifrt::Array>>* array)
{
    return capture((*array)->shared_ptr_sharding());
}

extern "C" Holded<std::shared_ptr<const xla::PjRtLayout>>* ifrt_array_layout(Holded<tsl::RCReference<xla::ifrt::Array>>* array)
{
    return capture(MyValueOrThrow((*array)->layout()));
}

extern "C" span<Holded<tsl::RCReference<xla::ifrt::Array>>*> ifrt_array_disassemble_into_single_device_arrays(Holded<tsl::RCReference<xla::ifrt::Array>>* array, ArrayCopySemantics copy_semantics, SingleDeviceShardSemantics shard_semantics)
{
    auto arrays = MyValueOrThrow((*array)->DisassembleIntoSingleDeviceArrays(copy_semantics, shard_semantics));
    return convert(Type<span<Holded<tsl::RCReference<xla::ifrt::Array>>*>>(), arrays);
}

extern "C" Holded<tsl::RCReference<xla::ifrt::Array>>* ifrt_array_fully_replicated_shard(Holded<tsl::RCReference<xla::ifrt::Array>>* array, ArrayCopySemantics copy_semantics)
{
    return capture(MyValueOrThrow((*array)->FullyReplicatedShard(copy_semantics)));
}

extern "C" Future<>* ifrt_array_copy_to_host_buffer(Holded<tsl::RCReference<xla::ifrt::Array>>* array, void* data, const int64_t* byte_strides, int semantics)
{
    return new Future<>((*array)->CopyToHostBuffer(
        data,
        absl::Span<const int64_t>(byte_strides, (*array)->shape().num_elements()),
        ArrayCopySemantics(semantics)));
}
