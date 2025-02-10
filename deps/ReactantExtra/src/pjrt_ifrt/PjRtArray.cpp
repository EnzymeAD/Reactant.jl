#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

using namespace xla::ifrt;
using namespace reactant;

namespace reactant {
auto convert(Type<PjRtArray::PjRtBuffers>, span<Holded<std::shared_ptr<xla::PjRtBuffer>>*> span) -> PjRtArray::PjRtBuffers {
    PjRtArray::PjRtBuffers buffers;
    for (int i = 0; i < span.size(); i++) {
        buffers.push_back(span[i]->value());
    }
    return buffers;
}
}  // namespace reactant

extern "C" Holded<tsl::RCReference<PjRtArray>>* ifrt_pjrt_array_create_general_static(PjRtCompatibleClient* client, DType* dtype, Shape* shape, Holded<std::shared_ptr<const Sharding>>* c_sharding, span<Holded<std::shared_ptr<xla::PjRtBuffer>>*> c_pjrt_buffers) {
    auto buffers = convert(Type<PjRtArray::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture(MyValueOrThrow(PjRtArray::Create(client, *dtype, *shape, c_sharding->value(), buffers)));
}

extern "C" Holded<tsl::RCReference<PjRtArray>>* ifrt_pjrt_array_create_general_dynamic(PjRtCompatibleClient* client, DType* dtype, DynamicShape* shape, Holded<std::shared_ptr<const Sharding>>* c_sharding, span<Holded<std::shared_ptr<xla::PjRtBuffer>>*> c_pjrt_buffers) {
    auto buffers = convert(Type<PjRtArray::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture(MyValueOrThrow(PjRtArray::Create(client, *dtype, *shape, c_sharding->value(), buffers)));
}

extern "C" Holded<tsl::RCReference<PjRtArray>>* ifrt_pjrt_array_create_shard_single(PjRtCompatibleClient* client, Holded<std::shared_ptr<xla::PjRtBuffer>>* c_pjrt_buffer) {
    return reactant::capture(MyValueOrThrow(PjRtArray::Create(client, c_pjrt_buffer->value())));
}

extern "C" Holded<tsl::RCReference<PjRtArray>>* ifrt_pjrt_array_create_shard_concrete_static(PjRtCompatibleClient* client, Shape* shape, span<Holded<std::shared_ptr<xla::PjRtBuffer>>*> c_pjrt_buffers) {
    auto buffers = convert(Type<PjRtArray::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture(MyValueOrThrow(PjRtArray::Create(client, *shape, buffers)));
}

extern "C" Holded<tsl::RCReference<PjRtArray>>* ifrt_pjrt_array_create_shard_concrete_dynamic(PjRtCompatibleClient* client, DynamicShape* shape, span<Holded<std::shared_ptr<xla::PjRtBuffer>>*> c_pjrt_buffers) {
    auto buffers = convert(Type<PjRtArray::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture(MyValueOrThrow(PjRtArray::Create(client, *shape, buffers)));
}

extern "C" void ifrt_pjrt_array_dtor(Holded<tsl::RCReference<PjRtArray>>* array) {
    delete array;
}

// TODO this takes pointer but in other places we use a `Holded`?
// extern "C" MemoryKind* ifrt_pjrt_make_memory_kind_from_pjrt_buffer(xla::PjRtBuffer* pjrt_buffer) {
//     return new MemoryKind(MakeMemoryKindFromPjRtBuffer(pjrt_buffer));
// }

// NOTE we just implement `mutable_pjrt_buffers` and not `pjrt_buffers` because the only difference is `const`ness which is not passed through C API
extern "C" span<Holded<std::shared_ptr<xla::PjRtBuffer>>*> ifrt_pjrt_array_pjrt_buffers(Holded<tsl::RCReference<PjRtArray>>* array) {
    auto buffers = MyValueOrThrow((*array)->mutable_pjrt_buffers());
    auto ptr = new Holded<std::shared_ptr<xla::PjRtBuffer>>*[buffers.size()];
    for (int i = 0; i < buffers.size(); i++) {
        ptr[i] = reactant::capture(buffers[i]);
    }
    return span(buffers.size(), ptr);
}

extern "C" bool ifrt_pjrt_array_has_static_shape(Holded<tsl::RCReference<PjRtArray>>* array) {
    return (*array)->has_static_shape();
}

extern "C" bool ifrt_pjrt_array_has_dynamic_shape(Holded<tsl::RCReference<PjRtArray>>* array) {
    return (*array)->has_dynamic_shape();
}

extern "C" DynamicShape* ifrt_pjrt_array_dynamic_shape(Holded<tsl::RCReference<PjRtArray>>* array) {
    return new DynamicShape((*array)->dynamic_shape());
}

extern "C" Holded<tsl::RCReference<Array>>* ifrt_pjrt_array_copy(Holded<tsl::RCReference<PjRtArray>>* array, span<xla::ifrt::Device*> c_devices, MemoryKind* c_memory_kind, ArrayCopySemantics semantics) {
    std::optional<tsl::RCReference<xla::ifrt::DeviceList>> devices;
    if (!c_devices.empty())
        devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);

    std::optional<xla::ifrt::MemoryKind> memory_kind;
    if (c_memory_kind != nullptr)
        memory_kind = *c_memory_kind;

    return capture(MyValueOrThrow((*array)->Copy(devices, memory_kind, semantics)));
}
