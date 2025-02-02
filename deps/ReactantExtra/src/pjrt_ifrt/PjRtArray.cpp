#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

using namespace xla::ifrt;
using namespace reactant;

namespace reactant {
template <>
auto convert(Type<xla::Array::PjRtBuffers>, span<xla::PjRtBuffer*> span) -> PjRtArray::PjRtBuffers {
    xla::PjRtBuffers buffers;
    for (int i = 0; i < span.size(); i++) {
        buffers.push_back(reactant::get_shared(span[i]));
    }
    return buffers;
}
}  // namespace reactant

extern "C" PjRtArray* ifrt_pjrt_array_create_general_static(PjRtCompatibleClient* client, DType* dtype, Shape* shape, Sharding* c_sharding, span<xla::PjRtBuffer*> c_pjrt_buffers) {
    auto buffers = convert(Type<xla::Array::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture_rcreference(MyValueOrThrow(PjRtArray(client, *dtype, *shape, reactant::get_shared(c_sharding), buffers)));
}

extern "C" PjRtArray* ifrt_pjrt_array_create_general_dynamic(PjRtCompatibleClient* client, DType* dtype, DynamicShape* shape, Sharding* c_sharding, span<xla::PjRtBuffer*> c_pjrt_buffers) {
    auto buffers = convert(Type<xla::Array::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture_rcreference(MyValueOrThrow(PjRtArray(client, *dtype, *shape, reactant::get_shared(c_sharding), buffers)));
}

extern "C" PjRtArray* ifrt_pjrt_array_create_shard_single(PjRtCompatibleClient* client, xla::PjRtBuffer* c_pjrt_buffers) {
    auto buffers = convert(Type<xla::Array::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture_rcreference(MyValueOrThrow(PjRtArray(client, *dtype, *shape, reactant::get_shared(c_sharding), buffers)));
}

extern "C" PjRtArray* ifrt_pjrt_array_create_shard_concrete_static(PjRtCompatibleClient* client, Shape* shape, span<xla::PjRtBuffer*> c_pjrt_buffers) {
    auto buffers = convert(Type<xla::Array::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture_rcreference(MyValueOrThrow(PjRtArray(client, *shape, buffers)));
}

extern "C" PjRtArray* ifrt_pjrt_array_create_shard_concrete_dynamic(PjRtCompatibleClient* client, DynamicShape* shape, span<xla::PjRtBuffer*> c_pjrt_buffers) {
    auto buffers = convert(Type<xla::Array::PjRtBuffers>(), c_pjrt_buffers);
    return reactant::capture_rcreference(MyValueOrThrow(PjRtArray(client, *shape, buffers)));
}

extern "C" void ifrt_pjrt_array_dtor(PjRtArray* array) {
    reactant::destruct_or_release_if_rcreference(array);
}

extern "C" MemoryKind* ifrt_pjrt_make_memory_kind_from_pjrt_buffer(xla::PjRtBuffer* pjrt_buffer) {
    return new MemoryKind(MakeMemoryKindFromPjRtBuffer(pjrt_buffer));
}

// NOTE we just implement `mutable_pjrt_buffers` and not `pjrt_buffers` because the only difference is `const`ness which is not passed through C API
extern "C" span<xla::PjRtBuffer*> ifrt_pjrt_array_pjrt_buffers(PjRtArray* array) {
    auto buffers = array->mutable_pjrt_buffers();
    auto ptr = new xla::PjRtBuffer*[buffers.size()];
    for (int i = 0; i < buffers.size(); i++) {
        ptr[i] = reactant::capture_shared(buffers[i]);
    }
    return span(buffers.size(), ptr);
}
