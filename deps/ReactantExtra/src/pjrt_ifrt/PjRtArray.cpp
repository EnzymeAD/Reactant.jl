#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO constructors / `Create`

extern "C" MemoryKind* ifrt_pjrt_make_memory_kind_from_pjrt_buffer(xla::PjRtBuffer* pjrt_buffer) {
    return new MemoryKind(MakeMemoryKindFromPjRtBuffer(pjrt_buffer));
}

extern "C" PjRtArray* ifrt_pjrt_array_create_shape(PjRtCompatibleClient* client, DType* dtype, Shape* shape, Sharding* sharding, span<xla::PjRtBuffer*> c_pjrt_buffers) {
    // auto sharding_shared = reactant::get
}

extern "C" span<xla::PjRtBuffer*> ifrt_pjrt_array_pjrt_buffers(PjRtArray* array) {
    auto buffers = array->pjrt_buffers();
    auto ptr = new xla::PjRtBuffer*[buffers.size()];
    for (int i = 0; i < buffers.size(); i++) {
        ptr[i] = reactant::capture_shared(buffers[i]);
    }
    return span(buffers.size(), ptr);
}

extern "C" span<xla::PjRtBuffer*> ifrt_pjrt_array_mutable_pjrt_buffers(PjRtArray* array) {
    auto buffers = MyValueOrThrow(array->mutable_pjrt_buffers());
    auto ptr = new xla::PjRtBuffer*[buffers.size()];
    for (int i = 0; i < buffers.size(); i++) {
        ptr[i] = reactant::capture_shared(buffers[i]);
    }
    return span(buffers.size(), ptr);
}
