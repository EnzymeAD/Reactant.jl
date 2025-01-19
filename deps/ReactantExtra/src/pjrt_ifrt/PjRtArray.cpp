#include "src/type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO constructors / `Create`

extern "C" std::tuple<size_t, xla::PjRtBuffer* const*> ifrt_pjrt_array_pjrt_buffers(PjRtArray* array) {
    auto buffers = array->pjrt_buffers();
    auto buffers_ptr = new xla::PjRtBuffer*[buffers.size()];
    for (int i = 0; i < buffers.size(); i++) {
        buffers_ptr[i] = buffers[i].get();
    }
    return std::make_tuple(buffers.size(), buffers_ptr);
}
