#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtTuple* ifrt_pjrt_tuple_ctor(PjRtCompatibleClient* client, span<Value*> c_values)
{
    auto values = convert(Type<absl::Span<tsl::RCReference<Value>>>(), c_values);
    return capture_rcreference(MyValueOrThrow(PjRtTuple::Create(client, values)));
}

extern "C" void ifrt_pjrt_tuple_free(PjRtTuple* tuple) { delete tuple; }
