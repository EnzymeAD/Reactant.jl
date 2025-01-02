#include "type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtTuple* ifrt_pjrt_tuple_ctor(PjRtCompatibleClient* client, Value* values, int nvalues)
{
    auto values_ptr = new tsl::RCReference<Value>[nvalues];
    for (int i = 0; i < nvalues; i++) {
        values_ptr[i] = tsl::RCReference<Value>();
        values_ptr[i].reset(&values[i]);
    }
    auto span = absl::Span<tsl::RCReference<Value>>(values_ptr, nvalues);
    return MyValueOrThrow(PjRtTuple::Create(client, span)).release();
}

extern "C" void ifrt_pjrt_tuple_free(PjRtTuple* tuple) { delete tuple; }
