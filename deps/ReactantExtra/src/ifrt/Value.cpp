#include "src/type_conversion.hpp"
#include "xla/python/ifrt/value.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" Client* ifrt_value_client(Value* value)
{
    return value->client();
}

extern "C" Future<>* ifrt_value_get_ready_future(Value* value)
{
    return new Future<>(value->GetReadyFuture());
}

extern "C" Future<>* ifrt_value_delete(Value* value)
{
    return new Future<>(value->Delete());
}

extern "C" bool ifrt_value_is_deleted(Value* value)
{
    return value->IsDeleted();
}

extern "C" const char* ifrt_value_debug_string(Value* value)
{
    return convert(Type<const char*>(), value->DebugString());
}
