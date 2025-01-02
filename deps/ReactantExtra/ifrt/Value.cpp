#include "../type_conversion.hpp"
#include "xla/python/ifrt/value.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::Client* ifrt_value_client(ifrt::Value* value)
{
    return value->client();
}

extern "C" ifrt::Future<> ifrt_value_get_ready_future(ifrt::Value* value)
{
    return value->GetReadyFuture();
}

extern "C" ifrt::Future<> ifrt_value_delete(ifrt::Value* value)
{
    return value->Delete();
}

extern "C" bool ifrt_value_is_deleted(ifrt::Value* value)
{
    return value->IsDeleted();
}

extern "C" const char* ifrt_value_debug_string(ifrt::Value* value)
{
    return cstr_from_string(value->DebugString());
}