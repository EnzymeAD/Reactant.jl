#include "src/type_conversion.hpp"
#include "xla/python/ifrt/host_callback.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" const char* ifrt_hostcallback_serialize(HostCallback* host_callback)
{
    return convert(Type<const char*>(), host_callback->Serialize());
}
