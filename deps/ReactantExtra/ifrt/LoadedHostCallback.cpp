#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/host_callback.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" Client* ifrt_loadedhostcallback_client(LoadedHostCallback* host_callback)
{
    return host_callback->client();
}

extern "C" const char* ifrt_loadedhostcallback_serialize(LoadedHostCallback* host_callback)
{
    // auto msg = ;
    return convert(Type<const char*>(), MyValueOrThrow(host_callback->Serialize()));
}
