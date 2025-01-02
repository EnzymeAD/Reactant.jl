#include "../type_conversion.hpp"
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
    return cstr_from_string(MyValueOrThrow(host_callback->Serialize()));
}
