#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/device.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" Client* ifrt_device_client(Device* device)
{
    return device->client();
}

// DeviceId is a struct with a single int32_t field --> check out
// xla/pjrt/pjrt_common.h
extern "C" DeviceId ifrt_device_id(Device* device)
{
    return device->Id();
}

// TODO ifrt_device_attributes

extern "C" const char* ifrt_device_kind(Device* device)
{
    return convert(Type<const char*>(), device->Kind());
}

extern "C" const char* ifrt_device_to_string(Device* device)
{
    return convert(Type<const char*>(), device->ToString());
}

extern "C" const char* ifrt_device_debug_string(Device* device)
{
    return convert(Type<const char*>(), device->DebugString());
}

extern "C" Memory* ifrt_device_default_memory(Device* device)
{
    return MyValueOrThrow(device->DefaultMemory());
}

// TODO ifrt_device_memories

extern "C" bool ifrt_device_is_addressable(Device* device)
{
    return device->IsAddressable();
}

extern "C" int ifrt_device_process_index(Device* device)
{
    return device->ProcessIndex();
}
