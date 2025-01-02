#include "../type_conversion.hpp"
#include "xla/python/ifrt/device.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::Client* ifrt_device_client(ifrt::Device* device)
{
    return device->client();
}

// DeviceId is a struct with a single int32_t field --> check out
// xla/pjrt/pjrt_common.h
extern "C" ifrt::DeviceId ifrt_device_id(ifrt::Device* device)
{
    return device->Id();
}

// TODO ifrt_device_attributes

extern "C" const char* ifrt_device_kind(ifrt::Device* device)
{
    return cstr_from_string(device->Kind());
}

extern "C" const char* ifrt_device_to_string(ifrt::Device* device)
{
    return cstr_from_string(device->ToString());
}

extern "C" const char* ifrt_device_debug_string(ifrt::Device* device)
{
    return cstr_from_string(device->DebugString());
}

extern "C" ifrt::Memory* ifrt_device_default_memory(ifrt::Device* device)
{
    return MyValueOrThrow(device->DefaultMemory());
}

// TODO ifrt_device_memories

extern "C" bool ifrt_device_is_addressable(ifrt::Device* device)
{
    return device->IsAddressable();
}

extern "C" int ifrt_device_process_index(ifrt::Device* device)
{
    return device->ProcessIndex();
}
