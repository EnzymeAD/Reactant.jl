#include "type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_device.h"

using namespace xla::ifrt;
using namespace reactant;

// DeviceId is a struct with a single int32_t field --> check out
// xla/pjrt/pjrt_common.h
// TODO support `attributes` parameter
extern "C" PjRtDevice* ifrt_pjrt_device_ctor(
    PjRtClient* client,
    DeviceId device_id,
    const char* kind,
    const char* to_string,
    const char* debug_string,
    int process_index,
    xla::PjRtDevice* pjrt_device)
{
    return new PjRtDevice(
        client, device_id, kind, to_string, debug_string, process_index,
        absl::flat_hash_map<std::string, PjRtDeviceAttribute>(), pjrt_device);
}

extern "C" void ifrt_pjrt_device_free(PjRtDevice* device)
{
    delete device;
}

extern "C" xla::PjRtDevice* ifrt_pjrt_device_pjrt_device(PjRtDevice* device)
{
    return device->pjrt_device();
}
