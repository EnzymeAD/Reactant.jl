#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/topology.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" const char* ifrt_topology_platform_name(Topology* topology)
{
    return convert(Type<const char*>(), topology->platform_name());
}

extern "C" const char* ifrt_topology_platform_version(Topology* topology)
{
    return convert(Type<const char*>(), topology->platform_version());
}

// returns PjRtPlatformId which is a type alias for uint64_t
extern "C" uint64_t ifrt_topology_platform_id(Topology* topology)
{
    return topology->platform_id();
}

extern "C" span<const xla::PjRtDeviceDescription*> ifrt_topology_device_descriptions(Topology* topology)
{
    auto descriptions = topology->DeviceDescriptions();
    auto descriptions_ptr = new const xla::PjRtDeviceDescription*[descriptions.size()];
    for (int i = 0; i < descriptions.size(); i++) {
        descriptions_ptr[i] = descriptions[i].release();
    }
    return span(descriptions.size(), descriptions_ptr);
}

// TODO xla::Topology::GetDefaultLayout

extern "C" const char* ifrt_topology_serialize(Topology* topology)
{
    return convert(Type<const char*>(), MyValueOrThrow(topology->Serialize()));
}

// TODO xla::Topology::Attributes
