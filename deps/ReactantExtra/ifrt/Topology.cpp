#include "../type_conversion.hpp"
#include "xla/python/ifrt/topology.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" const char* ifrt_topology_platform_name(ifrt::Topology* topology)
{
    return cstr_from_string(topology->platform_name());
}

extern "C" const char*
ifrt_topology_platform_version(ifrt::Topology* topology)
{
    return cstr_from_string(topology->platform_version());
}

// returns PjRtPlatformId which is a type alias for uint64_t
extern "C" uint64_t ifrt_topology_platform_id(ifrt::Topology* topology)
{
    return topology->platform_id();
}

extern "C" std::tuple<size_t, const xla::PjRtDeviceDescription**>
ifrt_topology_device_descriptions(ifrt::Topology* topology)
{
    auto descriptions = topology->DeviceDescriptions();
    auto descriptions_ptr = new const xla::PjRtDeviceDescription*[descriptions.size()];
    for (int i = 0; i < descriptions.size(); i++) {
        descriptions_ptr[i] = descriptions[i].release();
    }
    return std::make_tuple(descriptions.size(), descriptions_ptr);
}

// TODO xla::ifrt::Topology::GetDefaultLayout

extern "C" const char* ifrt_topology_serialize(ifrt::Topology* topology)
{
    return cstr_from_string(MyValueOrThrow(topology->Serialize()));
}

// TODO xla::ifrt::Topology::Attributes
