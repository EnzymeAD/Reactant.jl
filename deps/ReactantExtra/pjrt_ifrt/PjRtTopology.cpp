#include "src/type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtTopology* ifrt_pjrt_topology_ctor(const xla::PjRtTopologyDescription* description)
{
    return new PjRtTopology(std::shared_ptr<const xla::PjRtTopologyDescription> { description });
}

extern "C" void ifrt_pjrt_topology_free(PjRtTopology* topology) { delete topology; }

extern "C" const xla::PjRtTopologyDescription* ifrt_pjrt_topology_description(PjRtTopology* topology)
{
    return topology->description().get();
}
