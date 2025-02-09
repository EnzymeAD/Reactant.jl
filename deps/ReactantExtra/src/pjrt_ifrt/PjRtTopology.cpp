#include "src/type_conversion.hpp"
#include "src/memory_management.hpp"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtTopology* ifrt_pjrt_topology_ctor(Holded<std::shared_ptr<const xla::PjRtTopologyDescription>>* out_holded_description, const xla::PjRtTopologyDescription* c_description)
{
    auto description = std::shared_ptr<const xla::PjRtTopologyDescription>(c_description);
    (*out_holded_description) = *capture(description);
    return new PjRtTopology(description);
}

extern "C" void ifrt_pjrt_topology_dtor(PjRtTopology* topology) { delete topology; }

extern "C" Holded<std::shared_ptr<const xla::PjRtTopologyDescription>>* ifrt_pjrt_topology_description(PjRtTopology* topology)
{
    return capture(topology->description());
}
