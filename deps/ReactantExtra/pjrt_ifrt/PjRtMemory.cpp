#include "type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtMemory* ifrt_pjrt_memory_ctor(PjRtClient* client, xla::PjRtMemorySpace* memory_space)
{
    return new PjRtMemory(client, memory_space);
}

extern "C" void ifrt_pjrt_memory_free(PjRtMemory* memory)
{
    delete memory;
}

extern "C" PjRtClient* ifrt_pjrt_memory_client(PjRtMemory* memory)
{
    return memory->client();
}

extern "C" xla::PjRtMemorySpace* ifrt_pjrt_memory_space(PjRtMemory* memory)
{
    return memory->pjrt_memory();
}
