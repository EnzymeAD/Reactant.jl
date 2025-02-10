#include "src/type_conversion.hpp"
#include "xla/python/ifrt/memory.h"

using namespace xla::ifrt;
using namespace reactant;

// MemoryId is a struct with a single int32_t field --> check out
// xla/python/ifrt/memory.h
extern "C" MemoryId ifrt_memory_id(Memory* memory)
{
    return memory->Id();
}

extern "C" const MemoryKind* ifrt_memory_kind(Memory* memory)
{
    return &(memory->Kind());
}

extern "C" const char* ifrt_memory_to_string(Memory* memory)
{
    return convert(Type<const char*>(), memory->ToString());
}

extern "C" const char* ifrt_memory_debug_string(Memory* memory)
{
    return convert(Type<const char*>(), memory->DebugString());
}

extern "C" span<Device*> ifrt_memory_devices(Memory* memory)
{
    return convert(Type<span<Device*>>(), memory->Devices());
}
