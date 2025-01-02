#include "../type_conversion.hpp"
#include "xla/python/ifrt/memory.h"

using namespace xla::ifrt;
using namespace reactant;

// MemoryId is a struct with a single int32_t field --> check out
// xla/python/ifrt/memory.h
extern "C" ifrt::MemoryId ifrt_memory_id(ifrt::Memory* memory)
{
    return memory->Id();
}

extern "C" const ifrt::MemoryKind* ifrt_memory_kind(ifrt::Memory* memory)
{
    return &(memory->Kind());
}

extern "C" const char* ifrt_memory_to_string(ifrt::Memory* memory)
{
    return cstr_from_string(memory->ToString());
}

extern "C" const char* ifrt_memory_debug_string(ifrt::Memory* memory)
{
    return cstr_from_string(memory->DebugString());
}

extern "C" span<ifrt::Device*>
ifrt_memory_devices(ifrt::Memory* memory)
{
    return convert(Type<span<ifrt::Device*>>, memory->Devices());
}
